#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda/std/atomic>
#include <cooperative_groups.h>

// Constants for arithmetic coding
const uint32_t CODE_BITS = 32;
const uint32_t TOP_VALUE = (1u << (CODE_BITS - 1));
const uint32_t BOTTOM_VALUE = (TOP_VALUE >> 8);
const uint32_t MAX_FREQ = 16383;  // Must be less than BOTTOM_VALUE

// Structure to hold arithmetic coding state
struct ACState {
    uint32_t low;
    uint32_t high;
    uint32_t code;
    union {
        uint8_t* output;    // For encoding
        const uint8_t* input;  // For decoding
    };
    size_t pos;  // Position in input/output stream
    bool is_encoding;
};

__device__ void init_state(ACState* state, bool is_encoding) {
    state->low = 0;
    state->high = TOP_VALUE;
    state->pos = 0;
    state->is_encoding = is_encoding;
    
    if (!is_encoding) {
        // Initialize decoder state
        state->code = 0;
        // Read first CODE_BITS bits
        for (int i = 0; i < CODE_BITS / 8; i++) {
            state->code = (state->code << 8) | state->input[state->pos++];
        }
    }
}

__device__ void process_symbol(
    ACState* state,
    uint32_t symbol,
    const float* probs,
    uint32_t num_symbols,
    uint32_t* output_symbol
) {
    uint32_t range = state->high - state->low + 1;
    
    if (state->is_encoding) {
        // Encoding path
        float cumul = 0.0f;
        for (uint32_t i = 0; i < symbol; i++) {
            cumul += probs[i];
        }
        
        // Update state
        state->high = state->low + (uint32_t)(range * (cumul + probs[symbol]));
        state->low = state->low + (uint32_t)(range * cumul);
        
        // Renormalization for encoder
        while (true) {
            if (state->high < BOTTOM_VALUE) {
                state->output[state->pos++] = 0;
                state->low <<= 8;
                state->high = (state->high << 8) | 0xFF;
            }
            else if (state->low >= BOTTOM_VALUE) {
                state->output[state->pos++] = 1;
                state->low = (state->low - BOTTOM_VALUE) << 8;
                state->high = ((state->high - BOTTOM_VALUE) << 8) | 0xFF;
            }
            else break;
        }
    } else {
        // Decoding path
        uint32_t count = ((state->code - state->low + 1) * MAX_FREQ - 1) / range;
        
        // Find symbol
        float cumul = 0.0f;
        uint32_t sym;
        for (sym = 0; sym < num_symbols; sym++) {
            cumul += probs[sym];
            if ((uint32_t)(cumul * MAX_FREQ) > count) break;
        }
        *output_symbol = sym;
        
        // Update state
        float prev_cumul = cumul - probs[sym];
        state->high = state->low + (uint32_t)(range * cumul);
        state->low = state->low + (uint32_t)(range * prev_cumul);
        
        // Renormalization for decoder
        while (true) {
            if (state->high < BOTTOM_VALUE) {
                state->low <<= 8;
                state->high = (state->high << 8) | 0xFF;
                state->code = (state->code << 8) | state->input[state->pos++];
            }
            else if (state->low >= BOTTOM_VALUE) {
                state->low = (state->low - BOTTOM_VALUE) << 8;
                state->high = ((state->high - BOTTOM_VALUE) << 8) | 0xFF;
                state->code = ((state->code - BOTTOM_VALUE) << 8) | state->input[state->pos++];
            }
            else break;
        }
    }
}

__global__ void encode_arithmetic_kernel(
    const uint8_t* input,
    const float* probs,
    uint8_t* output,
    size_t input_len,
    uint32_t num_symbols
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= input_len) return;
    
    // Each thread handles one token's worth of data
    ACState state;
    init_state(&state, true);
    state.output = output + tid * sizeof(uint32_t);
    
    // Encode the symbol
    uint32_t dummy;
    process_symbol(&state, input[tid], probs, num_symbols, &dummy);
}

__global__ void decode_arithmetic_kernel(
    const uint8_t* input,
    const float* probs,
    uint8_t* output,
    size_t compressed_len,
    uint32_t num_symbols,
    size_t output_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_len) return;
    
    // Each thread decodes one token
    ACState state;
    init_state(&state, false);
    state.input = input + tid * sizeof(uint32_t);
    
    // Decode the symbol
    uint32_t decoded_symbol;
    process_symbol(&state, 0, probs, num_symbols, &decoded_symbol);
    output[tid] = (uint8_t)decoded_symbol;
}

extern "C" {

// Python-callable function to launch the encoding kernel
void encode_arithmetic(
    const uint8_t* input,
    const float* probs,
    uint8_t* output,
    size_t input_len,
    uint32_t num_symbols,
    cudaStream_t stream = 0
) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (input_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    encode_arithmetic_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, probs, output, input_len, num_symbols
    );
    
    if (stream == 0) {
        // Only synchronize on default stream
        cudaDeviceSynchronize();
    }
}

// Python-callable function to launch the decoding kernel
void decode_arithmetic(
    const uint8_t* input,
    const float* probs,
    uint8_t* output,
    size_t compressed_len,
    uint32_t num_symbols,
    size_t output_len,
    cudaStream_t stream = 0
) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (output_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    decode_arithmetic_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, probs, output, compressed_len, num_symbols, output_len
    );
    
    if (stream == 0) {
        // Only synchronize on default stream
        cudaDeviceSynchronize();
    }
}

} // extern "C"
