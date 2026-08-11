#pragma once

#include "ggml-backend.h"

#include <vector>

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    5
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API int ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                     size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices, int heartbeat_seconds);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}

// Batch pre-check: ask RPC server which tensors are cached and load them.
// Server loads from cache on hit. Returns vector of uint8_t (1 = loaded, 0 = missing).
// Tensors must already be allocated in RPC buffers.
std::vector<uint8_t> ggml_backend_rpc_batch_precheck(
    ggml_backend_buffer_t buffer,
    const char * file_basename,
    uint64_t file_mtime,
    const ggml_tensor * const * tensors,
    uint32_t tensor_count);

#endif
