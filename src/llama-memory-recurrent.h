#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                         bool   no_alloc,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
                         bool   rs_replay,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // differential slot save (seq state v5, see llama-context.cpp)
    //
    // layout of the capacity-sized recurrent section of a v5 slot file:
    //   [u32 cell_count][C x (i64 pos, u32 n_seq_id=0)][u32 s_trans][u32 n_layer]
    //   per layer with r: [i32 r_type][u64 r_row][C rows of r_row bytes]
    //   per layer with p: [u64 p_row][C rows of p_row bytes]
    //   per layer with s: [i32 s_type][u64 s_row][C rows of s_row bytes]
    // cell i sits at row offset i, so a used [0, n) prefix maps to fixed file
    // offsets exactly like the attention K/V rows; the rest of the section is
    // pre-allocated (sparse) and left untouched until a cell becomes valid
    struct save_layout_recr {
        struct layer {
            bool      has_r = false;
            ggml_type r_type = GGML_TYPE_F32;
            uint64_t  r_row  = 0;
            bool      has_p  = false;
            uint64_t  p_row  = 0;
            bool      has_s  = false;
            ggml_type s_type = GGML_TYPE_F32;
            uint64_t  s_row  = 0;
        };

        uint32_t              n_layer = 0; // total model layers (nulls kept, like the v4 stream)
        std::vector<layer>    layers;
        // one meta record per stored cell: (i32 pos, u32 n_seq_id); n_seq_id is
        // 0 and rs_replay is off in the storable case, so the record is 8 bytes
        uint32_t              meta_size = 8;

        // fixed section size (one logical row per layer): the recurrent cache
        // holds a single rolling state per sequence, not one per position, so
        // the section does not scale with the capacity
        uint64_t size() const {
            uint64_t res = 4 + meta_size + 8; // cell_count, meta, desc (s_trans, n_layer)
            for (const auto & l : layers) {
                if (l.has_r) { res += 12 + l.r_row + (l.has_p ? 8 + l.p_row : 0); }
                if (l.has_s) { res += 12 + l.s_row; }
            }
            return res;
        }
    };

    // fill the recurrent section layout; false when the state is not
    // storable in the differential format (v1: rs_replay snapshots,
    // rollback planes, empty cache)
    bool recr_save_layout(uint32_t C, save_layout_recr & out) const;

    // number of cells holding this sequence's recurrent state; the
    // differential format stores exactly one (the rolling tail)
    uint32_t recr_cell_count(llama_seq_id seq_id) const;

    // differential (v5) load of the recurrent section at recr_base; the
    // section is the v4 seq stream verbatim (ccnt, meta, desc, rows) and is
    // fed to state_read(); returns the restored cell count (0 on failure)
    // and extends *hash over the section bytes
    uint32_t state_load_v5(struct llama_file & file, llama_seq_id seq_id,
                           uint64_t * hash, uint64_t recr_base);

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    // deferred commit: keep one state plane plus the inputs of the last ubatch, replay the accepted part in the next ubatch
    bool rs_replay = false;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        // largest rollback whose snapshot the last ubatch guarantees: a ubatch of n <= n_rs_seq + 1 tokens
        // writes snapshots 0..n-1, a longer one only guarantees the final state (ggml_ssm_scan)
        uint32_t rs_max = 0;

        // rs_replay: valid tokens in the pending rows, and the pending tokens the current ubatch replays
        uint32_t rs_pend = 0;
        uint32_t rs_rep  = 0;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;
    // rs_replay pending gdn inputs: k, v, and the gates then the betas
    std::vector<ggml_tensor *> spk_l;
    std::vector<ggml_tensor *> spv_l;
    std::vector<ggml_tensor *> spgb_l;

    // true if no layers - can happen if the layer filter removes all layers
    bool is_empty() const;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the RS cache along with the allocated backend buffers:
    llama_memory_bufs ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;
    ggml_tensor * get_spk_l (int32_t il) const;
    ggml_tensor * get_spv_l (int32_t il) const;
    ggml_tensor * get_spgb_l(int32_t il) const;

    int32_t s_copy(int i) const;

    bool     get_rs_replay() const;
    uint32_t get_rs_rep(int i) const; // pending tokens replayed by ubatch seq i

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
