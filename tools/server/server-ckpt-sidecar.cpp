#include "server-ckpt-sidecar.h"

#include <fstream>

static bool write_blob(std::ofstream & f, const std::vector<uint8_t> & data) {
    uint64_t len = static_cast<uint64_t>(data.size());
    f.write(reinterpret_cast<const char *>(&len), sizeof(len));
    if (!data.empty()) {
        f.write(reinterpret_cast<const char *>(data.data()), len);
    }
    return f.good();
}

static bool read_blob(std::ifstream & f, std::vector<uint8_t> & data) {
    uint64_t len = 0;
    f.read(reinterpret_cast<char *>(&len), sizeof(len));
    if (len > 0) {
        data.resize(len);
        f.read(reinterpret_cast<char *>(data.data()), len);
    } else {
        data.clear();
    }
    return f.good();
}

size_t server_ckpt_sidecar_write(const std::string & filepath,
                                 const std::list<common_prompt_checkpoint> & checkpoints) {
    std::ofstream f(filepath, std::ios::binary | std::ios::trunc);
    if (!f.good()) {
        return 0;
    }

    // Header
    uint32_t magic  = SERVER_CKPT_SIDECAR_MAGIC;
    uint32_t version = SERVER_CKPT_SIDECAR_VERSION;
    uint32_t count   = static_cast<uint32_t>(checkpoints.size());

    f.write(reinterpret_cast<const char *>(&magic),   sizeof(magic));
    f.write(reinterpret_cast<const char *>(&version), sizeof(version));
    f.write(reinterpret_cast<const char *>(&count),   sizeof(count));

    // Per checkpoint
    for (const auto & ckpt : checkpoints) {
        int64_t  n_tokens = ckpt.n_tokens;
        int32_t  id_task  = ckpt.id_task;
        int32_t  pos_min  = ckpt.pos_min;
        int32_t  pos_max  = ckpt.pos_max;

        f.write(reinterpret_cast<const char *>(&n_tokens), sizeof(n_tokens));
        f.write(reinterpret_cast<const char *>(&id_task),  sizeof(id_task));
        f.write(reinterpret_cast<const char *>(&pos_min),  sizeof(pos_min));
        f.write(reinterpret_cast<const char *>(&pos_max),  sizeof(pos_max));

        if (!write_blob(f, ckpt.data_tgt))  return 0;
        if (!write_blob(f, ckpt.data_dft))  return 0;
        if (!write_blob(f, ckpt.data_spec)) return 0;
    }

    if (!f.good()) return 0;
    const size_t written = static_cast<size_t>(f.tellp());
    f.close();
    return written;
}

bool server_ckpt_sidecar_read(const std::string & filepath,
                              std::list<common_prompt_checkpoint> & checkpoints) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.good()) {
        return false;
    }

    // Header
    uint32_t magic, version, count;
    f.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
    f.read(reinterpret_cast<char *>(&version), sizeof(version));
    f.read(reinterpret_cast<char *>(&count),   sizeof(count));

    if (magic != SERVER_CKPT_SIDECAR_MAGIC) {
        return false;
    }
    if (version != SERVER_CKPT_SIDECAR_VERSION) {
        return false;
    }

    checkpoints.clear();

    for (uint32_t i = 0; i < count; i++) {
        common_prompt_checkpoint ckpt;

        int64_t  n_tokens;
        int32_t  id_task, pos_min, pos_max;

        f.read(reinterpret_cast<char *>(&n_tokens), sizeof(n_tokens));
        f.read(reinterpret_cast<char *>(&id_task),  sizeof(id_task));
        f.read(reinterpret_cast<char *>(&pos_min),  sizeof(pos_min));
        f.read(reinterpret_cast<char *>(&pos_max),  sizeof(pos_max));

        ckpt.n_tokens = n_tokens;
        ckpt.id_task  = id_task;
        ckpt.pos_min  = pos_min;
        ckpt.pos_max  = pos_max;

        if (!read_blob(f, ckpt.data_tgt))  return false;
        if (!read_blob(f, ckpt.data_dft))  return false;
        if (!read_blob(f, ckpt.data_spec)) return false;

        checkpoints.push_back(std::move(ckpt));
    }

    return true;
}

std::string server_ckpt_sidecar_path(const std::string & kv_filepath) {
    return kv_filepath + ".ckpt";
}
