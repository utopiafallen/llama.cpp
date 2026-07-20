#pragma once

#include "common.h"

#include <cstdint>
#include <list>
#include <string>
#include <vector>

static const uint32_t SERVER_CKPT_SIDECAR_MAGIC  = 0x4C4C4B43; // "LLKC"
static const uint32_t SERVER_CKPT_SIDECAR_VERSION = 1;

size_t server_ckpt_sidecar_write(const std::string & filepath,
                                 const std::list<common_prompt_checkpoint> & checkpoints);

bool server_ckpt_sidecar_read(const std::string & filepath,
                              std::list<common_prompt_checkpoint> & checkpoints);

std::string server_ckpt_sidecar_path(const std::string & kv_filepath);
