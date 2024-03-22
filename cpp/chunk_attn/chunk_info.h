#pragma once
#include "chunk.h"
#include <torch/torch.h>

namespace GPT {
struct ChunkInfo {
    ChunkInfo() = default;
    ChunkInfo(Chunk* chunk, int seq_idx_begin, int seq_idx_end)
      : chunk(chunk)
      , seq_idx_begin(seq_idx_begin)
      , seq_idx_end(seq_idx_end) {}

    Chunk* chunk;
    int seq_idx_begin;
    int seq_idx_end;
};
}