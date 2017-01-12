#pragma once

#include "ReassignedSpectrum.hpp"
#include "RingBuffer.hpp"

namespace RMWarp {

struct ChannelData {
    RingBuffer<float>   m_fifo_in{};
    RingBuffer<float>   m_fifo_out{};
};

};
