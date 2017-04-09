#pragma once

#include "ReassignedSpectrum.hpp"
#include "RingBuffer.hpp"

namespace RMWarp {

struct ChannelData {
    int                 m_size{};
    RingBuffer<float>   m_fifo_in{};
    RingBuffer<float>   m_fifo_out{};
    AccBuffer           m_accu{};
    MiniRing<ReassignedSpectrum> m_spectra(16ul, ReassignedSpectrum(m_size));
};
};
