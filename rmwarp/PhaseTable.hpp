_Pragma("once")

#include "rmwarp/ReFFT.hpp"
#include "rmwarp/ReSpectrum.hpp"

namespace RMWarp {
class PhaseTable {
public:
    using value_type = float;
    using vector_type = simd_vec<value_type>;
    using size_type = vector_type::size_type;
    using difference_type = vector_type::difference_type;
    using allocator_type = bs::allocator<value_type>;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
    static constexpr int value_alignment = simd_alignment<value_type>;
    static constexpr int item_alignment  = value_alignment / sizeof(value_type);

    allocator_type m_alloc{};

};
}
