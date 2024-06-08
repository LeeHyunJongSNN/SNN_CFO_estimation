/* -*- c++ -*- */
/*
 * Copyright 2024 gnuradio.org.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUSTOMS_OFDM_CARRIER_PREAMBLE_IMPL_H
#define INCLUDED_CUSTOMS_OFDM_CARRIER_PREAMBLE_IMPL_H

#include <gnuradio/customs/ofdm_carrier_preamble.h>

namespace gr {
namespace customs {

class ofdm_carrier_preamble_impl : public ofdm_carrier_preamble
{
private:
    //! FFT length
    const int d_fft_len;
    //! Which carriers/symbols carry data
    std::vector<std::vector<int>> d_occupied_carriers;
    //! Which carriers/symbols carry pilots symbols
    std::vector<std::vector<int>> d_pilot_carriers;
    //! Value of said pilot symbols
    const std::vector<std::vector<gr_complex>> d_pilot_symbols;
    //! Synch words
    const std::vector<std::vector<gr_complex>> d_sync_words;
    int d_symbols_per_set;
    const bool d_output_is_shifted;

protected:
    int calculate_output_stream_length(const gr_vector_int& ninput_items) override;

public:
    ofdm_carrier_preamble_impl(int fft_len,
                               const std::vector<std::vector<int>>& occupied_carriers,
                               const std::vector<std::vector<int>>& pilot_carriers,
                               const std::vector<std::vector<gr_complex>>& pilot_symbols,
                               const std::vector<std::vector<gr_complex>>& sync_words,
                               const std::string& len_tag_key,
                               const bool output_is_shifted);
    ~ofdm_carrier_preamble_impl() override;

    std::string len_tag_key() override { return d_length_tag_key_str; };

    const int fft_len() override { return d_fft_len; };

    std::vector<std::vector<int>> occupied_carriers() override
    {
        return d_occupied_carriers;
    };

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_int& ninput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace customs
} // namespace gr

#endif /* INCLUDED_CUSTOMS_OFDM_CARRIER_PREAMBLE_IMPL_H */
