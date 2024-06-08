/* -*- c++ -*- */
/*
 * Copyright 2024 gnuradio.org.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ofdm_carrier_preamble_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace customs {

ofdm_carrier_preamble::sptr ofdm_carrier_preamble::make(int fft_len,
                                                        const std::vector<std::vector<int>>& occupied_carriers,
                                                        const std::vector<std::vector<int>>& pilot_carriers,
                                                        const std::vector<std::vector<gr_complex>>& pilot_symbols,
                                                        const std::vector<std::vector<gr_complex>>& sync_words,
                                                        const std::string& len_tag_key,
                                                        const bool output_is_shifted)
{
    return gnuradio::make_block_sptr<ofdm_carrier_preamble_impl>(fft_len,
                                                                 occupied_carriers,
                                                                 pilot_carriers,
                                                                 pilot_symbols,
                                                                 sync_words,
                                                                 len_tag_key,
                                                                 output_is_shifted);
}


/*
 * The private constructor
 */
ofdm_carrier_preamble_impl::ofdm_carrier_preamble_impl(int fft_len,
                                                       const std::vector<std::vector<int>>& occupied_carriers,
                                                       const std::vector<std::vector<int>>& pilot_carriers,
                                                       const std::vector<std::vector<gr_complex>>& pilot_symbols,
                                                       const std::vector<std::vector<gr_complex>>& sync_words,
                                                       const std::string& len_tag_key,
                                                       const bool output_is_shifted)
    : gr::tagged_stream_block(
          "ofdm_carrier_preamble",
          gr::io_signature::make(1, 1, sizeof(gr_complex)),
          gr::io_signature::make(1, 1, sizeof(gr_complex) * fft_len),
          len_tag_key),
      d_fft_len(fft_len),
      d_occupied_carriers(occupied_carriers),
      d_pilot_carriers(pilot_carriers),
      d_pilot_symbols(pilot_symbols),
      d_sync_words(sync_words),
      d_symbols_per_set(0),
      d_output_is_shifted(output_is_shifted)
{
    // Sanity checks
    // If that is is null, the input is wrong -> force user to use ((),) in python
    if (d_occupied_carriers.empty()) {
        throw std::invalid_argument(
                "Occupied carriers must be of type vector of vector i.e. ((),).");
    }
    for (unsigned i = 0; i < d_occupied_carriers.size(); i++) {
        for (unsigned j = 0; j < d_occupied_carriers[i].size(); j++) {
            if (occupied_carriers[i][j] < 0) {
                d_occupied_carriers[i][j] += d_fft_len;
            }
            if (d_occupied_carriers[i][j] > d_fft_len || d_occupied_carriers[i][j] < 0) {
                throw std::invalid_argument("data carrier index out of bounds");
            }
            if (d_output_is_shifted) {
                d_occupied_carriers[i][j] =
                        (d_occupied_carriers[i][j] + fft_len / 2) % fft_len;
            }
        }
    }
    if (d_pilot_carriers.empty()) {
        throw std::invalid_argument(
                "Pilot carriers must be of type vector of vector i.e. ((),).");
    }
    for (unsigned i = 0; i < d_pilot_carriers.size(); i++) {
        for (unsigned j = 0; j < d_pilot_carriers[i].size(); j++) {
            if (d_pilot_carriers[i][j] < 0) {
                d_pilot_carriers[i][j] += d_fft_len;
            }
            if (d_pilot_carriers[i][j] > d_fft_len || d_pilot_carriers[i][j] < 0) {
                throw std::invalid_argument("pilot carrier index out of bounds");
            }
            if (d_output_is_shifted) {
                d_pilot_carriers[i][j] = (d_pilot_carriers[i][j] + fft_len / 2) % fft_len;
            }
        }
    }
    if (d_pilot_symbols.empty()) {
        throw std::invalid_argument(
                "Pilot symbols must be of type vector of vector i.e. ((),).");
    }
    for (unsigned i = 0; i < std::max(d_pilot_carriers.size(), d_pilot_symbols.size());
         i++) {
        if (d_pilot_carriers[i % d_pilot_carriers.size()].size() !=
            d_pilot_symbols[i % d_pilot_symbols.size()].size()) {
            throw std::invalid_argument("pilot_carriers do not match pilot_symbols");
        }
    }
    for (unsigned i = 0; i < d_sync_words.size(); i++) {
        if (d_sync_words[i].size() != (unsigned)d_fft_len) {
            throw std::invalid_argument("sync words must be fft length");
        }
    }

    for (unsigned i = 0; i < d_occupied_carriers.size(); i++) {
        d_symbols_per_set += d_occupied_carriers[i].size();
    }
    set_tag_propagation_policy(TPP_DONT);
    set_relative_rate((uint64_t)d_symbols_per_set, (uint64_t)d_occupied_carriers.size());
}

/*
 * Our virtual destructor.
 */
ofdm_carrier_preamble_impl::~ofdm_carrier_preamble_impl() {}

int ofdm_carrier_preamble_impl::calculate_output_stream_length(
    const gr_vector_int& ninput_items)
{
    int nin = ninput_items[0];
    int nout = (nin / d_symbols_per_set) * d_occupied_carriers.size();
    int k = 0;
    for (int i = 0; i < nin % d_symbols_per_set; k++) {
        nout++;
        i += d_occupied_carriers[k % d_occupied_carriers.size()].size();
    }
    return nout + d_sync_words.size();
}

int ofdm_carrier_preamble_impl::work(int noutput_items,
                                     gr_vector_int& ninput_items,
                                     gr_vector_const_void_star& input_items,
                                     gr_vector_void_star& output_items)
{
    const gr_complex* in = (const gr_complex*)input_items[0];
    gr_complex* out = (gr_complex*)output_items[0];
    std::vector<tag_t> tags;

    memset((void*)out, 0x00, sizeof(gr_complex) * d_fft_len * noutput_items);
    // Copy Sync word
    for (unsigned i = 0; i < d_sync_words.size(); i++) {
        memcpy((void*)out, (void*)&d_sync_words[i][0], sizeof(gr_complex) * d_fft_len);
        out += d_fft_len;
    }

    // Return STS and LTS
    return d_sync_words.size();
}

} /* namespace customs */
} /* namespace gr */
