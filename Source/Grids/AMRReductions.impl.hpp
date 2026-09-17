/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#if !defined(AMRREDUCTIONS_HPP_)
#error "This file should only be included through AMRReductions.hpp"
#endif

#ifndef AMRREDUCTIONS_IMPL_HPP_
#define AMRREDUCTIONS_IMPL_HPP_

#include "AMRReductions.hpp"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>

namespace AMRReductions
{
// AMREX_D_TERM (rather than dx[0]*dx[0]*dx[0]) is kept deliberately: it makes
// the cell measure correct for whatever AMREX_SPACEDIM the code is compiled
// with, e.g. a future Cartoon-formalism build running on a lower-dimensional
// grid. Note that Cartoon norms would still need an extra symmetry-reduction
// measure factor (e.g. 2*pi*r) on top of this to be physically meaningful.
amrex::Real domainVolume(const amrex::Geometry &a_coarse_geometry)
{
    const auto dx                 = a_coarse_geometry.CellSizeArray();
    const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    return cell_volume * a_coarse_geometry.Domain().d_numPts();
}

amrex::Real volumeWeightedSumOfSquares(
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>> &a_data,
    int a_start_component, int a_num_components,
    const amrex::Vector<amrex::Geometry> &a_geometry,
    const amrex::Vector<amrex::IntVect> &a_refinement_ratios)
{
    AMREX_ALWAYS_ASSERT(!a_data.empty());
    AMREX_ALWAYS_ASSERT(a_data.size() == a_geometry.size());
    // a_refinement_ratios may be sized to amr.max_level rather than the
    // current finest level, so it only needs to be at least as long as
    // required to cover the levels actually present in a_data.
    AMREX_ALWAYS_ASSERT(a_data.size() <= a_refinement_ratios.size() + 1);

    amrex::Real sum = 0.0;
    for (std::size_t level = 0; level < a_data.size(); ++level)
    {
        const auto &data              = *a_data[level];
        const auto dx                 = a_geometry[level].CellSizeArray();
        const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
        const auto &data_arrays       = data.const_arrays();

        if (level + 1 < a_data.size())
        {
            const amrex::iMultiFab mask =
                amrex::makeFineMask(data, *a_data[level + 1], amrex::IntVect(0),
                                    a_refinement_ratios[level],
                                    amrex::Periodicity::NonPeriodic(), 0, 1);
            const auto &mask_arrays = mask.const_arrays();

            const auto reduction = amrex::ParReduce(
                amrex::TypeList<amrex::ReduceOpSum>{},
                amrex::TypeList<amrex::Real>{}, data, amrex::IntVect(0),
                [=] AMREX_GPU_DEVICE(
                    int box_no, int ix, int iy,
                    int iz) noexcept -> amrex::GpuTuple<amrex::Real>
                {
                    if (mask_arrays[box_no](ix, iy, iz) != 0)
                    {
                        return {0.0};
                    }
                    amrex::Real sum_of_squares = 0.0;
                    for (int comp = a_start_component;
                         comp < a_start_component + a_num_components; ++comp)
                    {
                        const amrex::Real value =
                            data_arrays[box_no](ix, iy, iz, comp);
                        sum_of_squares += value * value;
                    }
                    return {cell_volume * sum_of_squares};
                });
            sum += reduction;
        }
        else
        {
            const auto reduction = amrex::ParReduce(
                amrex::TypeList<amrex::ReduceOpSum>{},
                amrex::TypeList<amrex::Real>{}, data, amrex::IntVect(0),
                [=] AMREX_GPU_DEVICE(
                    int box_no, int ix, int iy,
                    int iz) noexcept -> amrex::GpuTuple<amrex::Real>
                {
                    amrex::Real sum_of_squares = 0.0;
                    for (int comp = a_start_component;
                         comp < a_start_component + a_num_components; ++comp)
                    {
                        const amrex::Real value =
                            data_arrays[box_no](ix, iy, iz, comp);
                        sum_of_squares += value * value;
                    }
                    return {cell_volume * sum_of_squares};
                });
            sum += reduction;
        }
    }

    amrex::ParallelAllReduce::Sum(sum,
                                  amrex::ParallelContext::CommunicatorSub());
    return sum;
}

std::pair<amrex::Real, amrex::Real> volumeWeightedSumOfSquares(
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>> &a_data,
    int a_start_component_1, int a_num_components_1, int a_start_component_2,
    int a_num_components_2, const amrex::Vector<amrex::Geometry> &a_geometry,
    const amrex::Vector<amrex::IntVect> &a_refinement_ratios)
{
    AMREX_ALWAYS_ASSERT(!a_data.empty());
    AMREX_ALWAYS_ASSERT(a_data.size() == a_geometry.size());
    AMREX_ALWAYS_ASSERT(a_data.size() <= a_refinement_ratios.size() + 1);

    amrex::Real sum_1 = 0.0;
    amrex::Real sum_2 = 0.0;
    for (std::size_t level = 0; level < a_data.size(); ++level)
    {
        const auto &data              = *a_data[level];
        const auto dx                 = a_geometry[level].CellSizeArray();
        const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
        const auto &data_arrays       = data.const_arrays();

        // Sum the squares of both component ranges in a single ParReduce
        // pass over the grid, since they share the same mask and loop.
        if (level + 1 < a_data.size())
        {
            const amrex::iMultiFab mask =
                amrex::makeFineMask(data, *a_data[level + 1], amrex::IntVect(0),
                                    a_refinement_ratios[level],
                                    amrex::Periodicity::NonPeriodic(), 0, 1);
            const auto &mask_arrays = mask.const_arrays();

            const auto reduction = amrex::ParReduce(
                amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
                amrex::TypeList<amrex::Real, amrex::Real>{}, data,
                amrex::IntVect(0),
                [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy,
                                     int iz) noexcept
                    -> amrex::GpuTuple<amrex::Real, amrex::Real>
                {
                    if (mask_arrays[box_no](ix, iy, iz) != 0)
                    {
                        return {0.0, 0.0};
                    }
                    amrex::Real sum_of_squares_1 = 0.0;
                    for (int comp = a_start_component_1;
                         comp < a_start_component_1 + a_num_components_1;
                         ++comp)
                    {
                        const amrex::Real value =
                            data_arrays[box_no](ix, iy, iz, comp);
                        sum_of_squares_1 += value * value;
                    }
                    amrex::Real sum_of_squares_2 = 0.0;
                    for (int comp = a_start_component_2;
                         comp < a_start_component_2 + a_num_components_2;
                         ++comp)
                    {
                        const amrex::Real value =
                            data_arrays[box_no](ix, iy, iz, comp);
                        sum_of_squares_2 += value * value;
                    }
                    return {cell_volume * sum_of_squares_1,
                            cell_volume * sum_of_squares_2};
                });
            sum_1 += amrex::get<0>(reduction);
            sum_2 += amrex::get<1>(reduction);
        }
        else
        {
            const auto reduction = amrex::ParReduce(
                amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
                amrex::TypeList<amrex::Real, amrex::Real>{}, data,
                amrex::IntVect(0),
                [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy,
                                     int iz) noexcept
                    -> amrex::GpuTuple<amrex::Real, amrex::Real>
                {
                    amrex::Real sum_of_squares_1 = 0.0;
                    for (int comp = a_start_component_1;
                         comp < a_start_component_1 + a_num_components_1;
                         ++comp)
                    {
                        const amrex::Real value =
                            data_arrays[box_no](ix, iy, iz, comp);
                        sum_of_squares_1 += value * value;
                    }
                    amrex::Real sum_of_squares_2 = 0.0;
                    for (int comp = a_start_component_2;
                         comp < a_start_component_2 + a_num_components_2;
                         ++comp)
                    {
                        const amrex::Real value =
                            data_arrays[box_no](ix, iy, iz, comp);
                        sum_of_squares_2 += value * value;
                    }
                    return {cell_volume * sum_of_squares_1,
                            cell_volume * sum_of_squares_2};
                });
            sum_1 += amrex::get<0>(reduction);
            sum_2 += amrex::get<1>(reduction);
        }
    }

    amrex::ParallelAllReduce::Sum(sum_1,
                                  amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Sum(sum_2,
                                  amrex::ParallelContext::CommunicatorSub());
    return {sum_1, sum_2};
}

amrex::Real volumeWeightedL2Norm(
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>> &a_data,
    int a_start_component, int a_num_components,
    const amrex::Vector<amrex::Geometry> &a_geometry,
    const amrex::Vector<amrex::IntVect> &a_refinement_ratios)
{
    const amrex::Real sum_of_squares =
        volumeWeightedSumOfSquares(a_data, a_start_component, a_num_components,
                                   a_geometry, a_refinement_ratios);
    return std::sqrt(sum_of_squares / domainVolume(a_geometry[0]));
}

std::pair<amrex::Real, amrex::Real> volumeWeightedL2Norm(
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>> &a_data,
    int a_start_component_1, int a_num_components_1, int a_start_component_2,
    int a_num_components_2, const amrex::Vector<amrex::Geometry> &a_geometry,
    const amrex::Vector<amrex::IntVect> &a_refinement_ratios)
{
    const auto [sum_of_squares_1, sum_of_squares_2] =
        volumeWeightedSumOfSquares(a_data, a_start_component_1,
                                   a_num_components_1, a_start_component_2,
                                   a_num_components_2, a_geometry,
                                   a_refinement_ratios);
    const amrex::Real domain_volume = domainVolume(a_geometry[0]);
    return {std::sqrt(sum_of_squares_1 / domain_volume),
            std::sqrt(sum_of_squares_2 / domain_volume)};
}
} // namespace AMRReductions

#endif /* AMRREDUCTIONS_IMPL_HPP_ */
