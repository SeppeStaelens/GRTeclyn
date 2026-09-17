/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef AMRREDUCTIONS_HPP_
#define AMRREDUCTIONS_HPP_

#include <AMReX_Amr.H>
#include <AMReX_REAL.H>

#include <memory>
#include <utility>

namespace AMRReductions
{
/** Return the volume of the coarse-level domain, i.e. the volume the
 *  hierarchy would have if it were unrefined everywhere. */
amrex::Real domainVolume(const amrex::Geometry &a_coarse_geometry);

/** Return the volume integral, over the whole AMR hierarchy, of the sum of
 *  squares of a_num_components components starting at a_start_component. */
amrex::Real volumeWeightedSumOfSquares(
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>> &a_data,
    int a_start_component, int a_num_components,
    const amrex::Vector<amrex::Geometry> &a_geometry,
    const amrex::Vector<amrex::IntVect> &a_refinement_ratios);

/** As above, but computes the sums of squares of two (disjoint) component
 *  ranges in a single pass over the grid. */
std::pair<amrex::Real, amrex::Real> volumeWeightedSumOfSquares(
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>> &a_data,
    int a_start_component_1, int a_num_components_1, int a_start_component_2,
    int a_num_components_2, const amrex::Vector<amrex::Geometry> &a_geometry,
    const amrex::Vector<amrex::IntVect> &a_refinement_ratios);

/** Return the volume-weighted L2 norm, i.e.
 *  sqrt(volumeWeightedSumOfSquares / domainVolume), of a_num_components
 *  components starting at a_start_component. */
amrex::Real volumeWeightedL2Norm(
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>> &a_data,
    int a_start_component, int a_num_components,
    const amrex::Vector<amrex::Geometry> &a_geometry,
    const amrex::Vector<amrex::IntVect> &a_refinement_ratios);

/** As above, but computes the L2 norms of two (disjoint) component ranges in
 *  a single pass over the grid. */
std::pair<amrex::Real, amrex::Real> volumeWeightedL2Norm(
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>> &a_data,
    int a_start_component_1, int a_num_components_1, int a_start_component_2,
    int a_num_components_2, const amrex::Vector<amrex::Geometry> &a_geometry,
    const amrex::Vector<amrex::IntVect> &a_refinement_ratios);
} // namespace AMRReductions

#include "AMRReductions.impl.hpp"

#endif /* AMRREDUCTIONS_HPP_ */
