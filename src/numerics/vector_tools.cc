// ---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2023 by the ideal.II authors
//
// This file is part of the ideal.II library.
//
// The ideal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of ideal.II.
//
// ---------------------------------------------------------------------

#include <ideal.II/fe/spacetime_fe_values.hh>

#include <ideal.II/numerics/vector_tools.hh>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/dofs/dof_tools.h>

namespace idealii::slab::VectorTools
{
  template <int dim, typename Number>
  void
  interpolate_boundary_values(
    DoFHandler<dim>                 &dof_handler,
    const dealii::types::boundary_id boundary_component,
    dealii::Function<dim, Number>   &boundary_function,
    const std::shared_ptr<dealii::AffineConstraints<Number>>
                                &spacetime_constraints,
    const dealii::ComponentMask &component_mask)
  {
    auto space_constraints =
      std::make_shared<dealii::AffineConstraints<Number>>();

    dealii::Quadrature<1> quad_time(
      dof_handler.temporal()->get_fe(0).get_unit_support_points());
    dealii::FEValues<1> fev(dof_handler.temporal()->get_fe(0),
                            quad_time,
                            dealii::update_quadrature_points);

    unsigned int n_space_dofs = dof_handler.n_dofs_space();
    unsigned int time_dof     = 0;
    std::vector<dealii::types::global_dof_index> local_indices(
      dof_handler.dofs_per_cell_time());
    // loop over time cells instead
    for (const auto &cell_time :
         dof_handler.temporal()->active_cell_iterators())
      {
        fev.reinit(cell_time);

        cell_time->get_dof_indices(local_indices);
        for (unsigned int q = 0; q < quad_time.size(); q++)
          {
            space_constraints->clear();
            time_dof = local_indices[q];
            boundary_function.set_time(fev.quadrature_point(q)[0]);
            dealii::VectorTools::interpolate_boundary_values(
              *dof_handler.spatial(),
              boundary_component,
              boundary_function,
              *space_constraints,
              component_mask);

            for (unsigned int i = 0; i < n_space_dofs; i++)
              {
                // check if this is a constrained dof
                if (space_constraints->is_constrained(i))
                  {
                    const std::vector<std::pair<dealii::types::global_dof_index,
                                                double>> *entries =
                      space_constraints->get_constraint_entries(i);

                    spacetime_constraints->add_line(i +
                                                    time_dof * n_space_dofs);
                    // non Dirichlet constraint
                    if (!entries->empty())
                      {
                        for (auto entry : *entries)
                          {
                            spacetime_constraints->add_entry(
                              i + time_dof * n_space_dofs,
                              entry.first + time_dof * n_space_dofs,
                              entry.second);
                          }
                      }
                    else
                      {
                        spacetime_constraints->set_inhomogeneity(
                          i + time_dof * n_space_dofs,
                          space_constraints->get_inhomogeneity(i));
                      }
                  }
              }
          }
      }
  }

  template <int dim, typename Number>
  void
  project_boundary_values_curl_conforming_l2(
    DoFHandler<dim>                 &dof_handler,
    unsigned int                     first_vector_component,
    dealii::Function<dim, Number>   &boundary_function,
    const dealii::types::boundary_id boundary_component,
    const std::shared_ptr<dealii::AffineConstraints<Number>>
      &spacetime_constraints)
  {
    auto space_constraints =
      std::make_shared<dealii::AffineConstraints<Number>>();

    const dealii::Quadrature<1> quad_time(
      dof_handler.temporal()->get_fe(0).get_unit_support_points());
    dealii::FEValues<1> fev(dof_handler.temporal()->get_fe(0),
                            quad_time,
                            dealii::update_quadrature_points);

    unsigned int n_space_dofs = dof_handler.n_dofs_space();
    unsigned int time_dof     = 0;
    std::vector<dealii::types::global_dof_index> local_indices(
      dof_handler.dofs_per_cell_time());
    // loop over time cells instead
    for (const auto &cell_time :
         dof_handler.temporal()->active_cell_iterators())
      {
        fev.reinit(cell_time);

        cell_time->get_dof_indices(local_indices);
        for (unsigned int q = 0; q < quad_time.size(); q++)
          {
            space_constraints->clear();
            time_dof = local_indices[q];
            boundary_function.set_time(fev.quadrature_point(q)[0]);
            dealii::VectorTools::project_boundary_values_curl_conforming_l2(
              *dof_handler.spatial(),
              first_vector_component,
              boundary_function,
              boundary_component,
              *space_constraints);

            for (unsigned int i = 0; i < n_space_dofs; i++)
              {
                // check if this is a constrained dof
                if (space_constraints->is_constrained(i))
                  {
                    const std::vector<std::pair<dealii::types::global_dof_index,
                                                double>> *entries =
                      space_constraints->get_constraint_entries(i);

                    spacetime_constraints->add_line(i +
                                                    time_dof * n_space_dofs);
                    // non Dirichlet constraint
                    if (!entries->empty())
                      {
                        for (auto entry : *entries)
                          {
                            spacetime_constraints->add_entry(
                              i + time_dof * n_space_dofs,
                              entry.first + time_dof * n_space_dofs,
                              entry.second);
                          }
                      }
                    else
                      {
                        spacetime_constraints->set_inhomogeneity(
                          i + time_dof * n_space_dofs,
                          space_constraints->get_inhomogeneity(i));
                      }
                  }
              }
          }
      }
  }

  template <int dim>
  double
  calculate_L2_squared_error_on_slab(DoFHandler<dim>        &dof_handler,
                                     dealii::Vector<double> &spacetime_vector,
                                     dealii::Function<dim>  &exact_solution,
                                     spacetime::Quadrature<dim>   &quad,
                                     dealii::VectorTools::NormType norm)
  {
    double              slab_norm_sqr = 0.;
    dealii::FEValues<1> fev_time(dof_handler.temporal()->get_fe(),
                                 *quad.temporal(),
                                 dealii::update_values |
                                   dealii::update_JxW_values |
                                   dealii::update_quadrature_points);

    dealii::Vector<double> space_vec;
    dealii::Vector<double> difference_per_cell;
    space_vec.reinit(dof_handler.n_dofs_space());
    difference_per_cell.reinit(
      dof_handler.spatial()->get_triangulation().n_active_cells());
    double       t            = 0;
    unsigned int offset       = 0;
    unsigned int n_dofs_space = dof_handler.n_dofs_space();
    for (auto cell_time : dof_handler.temporal()->active_cell_iterators())
      {
        offset =
          cell_time->index() * dof_handler.dofs_per_cell_time() * n_dofs_space;

        fev_time.reinit(cell_time);
        for (unsigned int q = 0; q < fev_time.n_quadrature_points; q++)
          {
            t = fev_time.quadrature_point(q)[0];
            exact_solution.set_time(t);
            // calculate spatial vector at t
            space_vec = 0;
            for (unsigned int ii = 0; ii < dof_handler.dofs_per_cell_time();
                 ii++)
              {
                double factor = fev_time.shape_value(ii, q);
                for (unsigned int i = 0; i < n_dofs_space; i++)
                  {
                    space_vec[i] +=
                      spacetime_vector[i + ii * n_dofs_space + offset] * factor;
                  }
              }
            // calculate L2 norm at current temporal QP
            difference_per_cell = 0;
            dealii::VectorTools::integrate_difference(*dof_handler.spatial(),
                                                      space_vec,
                                                      exact_solution,
                                                      difference_per_cell,
                                                      *quad.spatial(),
                                                      norm);

            slab_norm_sqr += difference_per_cell.norm_sqr() * fev_time.JxW(q);
          }
      }
    return slab_norm_sqr;
  }

  template <int dim>
  double
  calculate_L2_squared_error_on_slab(
    DoFHandler<dim>                       &dof_handler,
    dealii::TrilinosWrappers::MPI::Vector &spacetime_vector,
    dealii::Function<dim, double>         &exact_solution,
    spacetime::Quadrature<dim>            &quad,
    dealii::VectorTools::NormType          norm)
  {
    double              local_slab_norm_sqr = 0.;
    dealii::FEValues<1> fev_time(dof_handler.temporal()->get_fe(),
                                 *quad.temporal(),
                                 dealii::update_values |
                                   dealii::update_JxW_values |
                                   dealii::update_quadrature_points);
    MPI_Comm            comm = spacetime_vector.get_mpi_communicator();
    dealii::IndexSet    space_owned_dofs =
      dof_handler.spatial()->locally_owned_dofs();
    dealii::IndexSet space_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler.spatial(),
                                                    space_relevant_dofs);

    dealii::TrilinosWrappers::MPI::Vector owned_space_vec;
    dealii::TrilinosWrappers::MPI::Vector relevant_space_vec;
    owned_space_vec.reinit(space_owned_dofs, comm);
    relevant_space_vec.reinit(space_owned_dofs, space_relevant_dofs, comm);

    dealii::Vector<double> difference_per_cell;
    difference_per_cell.reinit(
      dof_handler.spatial()->get_triangulation().n_active_cells());

    double       t            = 0;
    unsigned int offset       = 0;
    unsigned int n_dofs_space = dof_handler.n_dofs_space();
    std::vector<dealii::types::global_dof_index> owned_indices =
      space_owned_dofs.get_index_vector();
    for (auto cell_time : dof_handler.temporal()->active_cell_iterators())
      {
        offset = cell_time->active_cell_index() *
                 dof_handler.dofs_per_cell_time() * n_dofs_space;

        fev_time.reinit(cell_time);
        for (unsigned int q = 0; q < fev_time.n_quadrature_points; q++)
          {
            t = fev_time.quadrature_point(q)[0];
            exact_solution.set_time(t);
            // calculate spatial vector at t
            owned_space_vec = 0;
            for (unsigned int ii = 0; ii < dof_handler.dofs_per_cell_time();
                 ii++)
              {
                double factor = fev_time.shape_value(ii, q);
                for (auto i : owned_indices)
                  {
                    owned_space_vec[i] +=
                      spacetime_vector[i + ii * n_dofs_space + offset] * factor;
                  }
              }

            relevant_space_vec = owned_space_vec;
            // calculate L2 norm at current temporal QP
            difference_per_cell = 0;
            dealii::VectorTools::integrate_difference(*dof_handler.spatial(),
                                                      relevant_space_vec,
                                                      exact_solution,
                                                      difference_per_cell,
                                                      *quad.spatial(),
                                                      norm);

            local_slab_norm_sqr +=
              difference_per_cell.norm_sqr() * fev_time.JxW(q);
          }
      }

    return dealii::Utilities::MPI::sum(local_slab_norm_sqr, comm);
  }


  template <int dim>
  dealii::Vector<double>
  compute_mean_values(
    DoFHandler<dim>                             &dof,
    spacetime::Quadrature<dim>                  &quad,
    const dealii::TrilinosWrappers::MPI::Vector &spacetime_vector)
  {
    AssertDimension(spacetime_vector.size(), dof.n_dofs_spacetime());
    const unsigned int n_components = dof.spatial()->get_fe().n_components();
    dealii::Vector<double>   local_mean(n_components);
    spacetime::FEValues<dim> fe_values(*dof.get_fe(),
                                       quad,
                                       dealii::update_values |
                                         dealii::update_JxW_values |
                                         dealii::update_quadrature_points);

    const MPI_Comm comm = spacetime_vector.get_mpi_communicator();

    double       area             = 0.;
    const unsigned int n_quad_spacetime = fe_values.n_quadrature_points;

    std::vector values(n_quad_spacetime, dealii::Vector<double>(n_components));

    for (auto cell_space : dof.spatial()->active_cell_iterators() |
                             dealii::IteratorFilters::LocallyOwnedCell())
      {
        fe_values.reinit_space(cell_space);

        for (auto cell_time : dof.temporal()->active_cell_iterators())
          {
            fe_values.reinit_time(cell_time);

            fe_values.get_function_values(spacetime_vector, values);

            for (unsigned int q = 0; q < n_quad_spacetime; ++q)
              {
                auto dq = fe_values.JxW(q);

                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    local_mean[c] += values[q][c] * dq;
                  }

                area += dq;
              }
          }
      }

    const std::vector local_array(local_mean.begin(), local_mean.end());
    std::vector<double> global_array(n_components);
    MPI_Allreduce(local_array.data(), global_array.data(), n_components, MPI_DOUBLE, MPI_SUM, comm);
    for (unsigned int c = 0; c < n_components; ++c)
      {
        local_mean[c] = global_array[c];
      }
    area = dealii::Utilities::MPI::sum(area, comm);
    local_mean /= area;

    return local_mean;
  }

} // namespace idealii::slab::VectorTools

#include "vector_tools.inst"
