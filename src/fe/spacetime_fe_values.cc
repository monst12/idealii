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

#include <deal.II/lac/trilinos_vector.h>
namespace idealii::spacetime
{

  ////////////////////////////////////////////////////////////
  // FEValues ////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
  template <int dim>
  FEValues<dim>::FEValues(const DG_FiniteElement<dim> &fe,
                          const Quadrature<dim>       &quad,
                          const dealii::UpdateFlags    uflags)
    : _fe(fe)
    , _quad(quad)
    , _fev_space(std::make_shared<dealii::FEValues<dim>>(*fe.spatial(),
                                                         *quad.spatial(),
                                                         uflags))
    , _fev_time(std::make_shared<dealii::FEValues<1>>(*fe.temporal(),
                                                      *quad.temporal(),
                                                      uflags))
    , local_space_dof_index(fe.spatial()->dofs_per_cell)
    , local_time_dof_index(fe.temporal()->dofs_per_cell)
    , n_dofs_space(0)
    , time_cell_index(0)
    , n_dofs_space_cell(_fe.spatial()->dofs_per_cell)
    , n_quads_space(_fev_space->n_quadrature_points)
    , n_quadrature_points(_fev_space->n_quadrature_points *
                          _fev_time->n_quadrature_points)
  {}

  template <int dim>
  void
  FEValues<dim>::reinit_space(
    const dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
      &cell_space)
  {
    _fev_space->reinit(cell_space);
    cell_space->get_dof_indices(local_space_dof_index);
    n_dofs_space = cell_space->get_dof_handler().n_dofs();
  }

  template <int dim>
  void
  FEValues<dim>::reinit_time(
    const dealii::TriaIterator<dealii::DoFCellAccessor<1, 1, false>> &cell_time)
  {
    _fev_time->reinit(cell_time);
    cell_time->get_dof_indices(local_time_dof_index);
    time_cell_index = cell_time->index();
  }

  template <int dim>
  double
  FEValues<dim>::shape_value(const unsigned int function_no,
                             const unsigned int point_no) const
  {
    return _fev_space->shape_value(function_no % n_dofs_space_cell,
                                   point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  double
  FEValues<dim>::shape_dt(const unsigned int function_no,
                          const unsigned int point_no) const
  {
    return _fev_space->shape_value(function_no % n_dofs_space_cell,
                                   point_no % n_quads_space) *
           _fev_time->shape_grad(function_no / n_dofs_space_cell,
                                 point_no / n_quads_space)[0];
  }

  template <int dim>
  dealii::Tensor<1, dim>
  FEValues<dim>::shape_space_grad(const unsigned int function_no,
                                  const unsigned int point_no) const
  {
    return _fev_space->shape_grad(function_no % n_dofs_space_cell,
                                  point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  template <class InputVector>
  void
  FEValues<dim>::get_function_values(
    const InputVector                                             &fe_function,
    std::vector<dealii::Vector<typename InputVector::value_type>> &values) const
  {
    Assert(values.size() == n_quadrature_points,
           dealii::ExcDimensionMismatch(values.size(), n_quadrature_points));
    unsigned int comp_i_x = 0;
    unsigned int q        = 0;

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        Assert(values[q].size() == _fe.spatial()->n_components(),
               dealii::ExcDimensionMismatch(values[q].size(),
                                            _fe.spatial()->n_components()));
        values[q] = 0;
      }

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        for (unsigned int i_x = 0; i_x < n_dofs_space_cell; ++i_x)
          {
            comp_i_x = _fe.spatial()->system_to_component_index(i_x).first;
            const double phi_x =
              _fev_space->shape_value_component(i_x, q_x, comp_i_x);
            for (unsigned int q_t = 0; q_t < _fev_time->n_quadrature_points;
                 ++q_t)
              {
                q = q_x + n_quads_space * q_t;
                for (unsigned int i_t = 0; i_t < _fev_time->dofs_per_cell;
                     ++i_t)
                  {
                    values[q](comp_i_x) +=
                      phi_x * _fev_time->shape_value(i_t, q_t) *
                      fe_function[local_space_dof_index[i_x] +
                                  n_dofs_space * local_time_dof_index[i_t]];
                  }
              }
          }
      }
  }


  template <int dim>
  template <class InputVector>
  void
  FEValues<dim>::get_function_dt(
    const InputVector                                             &fe_function,
    std::vector<dealii::Vector<typename InputVector::value_type>> &values) const
  {
    Assert(values.size() == n_quadrature_points,
           dealii::ExcDimensionMismatch(values.size(), n_quadrature_points));
    unsigned int comp_i_x = 0;
    unsigned int q        = 0;

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        Assert(values[q].size() == _fe.spatial()->n_components(),
               dealii::ExcDimensionMismatch(values[q].size(),
                                            _fe.spatial()->n_components()));
        values[q] = 0;
      }

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        for (unsigned int i_x = 0; i_x < n_dofs_space_cell; ++i_x)
          {
            comp_i_x = _fe.spatial()->system_to_component_index(i_x).first;
            const double phi_x =
              _fev_space->shape_value_component(i_x, q_x, comp_i_x);
            for (unsigned int q_t = 0; q_t < _fev_time->n_quadrature_points;
                 ++q_t)
              {
                q = q_x + n_quads_space * q_t;
                for (unsigned int i_t = 0; i_t < _fev_time->dofs_per_cell;
                     ++i_t)
                  {
                    values[q](comp_i_x) +=
                      phi_x * _fev_time->shape_grad(i_t, q_t)[0] *
                      fe_function[local_space_dof_index[i_x] +
                                  n_dofs_space * local_time_dof_index[i_t]];
                  }
              }
          }
      }
  }

  template <int dim>
  template <class InputVector>
  void
  FEValues<dim>::get_function_space_gradients(
    const InputVector &fe_function,
    std::vector<
      std::vector<dealii::Tensor<1, dim, typename InputVector::value_type>>>
      &gradients) const
  {
    Assert(gradients.size() == n_quadrature_points,
           dealii::ExcDimensionMismatch(gradients.size(), n_quadrature_points));
    dealii::Tensor<1, dim> grad_phi_x;
    unsigned int           comp_i_x = 0;
    unsigned int           q        = 0;

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        Assert(gradients[q].size() == _fe.spatial()->n_components(),
               dealii::ExcDimensionMismatch(gradients[q].size(),
                                            _fe.spatial()->n_components()));
        for (unsigned int c = 0; c < _fe.spatial()->n_components(); ++c)
          {
            gradients[q][c] = 0;
          }
      }

    double u_i = 0;
    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        for (unsigned int i_x = 0; i_x < n_dofs_space_cell; ++i_x)
          {
            comp_i_x   = _fe.spatial()->system_to_component_index(i_x).first;
            grad_phi_x = _fev_space->shape_grad_component(i_x, q_x, comp_i_x);
            for (unsigned int q_t = 0; q_t < _fev_time->n_quadrature_points;
                 ++q_t)
              {
                q = q_x + n_quads_space * q_t;
                for (unsigned int i_t = 0; i_t < _fev_time->dofs_per_cell;
                     ++i_t)
                  {
                    u_i = fe_function[local_space_dof_index[i_x] +
                                      n_dofs_space * local_time_dof_index[i_t]];
                    gradients[q][comp_i_x] +=
                      grad_phi_x * _fev_time->shape_value(i_t, q_t) * u_i;
                  }
              }
          }
      }
  }

  template <int dim>
  typename dealii::FEValuesViews::Scalar<dim>::value_type
  FEValues<dim>::scalar_value(
    const dealii::FEValuesExtractors::Scalar &extractor,
    const unsigned int                        function_no,
    const unsigned int                        point_no) const
  {
    return (*_fev_space)[extractor].value(function_no % n_dofs_space_cell,
                                          point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  typename dealii::FEValuesViews::Scalar<dim>::value_type
  FEValues<dim>::scalar_dt(const dealii::FEValuesExtractors::Scalar &extractor,
                           const unsigned int function_no,
                           const unsigned int point_no) const
  {
    return (*_fev_space)[extractor].value(function_no % n_dofs_space_cell,
                                          point_no % n_quads_space) *
           _fev_time->shape_grad(function_no / n_dofs_space_cell,
                                 point_no / n_quads_space)[0];
  }

  template <int dim>
  typename dealii::FEValuesViews::Scalar<dim>::gradient_type
  FEValues<dim>::scalar_space_grad(
    const dealii::FEValuesExtractors::Scalar &extractor,
    const unsigned int                        function_no,
    const unsigned int                        point_no) const
  {
    return (*_fev_space)[extractor].gradient(function_no % n_dofs_space_cell,
                                             point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::value_type
  FEValues<dim>::vector_value(
    const dealii::FEValuesExtractors::Vector &extractor,
    const unsigned int                        function_no,
    const unsigned int                        point_no) const
  {
    return (*_fev_space)[extractor].value(function_no % n_dofs_space_cell,
                                          point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::value_type
  FEValues<dim>::vector_dt(const dealii::FEValuesExtractors::Vector &extractor,
                           const unsigned int function_no,
                           const unsigned int point_no) const
  {
    return (*_fev_space)[extractor].value(function_no % n_dofs_space_cell,
                                          point_no % n_quads_space) *
           _fev_time->shape_grad(function_no / n_dofs_space_cell,
                                 point_no / n_quads_space)[0];
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::divergence_type
  FEValues<dim>::vector_divergence(
    const dealii::FEValuesExtractors::Vector &extractor,
    const unsigned int                        function_no,
    const unsigned int                        point_no) const
  {
    return (*_fev_space)[extractor].divergence(function_no % n_dofs_space_cell,
                                               point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::gradient_type
  FEValues<dim>::vector_space_grad(
    const dealii::FEValuesExtractors::Vector &extractor,
    const unsigned int                        function_no,
    const unsigned int                        point_no) const
  {
    return (*_fev_space)[extractor].gradient(function_no % n_dofs_space_cell,
                                             point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::curl_type
  FEValues<dim>::vector_space_curl(
    const dealii::FEValuesExtractors::Vector &extractor,
    const unsigned int                        function_no,
    const unsigned int                        point_no) const
  {
    return (*_fev_space)[extractor].curl(function_no % n_dofs_space_cell,
                                         point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  double
  FEValues<dim>::time_quadrature_point(
    const unsigned int quadrature_point) const
  {
    return _fev_time->quadrature_point(quadrature_point / n_quads_space)[0];
  }

  template <int dim>
  const dealii::Point<dim> &
  FEValues<dim>::space_quadrature_point(
    const unsigned int quadrature_point) const
  {
    return _fev_space->quadrature_point(quadrature_point % n_quads_space);
  }

  template <int dim>
  double
  FEValues<dim>::JxW(const unsigned int quadrature_point) const
  {
    return _fev_space->JxW(quadrature_point % n_quads_space) *
           _fev_time->JxW(quadrature_point / n_quads_space);
  }

  template <int dim>
  void
  FEValues<dim>::get_local_dof_indices(
    std::vector<dealii::types::global_dof_index> &indices) const
  {
    for (unsigned int i = 0; i < _fe.dofs_per_cell; i++)
      {
        indices[i + time_cell_index * _fe.dofs_per_cell] =
          local_space_dof_index[i % n_dofs_space_cell] +
          local_time_dof_index[i / n_dofs_space_cell] * n_dofs_space;
      }
  }

  template <int dim>
  std::shared_ptr<dealii::FEValues<dim>>
  FEValues<dim>::spatial() const
  {
    return _fev_space;
  }

  template <int dim>
  std::shared_ptr<dealii::FEValues<1>>
  FEValues<dim>::temporal() const
  {
    return _fev_time;
  }
  ////////////////////////////////////////////////////////////
  // FEJumpValues ////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
  template <int dim>
  FEJumpValues<dim>::FEJumpValues(const DG_FiniteElement<dim> &fe,
                                  const Quadrature<dim>       &quad,
                                  const dealii::UpdateFlags    uflags)
    : _fe(fe)
    , _quad(quad)
    , _fev_space(std::make_shared<dealii::FEValues<dim>>(*fe.spatial(),
                                                         *quad.spatial(),
                                                         uflags))
    , _fev_time(
        std::make_shared<dealii::FEValues<1>>(*fe.temporal(),
                                              dealii::QGaussLobatto<1>(2),
                                              uflags))
    , local_space_dof_index(fe.spatial()->dofs_per_cell)
    , local_time_dof_index(fe.temporal()->dofs_per_cell)
  {
    n_quadrature_points =
      _fev_space->n_quadrature_points * _fev_time->n_quadrature_points;
    n_dofs_space = 0;
  }

  template <int dim>
  void
  FEJumpValues<dim>::reinit_space(
    const dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
      &cell_space)
  {
    _fev_space->reinit(cell_space);
    cell_space->get_dof_indices(local_space_dof_index);
    n_dofs_space = cell_space->get_dof_handler().n_dofs();
  }

  template <int dim>
  void
  FEJumpValues<dim>::reinit_time(
    const dealii::TriaIterator<dealii::DoFCellAccessor<1, 1, false>> &cell_time)
  {
    _fev_time->reinit(cell_time);
    cell_time->get_dof_indices(local_time_dof_index);
  }

  template <int dim>
  double
  FEJumpValues<dim>::shape_value_plus(unsigned int function_no,
                                      unsigned int point_no) const
  {
    return _fev_space->shape_value(function_no % _fe.spatial()->dofs_per_cell,
                                   point_no % _fev_space->n_quadrature_points) *
           _fev_time->shape_value(function_no / _fe.spatial()->dofs_per_cell,
                                  0);
  }

  template <int dim>
  double
  FEJumpValues<dim>::shape_value_minus(unsigned int function_no,
                                       unsigned int point_no) const
  {
    return _fev_space->shape_value(function_no % _fe.spatial()->dofs_per_cell,
                                   point_no % _fev_space->n_quadrature_points) *
           _fev_time->shape_value(function_no / _fe.spatial()->dofs_per_cell,
                                  1);
  }

  template <int dim>
  double
  FEJumpValues<dim>::shape_dt_plus(unsigned int function_no, unsigned int point_no) const
  {
    return _fev_space->shape_value(function_no % _fe.spatial()->dofs_per_cell,
                                   point_no % _fev_space->n_quadrature_points) *
           _fev_time->shape_grad(function_no / _fe.spatial()->dofs_per_cell,
                                 0)[0];
  }

  template <int dim>
  double
  FEJumpValues<dim>::shape_dt_minus(unsigned int function_no, unsigned int point_no) const
  {
    return _fev_space->shape_value(function_no % _fe.spatial()->dofs_per_cell,
                                   point_no % _fev_space->n_quadrature_points) *
           _fev_time->shape_grad(function_no / _fe.spatial()->dofs_per_cell,
                                 1)[0];
  }

  template <int dim>
  template <class InputVector>
  void
  FEJumpValues<dim>::get_function_values_plus(
    const InputVector                                             &fe_function,
    std::vector<dealii::Vector<typename InputVector::value_type>> &values) const
  {
    unsigned int n_quads_space = _fev_space->n_quadrature_points;
    Assert(values.size() == n_quads_space,
           dealii::ExcDimensionMismatch(values.size(), n_quads_space));

    unsigned int comp_i_x = 0;

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        Assert(values[q_x].size() == _fe.spatial()->n_components(),
               dealii::ExcDimensionMismatch(values[q_x].size(),
                                            _fe.spatial()->n_components()));
        values[q_x] = 0;
      }

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        for (unsigned int i_x = 0; i_x < _fe.spatial()->dofs_per_cell; ++i_x)
          {
            comp_i_x = _fe.spatial()->system_to_component_index(i_x).first;
            const double phi_x =
              _fev_space->shape_value_component(i_x, q_x, comp_i_x);
            for (unsigned int i_t = 0; i_t < _fev_time->dofs_per_cell; ++i_t)
              {
                values[q_x](comp_i_x) +=
                  phi_x * _fev_time->shape_value(i_t, 0) *
                  fe_function[local_space_dof_index[i_x] +
                              n_dofs_space * local_time_dof_index[i_t]];
              }
          }
      }
  }

  template <int dim>
  template <class InputVector>
  void
  FEJumpValues<dim>::get_function_values_minus(
    const InputVector                                             &fe_function,
    std::vector<dealii::Vector<typename InputVector::value_type>> &values) const
  {
    unsigned int n_quads_space = _fev_space->n_quadrature_points;
    Assert(values.size() == n_quads_space,
           dealii::ExcDimensionMismatch(values.size(), n_quads_space));

    unsigned int comp_i_x = 0;

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        Assert(values[q_x].size() == _fe.spatial()->n_components(),
               dealii::ExcDimensionMismatch(values[q_x].size(),
                                            _fe.spatial()->n_components()));
        values[q_x] = 0;
      }

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        for (unsigned int i_x = 0; i_x < _fe.spatial()->dofs_per_cell; ++i_x)
          {
            comp_i_x = _fe.spatial()->system_to_component_index(i_x).first;
            const double phi_x =
              _fev_space->shape_value_component(i_x, q_x, comp_i_x);
            for (unsigned int i_t = 0; i_t < _fev_time->dofs_per_cell; ++i_t)
              {
                values[q_x](comp_i_x) +=
                  phi_x * _fev_time->shape_value(i_t, 1) *
                  fe_function[local_space_dof_index[i_x] +
                              n_dofs_space * local_time_dof_index[i_t]];
              }
          }
      }
  }

  template <int dim>
  template <class InputVector>
  void
  FEJumpValues<dim>::get_function_dt_plus(
    const InputVector                                             &fe_function,
    std::vector<dealii::Vector<typename InputVector::value_type>> &values) const
  {
    unsigned int n_quads_space = _fev_space->n_quadrature_points;
    Assert(values.size() == n_quads_space,
           dealii::ExcDimensionMismatch(values.size(), n_quads_space));

    unsigned int comp_i_x = 0;

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        Assert(values[q_x].size() == _fe.spatial()->n_components(),
               dealii::ExcDimensionMismatch(values[q_x].size(),
                                            _fe.spatial()->n_components()));
        values[q_x] = 0;
      }

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        for (unsigned int i_x = 0; i_x < _fe.spatial()->dofs_per_cell; ++i_x)
          {
            comp_i_x = _fe.spatial()->system_to_component_index(i_x).first;
            const double phi_x =
              _fev_space->shape_value_component(i_x, q_x, comp_i_x);
            for (unsigned int i_t = 0; i_t < _fev_time->dofs_per_cell; ++i_t)
              {
                values[q_x](comp_i_x) +=
                  phi_x * _fev_time->shape_grad(i_t, 0)[0] *
                  fe_function[local_space_dof_index[i_x] +
                              n_dofs_space * local_time_dof_index[i_t]];
              }
          }
      }
  }

  template <int dim>
  template <class InputVector>
  void
  FEJumpValues<dim>::get_function_dt_minus(
    const InputVector                                             &fe_function,
    std::vector<dealii::Vector<typename InputVector::value_type>> &values) const
  {
    unsigned int n_quads_space = _fev_space->n_quadrature_points;
    Assert(values.size() == n_quads_space,
           dealii::ExcDimensionMismatch(values.size(), n_quads_space));

    unsigned int comp_i_x = 0;

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        Assert(values[q_x].size() == _fe.spatial()->n_components(),
               dealii::ExcDimensionMismatch(values[q_x].size(),
                                            _fe.spatial()->n_components()));
        values[q_x] = 0;
      }

    for (unsigned int q_x = 0; q_x < n_quads_space; ++q_x)
      {
        for (unsigned int i_x = 0; i_x < _fe.spatial()->dofs_per_cell; ++i_x)
          {
            comp_i_x = _fe.spatial()->system_to_component_index(i_x).first;
            const double phi_x =
              _fev_space->shape_value_component(i_x, q_x, comp_i_x);
            for (unsigned int i_t = 0; i_t < _fev_time->dofs_per_cell; ++i_t)
              {
                values[q_x](comp_i_x) +=
                  phi_x * _fev_time->shape_grad(i_t, 1)[0] *
                  fe_function[local_space_dof_index[i_x] +
                              n_dofs_space * local_time_dof_index[i_t]];
              }
          }
      }
  }

  template <int dim>
  typename dealii::FEValuesViews::Scalar<dim>::value_type
  FEJumpValues<dim>::scalar_value_plus(
    const dealii::FEValuesExtractors::Scalar &extractor,
    unsigned int                              function_no,
    unsigned int                              point_no) const
  {
    return (*_fev_space)[extractor].value(function_no %
                                            _fe.spatial()->dofs_per_cell,
                                          point_no %
                                            _fev_space->n_quadrature_points) *
           _fev_time->shape_value(function_no / _fe.spatial()->dofs_per_cell,
                                  0);
  }

  template <int dim>
  typename dealii::FEValuesViews::Scalar<dim>::value_type
  FEJumpValues<dim>::scalar_value_minus(
    const dealii::FEValuesExtractors::Scalar &extractor,
    unsigned int                              function_no,
    unsigned int                              point_no) const
  {
    return (*_fev_space)[extractor].value(function_no %
                                            _fe.spatial()->dofs_per_cell,
                                          point_no %
                                            _fev_space->n_quadrature_points) *
           _fev_time->shape_value(function_no / _fe.spatial()->dofs_per_cell,
                                  1);
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::value_type
  FEJumpValues<dim>::vector_value_plus(
    const dealii::FEValuesExtractors::Vector &extractor,
    unsigned int                              function_no,
    unsigned int                              point_no) const
  {
    return (*_fev_space)[extractor].value(function_no %
                                            _fe.spatial()->dofs_per_cell,
                                          point_no %
                                            _fev_space->n_quadrature_points) *
           _fev_time->shape_value(function_no / _fe.spatial()->dofs_per_cell,
                                  0);
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::value_type
  FEJumpValues<dim>::vector_value_minus(
    const dealii::FEValuesExtractors::Vector &extractor,
    unsigned int                              function_no,
    unsigned int                              point_no) const
  {
    return (*_fev_space)[extractor].value(function_no %
                                            _fe.spatial()->dofs_per_cell,
                                          point_no %
                                            _fev_space->n_quadrature_points) *
           _fev_time->shape_value(function_no / _fe.spatial()->dofs_per_cell,
                                  1);
  }

  template <int dim>
  typename dealii::FEValuesViews::Scalar<dim>::value_type
  FEJumpValues<dim>::scalar_dt_plus(const typename dealii::FEValuesExtractors::Scalar &extractor,
                 unsigned int                                       function_no,
                 unsigned int point_no) const
  {
    return (*_fev_space)[extractor].value(function_no %
                                            _fe.spatial()->dofs_per_cell,
                                          point_no %
                                            _fev_space->n_quadrature_points) *
           _fev_time->shape_grad(function_no / _fe.spatial()->dofs_per_cell,
                                 0)[0];
  }

  template <int dim>
  typename dealii::FEValuesViews::Scalar<dim>::value_type
  FEJumpValues<dim>::scalar_dt_minus(const typename dealii::FEValuesExtractors::Scalar &extractor,
                  unsigned int function_no,
                  unsigned int point_no) const
  {
    return (*_fev_space)[extractor].value(function_no %
                                            _fe.spatial()->dofs_per_cell,
                                          point_no %
                                            _fev_space->n_quadrature_points) *
           _fev_time->shape_grad(function_no / _fe.spatial()->dofs_per_cell,
                                 1)[0];
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::value_type
  FEJumpValues<dim>::vector_dt_plus(const typename dealii::FEValuesExtractors::Vector &extractor,
                 unsigned int                                       function_no,
                 unsigned int point_no) const
  {
    return (*_fev_space)[extractor].value(function_no %
                                            _fe.spatial()->dofs_per_cell,
                                          point_no %
                                            _fev_space->n_quadrature_points) *
           _fev_time->shape_grad(function_no / _fe.spatial()->dofs_per_cell,
                                 0)[0];
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::value_type
  FEJumpValues<dim>::vector_dt_minus(const typename dealii::FEValuesExtractors::Vector &extractor,
                  unsigned int function_no,
                  unsigned int point_no) const
  {
    return (*_fev_space)[extractor].value(function_no %
                                            _fe.spatial()->dofs_per_cell,
                                          point_no %
                                            _fev_space->n_quadrature_points) *
           _fev_time->shape_grad(function_no / _fe.spatial()->dofs_per_cell,
                                 1)[0];
  }

  template <int dim>
  double
  FEJumpValues<dim>::JxW(unsigned int quadrature_point) const
  {
    return _fev_space->JxW(quadrature_point % _fev_space->n_quadrature_points);
  }

  template <int dim>
  std::shared_ptr<dealii::FEValues<dim>>
  FEJumpValues<dim>::spatial() const
  {
    return _fev_space;
  }

  template <int dim>
  std::shared_ptr<dealii::FEValues<1>>
  FEJumpValues<dim>::temporal() const
  {
    return _fev_time;
  }

  ////////////////////////////////////////////////////////////
  // FEFaceValues ////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
  template <int dim>
  FEFaceValues<dim>::FEFaceValues(const DG_FiniteElement<dim> &fe,
                                  const Quadrature<dim - 1>   &quad,
                                  const dealii::UpdateFlags    uflags,
                                  const dealii::UpdateFlags    additional_flags)
    : _fe(fe)
    , _quad(quad)
    , _fev_space(
        std::make_shared<dealii::FEFaceValues<dim>>(*fe.spatial(),
                                                    *quad.spatial(),
                                                    uflags | additional_flags))
    , _fev_time(std::make_shared<dealii::FEValues<1>>(*fe.temporal(),
                                                      *quad.temporal(),
                                                      uflags))
    , local_space_dof_index(fe.spatial()->dofs_per_cell)
    , local_time_dof_index(fe.temporal()->dofs_per_cell)
    , n_dofs_space(0)
    , time_cell_index(0)
    , n_dofs_space_cell(_fe.spatial()->dofs_per_cell)
    , n_quads_space(_fev_space->n_quadrature_points)
    , n_quadrature_points(_fev_space->n_quadrature_points *
                          _fev_time->n_quadrature_points)
  {}

  template <int dim>
  void
  FEFaceValues<dim>::reinit_space(
    const dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
                      &cell_space,
    const unsigned int face_no)
  {
    _fev_space->reinit(cell_space, face_no);
    cell_space->get_dof_indices(local_space_dof_index);
    n_dofs_space = cell_space->get_dof_handler().n_dofs();
  }

  template <int dim>
  void
  FEFaceValues<dim>::reinit_time(
    const dealii::TriaIterator<dealii::DoFCellAccessor<1, 1, false>> &cell_time)
  {
    _fev_time->reinit(cell_time);
    cell_time->get_dof_indices(local_time_dof_index);
    time_cell_index = cell_time->index();
  }

  template <int dim>
  double
  FEFaceValues<dim>::shape_value(const unsigned int function_no,
                                 const unsigned int point_no) const
  {
    return _fev_space->shape_value(function_no % n_dofs_space_cell,
                                   point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  typename dealii::FEValuesViews::Scalar<dim>::value_type
  FEFaceValues<dim>::scalar_value(
    const dealii::FEValuesExtractors::Scalar &extractor,
    const unsigned int                        function_no,
    const unsigned int                        point_no) const
  {
    return (*_fev_space)[extractor].value(function_no % n_dofs_space_cell,
                                          point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  typename dealii::FEValuesViews::Vector<dim>::value_type
  FEFaceValues<dim>::vector_value(
    const dealii::FEValuesExtractors::Vector &extractor,
    const unsigned int                        function_no,
    const unsigned int                        point_no) const
  {
    return (*_fev_space)[extractor].value(function_no % n_dofs_space_cell,
                                          point_no % n_quads_space) *
           _fev_time->shape_value(function_no / n_dofs_space_cell,
                                  point_no / n_quads_space);
  }

  template <int dim>
  double
  FEFaceValues<dim>::time_quadrature_point(
    const unsigned int quadrature_point) const
  {
    return _fev_time->quadrature_point(quadrature_point / n_quads_space)[0];
  }

  template <int dim>
  const dealii::Point<dim> &
  FEFaceValues<dim>::space_quadrature_point(
    const unsigned int quadrature_point) const
  {
    return _fev_space->quadrature_point(quadrature_point % n_quads_space);
  }

  template <int dim>
  const dealii::Tensor<1, dim> &
  FEFaceValues<dim>::space_normal_vector(const unsigned int i) const
  {
    return _fev_space->normal_vector(i % n_quads_space);
  }

  template <int dim>
  double
  FEFaceValues<dim>::JxW(const unsigned int quadrature_point) const
  {
    return _fev_space->JxW(quadrature_point % n_quads_space) *
           _fev_time->JxW(quadrature_point / n_quads_space);
  }

  template <int dim>
  void
  FEFaceValues<dim>::get_local_dof_indices(
    std::vector<dealii::types::global_dof_index> &indices) const
  {
    for (unsigned int i = 0; i < _fe.dofs_per_cell; i++)
      {
        indices[i + time_cell_index * _fe.dofs_per_cell] =
          local_space_dof_index[i % n_dofs_space_cell] +
          local_time_dof_index[i / n_dofs_space_cell] * n_dofs_space;
      }
  }

  template <int dim>
  std::shared_ptr<dealii::FEFaceValues<dim>>
  FEFaceValues<dim>::spatial() const
  {
    return _fev_space;
  }

  template <int dim>
  std::shared_ptr<dealii::FEValues<1>>
  FEFaceValues<dim>::temporal() const
  {
    return _fev_time;
  }
} // namespace idealii::spacetime

#include "spacetime_fe_values.inst"
