#ifndef IDEAL_II_SPACETIME_RANGE_H
#define IDEAL_II_SPACETIME_RANGE_H

#include "spacetime_iterator.hh"

namespace idealii::spacetime {

  template <int dim>
  class SpaceTimeRange {
  public:
    using SpaceCell =
        typename dealii::DoFHandler<dim>::active_cell_iterator;

    using TimeCell =
        typename dealii::DoFHandler<1>::active_cell_iterator;

    SpaceTimeRange(SpaceCell space_begin,
                   SpaceCell space_end,
                   TimeCell time_begin,
                   TimeCell time_end)
        :
        space_begin(space_begin),
        space_end(space_end),
        time_begin(time_begin),
        time_end(time_end)
    {}

    auto begin() const {
      return SpaceTimeIterator<dim>(
          space_begin,
          space_end,
          time_begin,
          time_end);
    }

    auto end() const {
      return SpaceTimeIterator<dim>(
          space_end,
          space_end,
          time_begin,
          time_end);
    }

  private:
    SpaceCell space_begin;
    SpaceCell space_end;

    TimeCell time_begin;
    TimeCell time_end;
  };

  template <int dim>
SpaceTimeRange<dim> cells(const dealii::DoFHandler<1> &tria, const dealii::DoFHandler<dim> &dof) {
    using CellFilter =
        dealii::FilteredIterator<typename dealii::DoFHandler<dim>::active_cell_iterator>;

    return SpaceTimeRange<dim>(
        CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
                   dof.begin_active()),
        CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
                   dof.end()),
        tria.temporal()->begin_active(),
        tria.temporal()->end());
  }
}
#endif // IDEAL_II_SPACETIME_RANGE_H
