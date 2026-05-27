#ifndef IDEAL_II_SPACETIME_ITERATOR_HH
#define IDEAL_II_SPACETIME_ITERATOR_HH

#include <deal.II/dofs/dof_handler.h>

namespace idealii::spacetime {
  template <int dim>
  class SpaceTimeIterator
  {
  public:

    using SpaceCell =
        typename dealii::DoFHandler<dim>::active_cell_iterator;

    using TimeCell =
        typename dealii::DoFHandler<1>::active_cell_iterator;

    SpaceTimeIterator(SpaceCell space,
                  SpaceCell space_end,
                  TimeCell time_begin,
                  TimeCell time_end);

    SpaceTimeCell<dim> operator*() const {
		return {space_cell, time_cell, time_index};
	}

    SpaceTimeIterator &operator++() {
      ++time_cell;
      ++time_index;

      if(time_cell == time_end)
        {
          ++space_cell;

          if(space_cell != space_end)
            {
              time_cell = time_begin;
              time_index = 0;
            }
        }

      return *this;
    }

    bool operator!=(const SpaceTimeIterator &other) const {
      return space_cell != other.space_cell;
    }

  private:
    SpaceCell space_cell;
    SpaceCell space_end;

    TimeCell time_cell;
    TimeCell time_begin;
    TimeCell time_end;

    unsigned int time_index;
  };

}
#endif // IDEAL_II_SPACETIME_ITERATOR_HH
