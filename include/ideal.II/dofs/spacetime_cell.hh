#ifndef IDEAL_II_SPACETIME_CELL_HH
#define IDEAL_II_SPACETIME_CELL_HH

#include <deal.II/dofs/dof_handler.h>

namespace idealii::spacetime {
  template <int dim>
class SpaceTimeCell {
  public:
    using SpaceCell =
        typename dealii::DoFHandler<dim>::active_cell_iterator;

    using TimeCell =
        typename dealii::DoFHandler<1>::active_cell_iterator;

    SpaceTimeCell(SpaceCell space_cell,
                  TimeCell time_cell,
                  unsigned int time_index)
        :
        space_cell(space_cell),
        time_cell(time_cell),
        time_index(time_index)
    {}

    SpaceCell spatial() const { return space_cell; }
    TimeCell  time()  const { return time_cell; }

    unsigned int index_time() const { return time_index; }

  private:
    SpaceCell space_cell;
    TimeCell  time_cell;
    unsigned int time_index;
  };
}

#endif // IDEAL_II_SPACETIME_CELL_HH
