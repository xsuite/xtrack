"""Microbenchmark example: python examples/lattice_design/004_line_view_benchmark.py.

Run on the baseline and on LineView with the same Python/environment. The
ordinary-element measurements work before the change; the specialised ones
are available after it. No tracking, compilation, or setup is timed.
"""
from timeit import repeat

import xtrack as xt
from xtrack import view as xt_view


def report(label, operation, number=100_000):
    best = min(repeat(operation, number=number, repeat=5))
    print(f'{label:35s} {1e9 * best / number:10.1f} ns/call')


def main():
    env = xt.Environment()
    env.new('d', xt.Drift, length=1.)
    env.new('q', xt.Quadrupole, length=0.5)
    line = env.new_line(components=['d', 'q'])
    view = line['d']
    report('ordinary lookup', lambda: line['d'])
    report('ordinary lookup + length', lambda: line['d'].length)
    report('ordinary existing view.length', lambda: view.length)
    report('raw element lookup', lambda: line.get('d'))
    names = [f'd{i}' for i in range(1000)]
    bulk = xt.Line(elements={name: xt.Drift(length=1.) for name in names})
    report('1000 ordinary lookups + length',
           lambda: [bulk[name].length for name in names], number=100)
    if not hasattr(xt_view, 'LineElementView'):
        return

    class QuadrupoleView(xt_view.LineElementView):
        def line_context(self):
            return self._view_line, self._view_name

    # Temporarily opt in an existing class, avoiding a custom tracking kernel.
    missing = object()
    previous = xt.Quadrupole.__dict__.get('_line_view_class', missing)
    xt.Quadrupole._line_view_class = QuadrupoleView
    try:
        view = line['q']
        report('specialised lookup', lambda: line['q'])
        report('specialised lookup + length', lambda: line['q'].length)
        report('specialised existing view.length', lambda: view.length)
        report('specialised method binding', lambda: view.line_context)
        report('specialised method call', lambda: view.line_context())
    finally:
        if previous is missing:
            del xt.Quadrupole._line_view_class
        else:
            xt.Quadrupole._line_view_class = previous


if __name__ == '__main__':
    main()
