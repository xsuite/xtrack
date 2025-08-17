import xtrack as xt
import xpart as xp

def test_optimize_with_radiation():
    
    env = xt.Environment()

    env.new('mb', xt.Bend, length=1, k0=1, h=1)
    env.new('mq', xt.Quadrupole, length=1, k1=1)
    env.new('ms', xt.Sextupole, length=1, k2=1)
    env.new('mo', xt.Octupole, length=1, k3=1)

    element_list = ['mb', 'mq', 'ms', 'mo']

    for element in element_list:
        line = env.new_line(components=[element])
        
        line.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Teapot(1))])
        
        line.build_tracker()
        line.configure_radiation('mean')
        
        part = xp.Particles(p0c=1e15, x=0.11)
        line.track(part)
        
        line.optimize_for_tracking()
        part_opt = xp.Particles(p0c=1e15, x=0.11)
        line.track(part_opt)
        
        assert part.x == part_opt.x
        assert part.y == part_opt.y
        assert part.px == part_opt.px
        assert part.py == part_opt.py
        assert part.zeta == part_opt.zeta
        assert part.pzeta == part_opt.pzeta

def test_optimize_with_delta_taper():
    
    env = xt.Environment()

    env.new('mb', xt.Bend, length=1, k0=1, h=1)
    env.new('mq', xt.Quadrupole, length=1, k1=1)
    env.new('ms', xt.Sextupole, length=1, k2=1)
    env.new('mo', xt.Octupole, length=1, k3=1)

    thin_classes = [xt.ThinSliceBendEntry, xt.ThinSliceBendExit, xt.ThinSliceBend, 
                    xt.ThinSliceQuadrupole, xt.ThinSliceSextupole, xt.ThinSliceOctupole]

    element_list = ['mb', 'mq', 'ms', 'mo']

    for element in element_list:
        line = env.new_line(components=[element])
        

        line.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Teapot(1))])

        for el in line.elements:
            if type(el) in thin_classes:
                el.delta_taper = 0.1
                print(el)
        
        line.build_tracker()
        line.configure_radiation('mean')
        
        part = xp.Particles(p0c=1e15, x=0.11)
        line.track(part)
        
        line.optimize_for_tracking()
        part_opt = xp.Particles(p0c=1e15, x=0.11)
        line.track(part_opt)
        
        assert part.x == part_opt.x
        assert part.y == part_opt.y
        assert part.px == part_opt.px
        assert part.py == part_opt.py
        assert part.zeta == part_opt.zeta
        assert part.pzeta == part_opt.pzeta
