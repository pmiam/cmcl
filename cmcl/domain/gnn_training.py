"""
Additional functions to create Graph Network Training Directories
"""
from pymatgen.core import Structure, Composition
import multiprocessing as mp

def count_unit_cells(struct:Structure)->int:
    """ compute number of unit cells in a structure """
    prime_struct = struct.get_primitive_structure(tolerance=0.25)
    formula = Composition(struct.formula)
    prime_formula = Composition(prime_struct.formula)
    #formula_dict = Composition(struct.formula).as_dict()
    #cell_count = sum([Bnum for B,Bnum in formula_dict.items() if B in Bel])
        
    f_unit, f_units_per_super_cell = formula.get_reduced_formula_and_factor()
    _, f_units_per_unit_cell = prime_formula.get_reduced_formula_and_factor()
    return f_unit, f_units_per_super_cell/f_units_per_unit_cell

def make_record_name(doc, cald:dict, step:Any)->str:
    """
    return string to uniquely identify a structure file and id_prop
    record from query info.

    unique id made of:
    formula + LoT + step
    """
    formula = doc.dict()['formula_pretty']
    LoT = cald['run_type']
    ttable = {ord('-'):None,
              ord(' '):None,
              ord(':'):None,
              ord('.'):None}
    dt = cald['completed_at']
    dt = str(dt).translate(ttable)
    return f"{formula}_{LoT}_{step}_{dt}"

def make_properties_entry(record:str, props:list) -> None:
    """
    write a cgcnn-compliant training target file
    """
    props=','.join(map(str,props))
    return f"{record},{props}"

def structure_to_training_set_entry(struct:Structure,
                                    record:str,
                                    props:list,
                                    fdir:Union[str,Path]) -> None:
    """
    write a structure to a POSCAR named record in directory fdir
    
    write structure properties to properties file in directory fdir
    """
    filename=os.path.join(fdir, record)
    struct.to(fmt='POSCAR', filename=filename)
    return make_properties_entry(record, props)

def main_parser_old(paths, l, p, fdir, csv, err):
    s = time.perf_counter()
    manager = mp.Manager()
    q = manager.Queue()
    eq = manager.Queue()
    
    aggrfile = os.path.join(fdir, csv)
    errfile = os.path.join(fdir, err)
    with open(aggrfile, 'a') as f, open(errfile, 'a') as ef:
        f.write("id,metadata,totE,decoE,bg\n")
        f.flush()

        for group in grouper(grouper(paths, l), p):
            ps=[]
            for g in group:
                #print(g)
                p = mp.Process(target=gworker, args=(g, q, eq, fdir)) 
                ps.append(p)
                p.start()
            for p in ps:
                p.join()

            while not q.empty():
                f.write(str(q.get()) + '\n')
            f.flush()

            while not eq.empty():
                ef.write(str(eq.get()) + '\n')
            ef.flush()

    d = time.perf_counter() - s
    return d

def safe_mp_write_to_file():
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() + 2)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for i in range(80):
        job = pool.apply_async(worker, (i, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

def make_training_data(doc, fdir):
    """
    turn task document to cgcnn-complaint training set entry
    """
    # f = doc.input.pseudo_potentials.functional
    # f = f.replace("_", "")
    strecords = []
    for calc in doc.calcs_reversed:
        cald = calc.dict()
        struct = cald['input']['structure'] #POSCAR
        fu, cell_count = count_unit_cells(struct)
        toten_pfu = cald['output']['energy']/cell_count

        runtype = cald['run_type']
        PBE = "PBE" if "GGA" in runtype.name else False
        HSE = "HSE" if "HSE" in runtype.name else False
        f = HSE or PBE

        metadata = str(cald['dir_name'])
        bg = cald['output']['bandgap']
        decoE = compute_decomposition_energy(fu, toten_pfu,
                                             functional=f) #from cmcl

        # predictions on POSCARs should predict CONTCAR energies

        record_name = make_record_name(doc, cald, "POSCAR")
        strecords.append(
            structure_to_training_set_entry(struct,
                                            record_name,
                                            props=[metadata, float(toten_pfu), decoE, bg],
                                            fdir=fdir)
            )
        
        for count, step in enumerate(cald['output']['ionic_steps']):
            struct = step['structure'] #XDATCAR iteration
            toten_pfu = step['e_fr_energy']/cell_count
            decoE = compute_decomposition_energy(fu, toten_pfu,
                                                 functional=f) #from cmcl
            bg = "" # bg cannot be saved from intermediates in
                    # vasp runs not configured to return them at
                    # each step
 
            # compare structure steps to POSCARs before saving?
            # match_kwargs = dict(ltol=0.2,stol=0.3, angle_tol=5,
            #                     primitive_cell=True, scale=True,
            #                     attempt_supercell=False,
            #                     allow_subset=False,
            #                     supercell_size=True)
            # if struct.matches(POSCAR, **match_kwargs):
 
            record_name = make_record_name(doc, cald, count+1)
            strecords.append(
                structure_to_training_set_entry(struct,
                                                record_name,
                                                props=[metadata, toten_pfu, decoE, bg],
                                                fdir=fdir)
                )
    return strecords

