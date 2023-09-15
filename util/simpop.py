from util.sbwrapper import Trial
from util.breedstrats import random_selection, phenotypic_selection
import networkx as nx
import numpy as np


def generate_random_population(do_splitting=True, do_selection=False, num_timesteps=5, p_split=0.3, num_base=None,
                               pop_size=1000, num_snps=12000, heritability=0.5, n_qtl=10, dom_p=0.1, alpha=0.2, beta=0.5, rr=1.,
                               uo_ratio=None, uo_mask_size=None, verbose=False):
    heritability = [heritability]
    if num_base is None:
        num_base = pop_size

    subpopulations = {}
    subpopulation_nonce = 0
    valid_subpops = []

    global_generation_count = 0
    generation_nonce = 0
    generations_graph = nx.DiGraph()
    population_graph = nx.DiGraph()

    experiment = Trial()
    experiment.set_dom_p(dom_p)
    experiment.set_gamma_params(alpha, beta)
    experiment.set_rr(rr)
    experiment.generate_random_founders(num_snps, num_base, ploidy=2)
    experiment.define_traits(h2=heritability, nqtl=n_qtl)

    if uo_ratio is not None and uo_mask_size is not None:
        experiment.set_uo_mask(uo_ratio, uo_mask_size)

    experiment.make_founder_generation()
    founders = experiment.get_generation(0)

    # Add founder generation to the graph
    generations_graph.add_node(generation_nonce, ids=[ind.id for ind in founders])

    # Add founders to the population graph
    for ind in founders:
        population_graph.add_node(ind.id, individual=ind)

    subpopulations[subpopulation_nonce] = [generation_nonce]
    valid_subpops.append(subpopulation_nonce)

    for i in range(num_timesteps):
        # Main tick loop
        if verbose:
            print('Tick {0}'.format(i))
        new_subpops = []

        # Measure the size of the population so we don't fall below the minimum size
        total_pop_size = np.sum([len(generations_graph.nodes[subpopulations[j][-1]]['ids']) for j in valid_subpops])
        tack_on = np.maximum(0., (np.ceil((float(pop_size - total_pop_size)) / float(len(valid_subpops)))))
        if verbose:
            print('Population size: {0} (will try to tack on {1} per subpop)'.format(total_pop_size, int(tack_on)))

        # Make the latest generation for all the subpopulations
        for j in valid_subpops:
            if verbose:
                print('Doing subpopulation {0}'.format(j))
            youngest_subpop_uid = subpopulations[j][-1]
            youngest_subpop_ids = generations_graph.nodes[youngest_subpop_uid]['ids']
            youngest_subpop = [population_graph.nodes[iid]['individual'] for iid in youngest_subpop_ids]

            if do_selection:
                crosses = phenotypic_selection(youngest_subpop, int(len(youngest_subpop) / 2), method='pairs')
                nc = np.random.randint(2, 9, len(crosses))

                # Fix to prevent subpopulations from going to one
                if len(youngest_subpop) <= 5 or i == (num_timesteps - 1):
                    nc[nc < 4] = 4

                # Fix to maintain a minimum pop size
                nc = nc + int(np.ceil(tack_on / float(len(nc))))

                experiment.make_crosses(crosses, num_children=nc)
            else:
                crosses = random_selection(youngest_subpop, len(youngest_subpop), method='pairs')

                nc = np.random.randint(1, 4, len(crosses))

                # Fix to prevent subpopulations from going to one
                if len(youngest_subpop) <= 3 or i == (num_timesteps - 1):
                    nc[nc == 1] = 2

                # Fix to maintain a minimum pop size
                nc = nc + int(np.ceil(tack_on / float(len(nc))))

                experiment.make_crosses(crosses, num_children=nc)

            global_generation_count += 1
            generation_nonce += 1
            current_gen = experiment.get_generation(global_generation_count)

            # Add individuals to population graph
            for ind in current_gen:
                population_graph.add_node(ind.id, individual=ind)
                population_graph.add_edge(ind.dam, ind.id)
                population_graph.add_edge(ind.sire, ind.id)

            # Add subpopulations
            cond = ((np.random.rand() > p_split) or len(current_gen) < 20) if do_splitting else True

            if cond:
                # All individuals are staying in this subpop
                subpopulations[j].append(generation_nonce)
                generations_graph.add_node(generation_nonce, ids=[ind.id for ind in current_gen])
                generations_graph.add_edge(youngest_subpop_uid, generation_nonce)
            else:
                halfers = int(len(current_gen) / 2)
                # Those staying make a new generation in this subpop
                staying = current_gen[:halfers]
                subpopulations[j].append(generation_nonce)
                generations_graph.add_node(generation_nonce, ids=[ind.id for ind in staying])

                generations_graph.add_edge(youngest_subpop_uid, generation_nonce)

                # Those going create a new generation in a new subpop
                subpopulation_nonce += 1
                new_subpops.append(subpopulation_nonce)
                if verbose:
                    print('Adding subpopulation {0}'.format(subpopulation_nonce))

                generation_nonce += 1
                subpopulations[subpopulation_nonce] = [generation_nonce]

                going = current_gen[halfers:]
                generations_graph.add_node(generation_nonce, ids=[ind.id for ind in going])
                generations_graph.add_edge(youngest_subpop_uid, generation_nonce)

        valid_subpops.extend(new_subpops)

        # Trim everyone not in the newest subpops to save memory
        to_keep_subpops = [subpopulations[j][-1] for j in valid_subpops]
        to_keep_ids = []
        for j in to_keep_subpops:
            to_keep_ids.extend(generations_graph.nodes[j]['ids'])
        experiment.trim_X(to_keep_ids)

        # Also have to do the same for all the individuals in the population graph
        for cid in population_graph.nodes:
            if cid not in to_keep_ids:
                population_graph.nodes[cid]['individual'].genotype = None

    latest_generations_idx = [subpopulations[j][-1] for j in valid_subpops]

    all_inds = []

    for k in latest_generations_idx:
        all_inds.extend(generations_graph.nodes[k]['ids'])

    output = [population_graph.nodes[iid]['individual'] for iid in all_inds]

    total_pop_size = np.sum([len(generations_graph.nodes[subpopulations[j][-1]]['ids']) for j in valid_subpops])
    if verbose:
        print('Population size: {0}'.format(total_pop_size))

    return output
