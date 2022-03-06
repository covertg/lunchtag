# One could imagine neat extensions or alternatives to the rejection sampling approach. For
# example, we could try using graph theory to generate assignments that minimizes the path length
# between each person (i.e. intentionally create "small world" lunch tag).

def assign_rejection_sampling(signups, previous_assignments=[], group_size=2, stratify_by=None,
                              allow_overflow=True, max_attempts=5000):
    """Create new lunch tag assignments using the "rejection sampling"
    approach of randomly shuffling groups until a suitable assignment is
    made.

    :param signups: Pandas dataframe of signups in the format given by
        `read_signups`.
    :param previous_assignments: Previous assignments (if available) in
        the format given by `read_assignment` or `read_assignments`, a
        list of Python sets enumerating all previous assignments.
    :param group_size: Size of group to assign each participant to. If
        `allow_overflow` is True (or True for a given strata), then this
        is effectively the *minimum* group size.
    :param stratify_by: Optional string representing an attr (column in
        the original signups csv) by which to stratify the random
        assignments. Currently, only one-to-one stratifying is
        supported; i.e., the assignment process will try to create even
        mixtures of all strata in each lunchtag group.        
    :param allow_overflow: bool, or a dict mapping the name of each
        strata to a bool. If True (or True for a given strata), then
        "leftover" participants (those who could not be assigned to a
        group of size exactly `group_size`) will be redistributed among
        the existing groups.
    :param max_attempts: Maximum number of attempts to sample a suitable
        random assignment.
    """
    # This is NOT the most efficient way to do this (see sim.ipynb), but it is easy to iterate on
    # and is still quite fast.
    # TODO some day allow for "underflow" as well.
    # TODO do a better job of tracking attrs of participants through the shuffling process rather than just IDs. maybe go more object-oriented.
    if previous_assignments:
        assert all([type(a) == set for a in previous_assignments])
    assert group_size <= signups.shape[0]
    
    # Setup
    if stratify_by:
        assert stratify_by in signups.columns
        strata = list(signups.groupby(stratify_by))
        strata_names, strata_dfs = list(zip(*strata))
    else:
        strata_names, strata_dfs = ['all'], [signups]
    assert group_size % len(strata_names) == 0
    if type(allow_overflow) == bool:
        allow_overflow = {name: allow_overflow for name in strata_names}
    else:
        assert set(strata_names) == set(allow_overflow.keys())
    nrows = [df.shape[0] for df in strata_dfs]
    n_groups = min(nrows) // group_size
    subgroup_size = group_size // len(strata_names)  # One to one mapping only rn.
    accept_proposal = False
    attempt = 0

    # Do the deed
    while not accept_proposal and attempt < max_attempts:
        attempt += 1
        shuffled_subgroups = [_shuffle_and_group(df, n_groups, subgroup_size, allow_overflow[name]) for name, df in zip(strata_names, strata_dfs)]
        proposal, leftover = list(zip(*shuffled_subgroups))
        assert all([len(p) == len(proposal[0]) for p in proposal])
        # This is a bit trippy but it works. `proposal` is in the form:
        # tuple(list(set1, set2, ..., set_n_groups),
        #       list(set1, set2, ..., set_n_groups)),
        #       ...)
        # Where each list represents one stratum, and each vertically-aligned slice of sets represents the
        # assignments that we want to merge together to make one assignment which is nicely stratified.
        # I.e. we want to end up with:
        #       list(set1, set2, ..., set_n_groups)
        # Which is basically a vertical merge of the above. So:
        proposal = [set.union(*sets) for sets in zip(*proposal)]
        # Furthermore we flatten `leftover`, which currently is in the form:
        #       tuple(set1, set2, ..., set_n_groups)
        leftover = [participant for l in leftover for participant in l]
        # An "acceptable" proposal does not repeat any assignments within `previous_assignments`.
        repeats = _get_repeats(proposal, previous_assignments)
        accept_proposal = len(repeats) == 0
    
    # Fin
    if not accept_proposal:
        print(f'! Failed to create new assignments after {attempt} tries. There will be repeated assignments: {repeats}')
    else:
        print(f'Made assignments in {attempt} {"try" if attempt == 1 else "tries"}.')
    if leftover:
        print(f'{len(leftover)} leftover participants were not assigned to a group: {leftover}')
    return proposal

def _shuffle_and_group(df, n_groups, k, allow_overflow):
    """
    :param n_groups: Exact number of assignment groups
    :param k: Minimum number of participants per group (exact number if
        `allow_overflow` is False)
    """
    min_participants = k * n_groups
    assert df.shape[0] >= min_participants  # Currently not supporting "underflow"
    # Propose a random assignment
    shuffle = df.sample(frac=1, ignore_index=True)
    # Convert shuffled df to a list of sets of unique identifiers, up to `n`
    ids = shuffle['ID'].to_list()
    sets = []
    for i in range(0, n_groups * k, k):
        sets.append(set(ids[i:i+k]))
    # Account for "overflow" either by redistributing the remaining participants, or just passing
    # them to the caller via leftover.
    leftover = {}
    if min_participants < df.shape[0]:
        overflow = ids[min_participants:]
        if allow_overflow:
            for j, id in enumerate(overflow):
                sets[j % len(sets)].add(id)
        else:
            leftover = set(overflow)  # Set just for consistency, there should be no repeats anyways
    return sets, leftover

def _get_repeats(proposal, prev_assignments):
    out = []
    for prop in proposal:
        for prev in prev_assignments:
            # prop and prev are both sets representing one "assignment." Their intersection should
            # always have at most 1 person if prop is indeed a new assignment.
            inter = prop.intersection(prev)
            if len(inter) >= 2:
                out.append(inter)
    return out