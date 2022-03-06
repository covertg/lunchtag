import pandas as pd
from pathlib import Path

def read_signups(filename, id, attrs=None):
    """Parses an input file of signups for a new round of lunch tag into
    a pandas DataFrame.

    :param filename: Filename of a signup file in comma-separated format
        (CSV). File must have a header row of column names.
    :param id: Column name to use as each person's unique identifier.
        E.g. email address.
    :param attrs: Column name or names in the signup file of any other
        attributes to preserve for the assignment process. E.g.
        group/strata.
    :return: A dataframe with 1 or more columns. One column labeled `ID`
        gives each participant's unique identifier; other columns
        correspond to each specified attribute in `attrs`.
    """
    # Read and verify signups
    df = pd.read_csv(filename)
    if id not in df.columns:
        raise ValueError(f'Column "{id}" not found in input file {filename}')
    dups = df.duplicated(subset=id)
    if dups.any():
        raise ValueError(f'Duplicate signup(s): {df[dups][id].to_list()}. Please fix this in the '
                          'signups file.')
    # Transform the columns by renaming identifier to ID and dropping all other non-attrs columns.
    df = df.rename(columns={id: 'ID'})
    keep_cols = ['ID']
    if attrs:
        keep_cols += attrs if type(attrs) in [list, tuple] else [attrs]
    df = df[keep_cols]
    df = df.astype('string')
    print(f'Considering {df.shape[0]} signups for this draw:')
    print(df)
    return df

def save_assignment(assignment, filename):
    assert all([type(a) == set for a in assignment])
    df = pd.DataFrame([','.join(list(a)) for a in assignment], columns=['IDs'])
    df.to_csv(filename, index=False)
    print(f'Wrote {sum([len(a) for a in assignment])} participants to {len(assignment)} groups.')

def _read_assignment(filename):
    df = pd.read_csv(filename)
    assert 'IDs' in df.columns
    return df['IDs'].apply(lambda s: set(s.split(',')))

def read_assignments(glob):
    out = []
    files = list(Path('.').glob(glob))
    for f in files:
        assignments = _read_assignment(f)
        out.extend(assignments)
    flattened = [person for assignment in out for person in list(assignment)]
    print(f'Read {len(out)} assignments of {len(set(flattened))} unique participants from {len(files)} file(s): {[str(f) for f in files]}')
    return out