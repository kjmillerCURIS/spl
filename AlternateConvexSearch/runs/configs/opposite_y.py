params = {
    'data_path' : 'data',
    'proteins' : [ '052', '074', '108', '131', '146' ],
    'folds' : [ 1, 2, 3, 4, 5 ],
    'seeds' : [ '0000', '1000', '2000', '3000' ],
    
    'name' : 'opposite_y',
    'extension' : '2011-05-03', # Set this to 'datetime' to make a new run or a specific string to continue an old one
    'description' : """
This control run covers cccp, slack, and uncertainty for the existing algorithms.
        """,
    
    # Each entry is a pair of parameter sets; one set is for training, and the other is the corresponding set for inference
    'param_pairs' : {
        'cccp' : ['-c 150 -k 0 -m 1.3 --t 1', ''],
        'slack' : ['-c 150 -k 100 -m 1.3 -f 0.55 --t 1', ''],
        'uncertainty' : ['-c 150 -k 100 -m 1.3 -f 0.55 -u 1 --t 1', '']
    },
    
    'raise_errors' : True
}