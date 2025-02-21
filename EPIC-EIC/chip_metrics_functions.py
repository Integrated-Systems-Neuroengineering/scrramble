def get_num_cores(params):
    layers = params['layers']

    # compute number of cores needed
    cores_per_param = [max(l1//256, 1) * max(l2//256, 1) for l1, l2 in zip(layers[:-1], layers[1:])]
    print(cores_per_param)
    dense_cores = sum(cores_per_param)
    acc_cores = [max(l2//256, 1) for l2 in layers[1:]]
    acc_cores = sum(acc_cores)

    tot_cores = dense_cores + acc_cores # store this is return parameters

    # same a secondary metrics list
    size_metrics = {
        'dense_cores': dense_cores,
        'accumulator_cores': acc_cores,
        'total_cores': tot_cores,
    }

    return size_metrics

metric_params = {
    'layers': [1024, 2048, 512, 256], # include layer sizes
}

print(get_num_cores(metric_params))
