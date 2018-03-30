def metacontroller(opt):
    opt.optimizer_type = 'sgd'
    for i in range(5):
        opt.learning_rate = 0.01
        range_of_lambda_vals = np.logspace(-3,4, num = 5 )
        # take i'th element in numpy n-d array

        opt.sigma_regularization_multiplier = float(np.random.choice(range_of_lambda_vals,1)[0]) # float( range_of_lambda_vals[i] )
        print('trying SGD for the ', i, ' th time, with lambda= ', opt.sigma_regularization_multiplier)
        print(opt)
        success = main(opt)
        if success:
            print( 'pure sgd was successful on try #' + str(i)  , ' with params '        )
            return
        else:
            print( 'SGD failed')

    opt.optimizer_type = 'adam'
    for i in range(40):
        opt.learning_rate = 0.001

        range_of_lambda_vals = np.logspace(-3,4, num = 40 )
        # take 0'th element in numpy n-d array
        opt.sigma_regularization_multiplier = float(np.random.choice(range_of_lambda_vals,1)[0]) # float( range_of_lambda_vals[i] )

        print('trying ADAM for the ', i, ' th time, with lambda = ', opt.sigma_regularization_multiplier)
        print(opt)
        success = main(opt)
        if success:
            print('adam back to sgd was successful on try #' + str(i), ' with params ')
            return
        else:
            print('Adam failed')



# def metacontroller(opt):
    #
    #     opt.optimizer_type = 'sgd'
    #     for i in range(1):
    #         opt.learning_rate = 0.01
    #         opt.sigma_regularization_multiplier = 100.0
    #         print('trying SGD 1e-2 for the ', i, ' th time.')
    #         print(opt)
    #         success = main(opt)
    #         if success:
    #             print('pure sgd was succesful on try #' + str(i), ' with params ')
    #             return
    #         else:
    #             print('SGD failed')
    #
    #     # for i in range(5):
    #     #     opt.learning_rate = 0.001
    #     #     opt.sigma_regularization_multiplier = np.random.uniform(1.0, 1e3)
    #     #     print('trying SGD 1e-3 for the ', i, ' th time.')
    #     #     print(opt)
    #     #     success = main(opt)
    #     #     if success:
    #     #         print( 'pure sgd was succesful on try #' + str(i)  , ' with params '        )
    #     #         print( opt)
    #     #         return
    #     #     else:
    #     #         print( 'SGD failed')
    #     #
    #     opt.optimizer_type = 'adam'
    #     for i in range(40):
    #         opt.learning_rate = 0.001
    #         opt.sigma_regularization_multiplier = 100.0
    #         print('trying ADAM for the ', i, ' th time.')
    #         print(opt)
    #         success = main(opt)
    #         if success:
    #             print('adam back to sgd was succesful on try #' + str(i), ' with params ')
    #             return
    #         else:
    #             print('Adam failed')
    #
    #     opt.optimizer_type = 'adam'
    #     for i in range(40):
    #         opt.learning_rate = 0.001
    #         opt.sigma_regularization_multiplier = np.random.uniform(1.0, 1e3)
    #         print('random sigma value trying ADAM for the ', i, ' th time.')
    #         print(opt)
    #         success = main(opt)
    #         if success:
    #             print('adam back to sgd was succesful on try #' + str(i), ' with params ')
    #             return
    #         else:
    #             print('Adam failed')


