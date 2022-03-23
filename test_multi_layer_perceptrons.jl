include("multi_layer_perceptrons.jl")
include("dataset.jl")

function testMultiLayerPerceptrons()
    
    println("set configs")

    rng = MersenneTwister(123)

    patterns = 2
    trainN = 4
    testN = 4
    nIn = 2
    nHidden = 3
    nOut = patterns

    epochs = 500
    learningRate = 0.1

    minibatchSize = 1 # here, we do on-line training
    minibatchN = Int(trainN / minibatchSize)

    println("make dataset")
    trainX, trainT = makeXORDataset()
    testX, testT = makeXORDataset()

    println("make minibatches")
    minibatchIndex = [i for i in 1:trainN]
    shuffle(rng, minibatchIndex)

    minibatchTrainX = zeros(minibatchN, minibatchSize, nIn)
    minibatchTrainT = zeros(minibatchN, minibatchSize, nOut)
    for i in 1:minibatchN
        for j in 1:minibatchSize
            index = minibatchIndex[(i - 1) * minibatchSize + j] # NOTE
            minibatchTrainX[i, j, :] = trainX[index, :]
            minibatchTrainT[i, j, :] = trainT[index, :]
        end
    end

    # build MLP model

    # construct
    println("construct MLP")
    classifier = MultiLayerPerceptrons(nIn, nHidden, nOut, rng)

    println("train")
    for epoch in 1:epochs
        println("epoch=$epoch")
        for batch in 1:minibatchN
            train!(classifier,
                   minibatchTrainX[batch, :, :],
                   minibatchTrainT[batch, :, :],
                   minibatchSize, learningRate)
        end
    end

    println("test")
    predictedT = zeros(testN, nOut)
    for i in 1:testN
        predictedT[i, :] = predict(classifier, testX[i, :])
    end

    println("evaluate")
    confusionMatrix = zeros(patterns, patterns)
    accuracy = 0.
    precision = zeros(patterns)
    recall = zeros(patterns)

    for i in 1:testN
        predicted = findall(x -> x == 1, predictedT[i, :]) # find the position of the value 1
        actual = findall(x -> x == 1, testT[i, :]) # find the position of the value 1
        col = predicted[1] # not [0]
        row = actual[1] # not [0]
        confusionMatrix[row, col] += 1
    end

    for i in 1:patterns
        col, row = 0, 0

        for j in 1:patterns
            if i == j
                accuracy += confusionMatrix[i, j]
                precision[i] += confusionMatrix[j, i] # NOTE
                recall[i] += confusionMatrix[i, j] 
            end
            
            col += confusionMatrix[j, i] # NOTE
            row += confusionMatrix[i, j]
        end

        precision[i] /= col
        recall[i] /= row
    end
    accuracy /= testN

    println("MLP model evaluation")
    println("Accuracy: $(accuracy * 100.)")
    println("Precision:")
    for i in 1:patterns
        println("class: $i: $(precision[i] * 100.)")
    end
    println("Recall:")
    for i in 1:patterns
        println("class: $i: $(recall[i] * 100.)")
    end
end

testMultiLayerPerceptrons()