using Random

include("dataset.jl")
include("logistic_regression.jl")

function testLogisticRegression()
    
    println("set configs")

    rng = MersenneTwister(1234)

    patterns = 3

    trainSize = 400
    trainN = trainSize * patterns

    testSize = 60
    testN = testSize * patterns

    nIn = 2
    nOut = patterns

    epochs = 100
    learningRate = 0.2

    minibatchSize = 50 # number of data in each minibatch
    minibatchN = Int(trainN / minibatchSize) # number of minibatches

    # class1 inputs x11 and x12: x11 ~ N(-2.0, 1.0), x12 ~ N(+2.0, 1.0)
    mu11, mu12 = -2.0, 2.0
    answer1 = [1, 0, 0]
    # class2 inputs x21 and x22: x21 ~ N(+2.0, 1.0), x22 ~ N(-2.0, 1.0)
    mu21, mu22 = 2.0, -2.0
    answer2 = [0, 1, 0]
    # class3 inputs x31 and x32: x31 ~ N(0.0, 1.0), x32 ~ N(0.0, 1.0)
    mu31, mu32 = 0.0, 0.0
    answer3 = [0, 0, 1]

    println("make train dataset")
    trainX = zeros(trainN, nIn)
    trainT = zeros(trainN, nOut)
    makeDataset(1, trainSize, mu11, mu12, answer1, trainX, trainT, rng)
    makeDataset(trainSize + 1, trainSize * 2, mu21, mu22, answer2, trainX, trainT, rng)
    makeDataset(trainSize * 2 + 1, trainN, mu31, mu32, answer3, trainX, trainT, rng)

    println("make test dataset")
    testX = zeros(testN, nIn)
    testT = zeros(testN, nOut)
    makeDataset(1, testSize, mu11, mu12, answer1, testX, testT, rng)
    makeDataset(testSize + 1, testSize * 2, mu21, mu22, answer2, testX, testT, rng)
    makeDataset(testSize * 2 + 1, testN, mu31, mu32, answer3, testX, testT, rng)

    println("make minibatches")

    minibatchIndex = [i for i in 1:trainN]
    shuffle(rng, minibatchIndex) # shuffle data index for SGD

    minibatchTrainX = zeros(minibatchN, minibatchSize, nIn)
    minibatchTrainT = zeros(minibatchN, minibatchSize, nOut)
    for i in 1:minibatchN
        for j in 1:minibatchSize
            index = minibatchIndex[(i - 1) * minibatchSize + j] # NOTE
            minibatchTrainX[i, j, :] = trainX[index, :]
            minibatchTrainT[i, j, :] = trainT[index, :]
        end
    end

    println("construct LogisticRegression")
    classifier = LogisticRegression(nIn, nOut)

    println("train")
    for epoch in 1:epochs
        println("eopch=$epoch")
        for batch in 1:minibatchN
            train!(classifier, minibatchTrainX[batch, :, :],
                               minibatchTrainT[batch, :, :],
                               minibatchSize, learningRate)
        end
        learningRate *= 0.95
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

    println("LogisticRegression model evaluation")
    println("Accuracy: $(accuracy * 100.)")
    println("Precision: $(precision * 100.)")
    println("Recall: $(recall * 100.)")
end

testLogisticRegression()