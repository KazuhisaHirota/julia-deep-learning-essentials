using Random

include("dataset.jl")
include("perceptrons.jl")

function testPerceptrons()
    
    println("set configs")
    
    rng = MersenneTwister(1234)
    
    trainN = 1000
    trainSize = Int(trainN / 2)

    testN = 200
    testSize = Int(testN / 2)

    nIn = 2

    epochs = 100
    learningRate = 1. # learning rate can be 1 in perceptrons

    # class1 inputs x11 and x12: x11 ~ N(-2.0, 1.0), x12 ~ N(+2.0, 1.0)
    mu11 = -2.
    mu12 = 2.
    answer1 = 1
    # class2 inputs x21 and x22: x21 ~ N(+2.0, 1.0), x22 ~ N(-2.0, 1.0)
    mu21 = 2.
    mu22 = -2.
    answer2 = -1

    println("make train dataset")
    trainX = zeros(trainN, nIn)
    trainT = zeros(Int, trainN)
    makeDataset(1, trainSize, mu11, mu12, answer1, trainX, trainT, rng)
    makeDataset(trainSize + 1, trainN, mu21, mu22, answer2, trainX, trainT, rng)

    println("make test dataset")
    testX = zeros(testN, nIn)
    testT = zeros(Int, testN)
    makeDataset(1, testSize, mu11, mu12, answer1, testX, testT, rng)
    makeDataset(testSize + 1, testN, mu21, mu22, answer2, testX, testT, rng)    

    println("construct Perceptrons")
    classifier = Perceptrons(nIn)

    println("train")
    epoch = 1
    while true
        println("epoch=$(epoch)")

        classified = 0
        for i in 1:trainN
            classified += train!(classifier, trainX[i,:], # not trainX[i]
                                 trainT[i], learningRate)
        end

        if classified == trainN # all data are classified correctly
            break
        end

        epoch += 1
        if epoch > epochs
            break
        end
    end

    println("test")
    predictedT = zeros(Int, testN)
    for i in 1:testN
        predictedT[i] = predict(classifier, testX[i,:]) # not testX[i]
    end

    println("evaluate")
    confusionMatrix = zeros(Int, 2, 2)
    accuracy = 0.
    precision = 0.
    recall = 0.

    for i in 1:testN
        if predictedT[i] > 0 # Positive
            if testT[i] > 0 # True Positive
                accuracy += 1.
                precision += 1.
                recall += 1.
                confusionMatrix[1, 1] += 1
            else # False Positive
                confusionMatrix[2, 1] += 1
            end
        else # Negative
            if testT[i] > 0 # False Negative
                confusionMatrix[1, 2] += 1
            else # True Negative
                accuracy += 1.
                confusionMatrix[2, 2] += 1
            end
        end
    end

    accuracy /= testN
    precision /= confusionMatrix[1, 1] + confusionMatrix[2, 1] # TP / (TP + FP)
    recall /= confusionMatrix[1, 1] + confusionMatrix[1, 2] # TP / (TP + FN)

    println("Perceptrons model evaluation")
    println("Accuracy: $(accuracy * 100.)")
    println("Precision: $(precision * 100.)")
    println("Recall: $(recall * 100.)")
end

testPerceptrons()