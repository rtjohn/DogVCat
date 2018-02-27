# Directory setup
originalData <- "~/RWork/originalDvcTrain"
baseDir <- "~/RWork/dvcSmall"
dir.create(baseDir)

# Create the directories for the 3 main splits
trainDir <- file.path(baseDir, 'train')
dir.create(trainDir)
valDir <- file.path(baseDir, 'val')
dir.create(valDir)
testDir <- file.path(baseDir, 'test')
dir.create(testDir)

# Split the directories based on class
trainCatsDir <- file.path(trainDir, 'cats')
dir.create(trainCatsDir)
trainDogsDir <- file.path(trainDir, 'dogs')
dir.create(trainDogsDir)

valCatsDir <- file.path(valDir, 'cats')
dir.create(valCatsDir)
valDogsDir <- file.path(valDir, 'dogs')
dir.create(valDogsDir)

testCatsDir <- file.path(testDir, 'cats')
dir.create(testCatsDir)
testDogsDir <- file.path(testDir, 'dogs')
dir.create(testDogsDir)

# Move the files into place
fnames <- paste0('cat.', 1:1000, '.jpg')
file.copy(file.path(originalData, fnames), file.path(trainCatsDir))
fnames <- paste0('cat.', 1001:1500, '.jpg')
file.copy(file.path(originalData, fnames), file.path(valCatsDir))
fnames <- paste0('cat.', 1501:2000, '.jpg')
file.copy(file.path(originalData, fnames), file.path(testCatsDir))

fnames <- paste0('dog.', 1:1000, '.jpg')
file.copy(file.path(originalData, fnames), file.path(trainDogsDir))
fnames <- paste0('dog.', 1001:1500, '.jpg')
file.copy(file.path(originalData, fnames), file.path(valDogsDir))
fnames <- paste0('dog.', 1501:2000, '.jpg')
file.copy(file.path(originalData, fnames), file.path(testDogsDir))



