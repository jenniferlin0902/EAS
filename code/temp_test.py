import data_providers.mnist
import data_providers.svhn
#svhn = data_providers.svhn.SVHNDataProvider()
#svhn.get_images_and_labels("test")

mnist = data_providers.mnist.MNISTDataProvider()
mnist.get_images_and_labels("train")