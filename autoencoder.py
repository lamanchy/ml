# coding=utf-8

import numpy as np
from image import Image
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt


def load_dataset():
    Image.load_images(max_images=None, pca_dimensions=None)
    dataset = [image.features for image in Image.get_images()]
    train_dataset = dataset[:300]
    test_dataset = dataset[300:]

    return train_dataset, test_dataset


def build_autoencoder(size_of_input=512, optimizer='adadelta', loss='binary_crossentropy'):
    encoding_dim = 128
    encoding_dim2 = 32
    size_of_feature_vector = size_of_input

    input_img = Input(shape=(size_of_feature_vector,))

    encoded = Dense(encoding_dim, activation='relu')(input_img)
    encoded2 = Dense(encoding_dim2, activation='sigmoid')(encoded)

    decoded = Dense(encoding_dim, activation='sigmoid')(encoded2)
    decoded2 = Dense(size_of_feature_vector, activation='relu')(decoded)

    # maps an input to its reconstruction
    autoencoder = Model(input_img, decoded2)

    # maps an input to its encoded representation
    #encoder = Model(input_img, encoded2)

    # create a placeholder for an encoded (32-dimensional) input
    #encoded_input = Input(shape=(encoding_dim2,))
    #decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    #decoder = Model(encoded_input, decoder_layer(encoded_input))

    # model configuration
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder


def train(autoencoder, train_dataset, test_dataset):
    history = autoencoder.fit(train_dataset,
                    train_dataset,
                    epochs=100,
                    batch_size=len(train_dataset),
                    shuffle=True,
                    validation_data=(test_dataset, test_dataset),
                    ).history

    return history


def model_loss_plot(autoencoder):
    plt.plot(autoencoder['loss'])
    plt.plot(autoencoder['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    # plt.show()  # show plot in new window or ...
    plt.savefig('model_loss.jpg')  # ... save plot


def test_predict_image(autoencoder, test_image):
    predictions = autoencoder.predict(np.array([test_image]))
    mse = np.mean(np.power(test_image - predictions, 2), axis=1)
    return mse


autoencoder = build_autoencoder()
train_dataset, test_dataset = load_dataset()
history = train(autoencoder, np.array(train_dataset), np.array(test_dataset))
model_loss_plot(history)

test_image = [0.0, 0.0, 0.0, 0.0, 1.8804943561553955, 0.7069336175918579, 0.22960565984249115, 0.4167569875717163, 0.5878481864929199, 0.0, 0.29236477613449097, 0.05501871556043625, 2.103684663772583, 0.18599669635295868, 0.22236040234565735, 1.3819539546966553, 0.0, 5.257460117340088, 1.644950032234192, 6.216856002807617, 3.1918089389801025, 0.27127736806869507, 0.04838934540748596, 0.0150375971570611, 0.4121567904949188, 9.812552452087402, 3.169222831726074, 1.1960031986236572, 12.583051681518555, 3.048166275024414, 4.002936840057373, 0.10063967108726501, 1.4095697402954102, 0.0, 5.170124053955078, 0.8049582839012146, 0.0, 0.0, 0.3788950741291046, 0.0, 11.188004493713379, 6.644944190979004, 0.6077565550804138, 0.0, 0.5871887803077698, 0.9757327437400818, 4.374422550201416, 0.6029790043830872, 24.111677169799805, 2.399060010910034, 0.9249032735824585, 7.494615077972412, 4.359602451324463, 3.4691922664642334, 0.14129602909088135, 0.10547368228435516, 0.6477589011192322, 24.93417739868164, 6.942101955413818, 0.0, 0.0, 0.6729903817176819, 0.5599526166915894, 16.769155502319336, 0.19585227966308594, 0.18705448508262634, 0.0, 2.027977705001831, 12.086137771606445, 1.8187229633331299, 0.09887564182281494, 0.0, 7.938812732696533, 2.867079496383667, 4.917279243469238, 1.9172372817993164, 3.108444929122925, 8.168086051940918, 18.725933074951172, 1.7848081588745117, 0.0, 1.291458249092102, 5.7026472091674805, 0.0, 0.3585396111011505, 4.060442924499512, 4.560173988342285, 0.0, 0.26919636130332947, 1.742539644241333, 0.07751485705375671, 0.0, 8.913522720336914, 0.0, 9.880024909973145, 0.0, 1.6577610969543457, 0.39396148920059204, 0.0, 0.2579393684864044, 1.645795226097107, 0.16297392547130585, 0.31148698925971985, 1.7641394138336182, 0.5090489387512207, 0.00747087225317955, 0.0, 24.067790985107422, 3.483781576156616, 0.0, 0.55499666929245, 0.0, 21.36237907409668, 1.3149563074111938, 0.0, 10.736258506774902, 2.7254629135131836, 0.9087203741073608, 7.632226467132568, 8.775213241577148, 1.900132179260254, 0.9228505492210388, 0.0, 4.570855140686035, 3.216202974319458, 8.615860939025879, 0.0, 0.642212986946106, 0.14217810332775116, 0.0, 25.17415428161621, 0.7180546522140503, 0.6815358400344849, 0.0, 0.9574713706970215, 0.0, 2.3726494312286377, 0.0, 3.8415520191192627, 1.5036367177963257, 3.404344320297241, 0.0, 1.2301899194717407, 0.7012369632720947, 19.88656234741211, 0.0, 2.1454927921295166, 0.9342029094696045, 0.020051289349794388, 1.5228164196014404, 3.2177681922912598, 0.0, 2.151061773300171, 0.15062391757965088, 1.4282398223876953, 5.079799175262451, 0.0, 5.714853763580322, 0.0, 0.3816084563732147, 0.0, 3.439964771270752, 3.842221975326538, 3.710348129272461, 0.31456032395362854, 0.07970570772886276, 0.0, 0.0, 0.38404425978660583, 0.0, 0.887657105922699, 0.0, 0.31945517659187317, 4.484966278076172, 0.42090535163879395, 1.2503970861434937, 2.4257993698120117, 2.4690113067626953, 1.110868215560913, 0.0, 0.08851050585508347, 0.830285370349884, 0.0, 0.4064304232597351, 16.787961959838867, 0.026361368596553802, 0.0, 1.163230299949646, 1.8646231889724731, 1.2714463472366333, 14.058565139770508, 20.562171936035156, 1.4034490585327148, 0.17165836691856384, 9.904515266418457, 0.0, 0.05095704272389412, 0.16356657445430756, 0.0, 0.0, 0.0, 15.006237030029297, 0.7549220323562622, 0.08223549276590347, 6.545355796813965, 12.04909610748291, 0.9507253170013428, 0.0, 0.2033447027206421, 0.5788671374320984, 0.4070279598236084, 2.374098062515259, 7.909828186035156, 0.028033262118697166, 0.0, 3.4285800457000732, 7.283825874328613, 0.0, 0.0, 2.65425443649292, 0.0, 0.8893184065818787, 0.0, 0.0, 0.3111889958381653, 1.5850036144256592, 3.2993412017822266, 0.255258709192276, 6.7240729331970215, 0.7683354020118713, 13.769340515136719, 0.0, 0.5023338198661804, 0.13599616289138794, 4.294544219970703, 0.0, 1.1544216871261597, 22.34720230102539, 0.0, 0.0, 0.27754226326942444, 0.0, 1.0748155117034912, 0.0, 15.226280212402344, 2.7827649116516113, 0.8164669871330261, 0.4607756435871124, 0.611034631729126, 8.458634376525879, 12.175947189331055, 0.6538210511207581, 2.3571853637695312, 5.381983757019043, 0.4960768520832062, 0.9277026057243347, 3.741736888885498, 3.077821969985962, 1.8859360218048096, 3.162126302719116, 0.5656725168228149, 3.1537976264953613, 2.651132345199585, 0.5686882138252258, 0.36464616656303406, 0.0, 2.5565690994262695, 1.9530521631240845, 0.0, 14.551795959472656, 0.0, 0.0, 6.616374969482422, 1.3920576572418213, 0.0, 0.0, 1.3248099088668823, 1.6788133382797241, 0.0, 5.402541160583496, 0.3910617530345917, 2.548312187194824, 3.6670520305633545, 31.27107810974121, 2.439645290374756, 0.0, 0.0, 8.502395629882812, 0.0, 3.4688994884490967, 1.2416741847991943, 2.3131215572357178, 3.209397315979004, 0.7329533100128174, 3.422468423843384, 0.13265573978424072, 2.355255126953125, 0.0, 0.0, 14.606285095214844, 3.580601215362549, 0.0, 0.8827252984046936, 0.0, 0.25343701243400574, 0.06305327266454697, 1.67046058177948, 2.638289213180542, 0.0, 0.0, 2.3739094734191895, 1.240071177482605, 0.0, 0.0, 0.0, 8.52182388305664, 0.0, 0.1819218099117279, 2.061605453491211, 3.6668622493743896, 0.19788962602615356, 0.7510825395584106, 0.16189579665660858, 0.0, 10.169921875, 0.0, 4.389651298522949, 1.2549371719360352, 2.5218827724456787, 0.0, 0.18030178546905518, 7.127324104309082, 2.631471872329712, 0.0, 5.937015533447266, 0.0, 0.0, 16.024349212646484, 1.9892821311950684, 3.5586071014404297, 1.8390382528305054, 9.2930908203125, 5.587996959686279, 6.402929306030273, 3.4297893047332764, 0.9935498833656311, 0.2036476582288742, 5.119307041168213, 0.5637862682342529, 0.0, 2.28129243850708, 1.971710205078125, 0.0, 1.2329974174499512, 0.0, 0.0, 0.0, 0.0, 2.180539608001709, 0.0, 0.06656529754400253, 0.0, 1.4375454187393188, 0.26542794704437256, 0.12420336157083511, 0.092351533472538, 28.80671501159668, 0.0, 3.3187406063079834, 1.5600481033325195, 0.4630994200706482, 0.0, 2.091104030609131, 0.2738979756832123, 18.787372589111328, 0.7383707165718079, 0.0, 0.26092728972435, 7.927730083465576, 9.737685203552246, 0.0, 2.7787232398986816, 1.5658581256866455, 0.0, 0.0, 0.8108332753181458, 1.0672250986099243, 0.8623135089874268, 0.8694003224372864, 4.857641696929932, 1.3110747337341309, 0.0, 0.09498541057109833, 2.234389543533325, 0.026573175564408302, 4.009541034698486, 1.0371092557907104, 0.0, 8.648454666137695, 0.0913550928235054, 0.0, 4.306004524230957, 1.1662076711654663, 0.0, 6.657422065734863, 0.2660459876060486, 0.9924789667129517, 5.025753021240234, 3.0183308124542236, 36.07769012451172, 2.5385520458221436, 0.0, 5.198990345001221, 9.977095603942871, 0.0, 1.06551194190979, 6.543551921844482, 4.670483112335205, 0.46150219440460205, 17.303436279296875, 12.724296569824219, 1.0349559783935547, 0.8894709348678589, 0.3003994822502136, 4.092144012451172, 0.0, 0.8502053022384644, 3.992276668548584, 4.529833793640137, 0.0, 2.55846905708313, 0.4472121000289917, 0.0, 0.0, 0.0, 4.2365312576293945, 0.3097991645336151, 0.0, 1.6837687492370605, 10.145158767700195, 0.7415428757667542, 2.934221029281616, 6.405832767486572, 6.358782768249512, 0.09216213971376419, 9.638236045837402, 0.0, 0.0, 0.00999804399907589, 0.0, 20.597726821899414, 0.0, 0.0, 2.211881160736084, 0.6253512501716614, 0.7937774658203125, 0.4366821050643921, 4.442399501800537, 0.0, 0.5956621170043945, 13.48652458190918, 0.0, 0.0, 1.4457333087921143, 0.22435523569583893, 9.062491416931152, 0.0, 0.07841304689645767, 2.9407951831817627, 4.9325456619262695, 0.0, 1.756137728691101, 0.0, 0.0, 0.028093751519918442, 3.355412006378174, 2.574017286300659, 2.5484931468963623, 3.834578275680542, 0.0, 0.35610508918762207, 2.017357587814331, 0.8423983454704285, 14.403274536132812, 1.8278242349624634, 0.3564389944076538, 6.613706111907959, 0.3129180073738098, 0.0, 2.1026413440704346, 0.0, 0.9723974466323853, 6.596174716949463, 2.2892961502075195, 0.0, 1.2007434368133545, 0.0022939674090594053, 1.3918628692626953, 6.383819103240967, 0.0, 8.452208518981934, 0.0, 1.0455684661865234, 7.294814109802246, 1.0880488157272339, 0.0, 0.047486189752817154, 0.0, 10.663717269897461, 0.0, 0.9801774024963379, 4.902711391448975]
print(test_predict_image(autoencoder, test_image))