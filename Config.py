class Test:
    image_dataset = "/home/emmavico/Documents/Posa_Testa/Test/*"
    working_directory = "./Dataset/Test/"


class Debug:
    active = False


image_dataset = "/home/emmavico/Documents/Posa_Testa/biwi_rgb_renamed/*/*" if not Debug.active else Test.image_dataset
working_directory = "./Dataset/" if not Debug.active else Test.working_directory
n_of_threads = 2


class ImageVisualizer:
    edge_color = (0, 255, 0)
    edge_thickness = 1
    landmark_color = (0, 0, 255)
    landmark_thickness = 2
    landmark_text_height = 0.7


class Extraction:
    scale_landmarks = True
    static_image_mode = True
    max_num_faces = 1
    edges_per_landmark = 5

    class LandmarksToIgnore:
        right_eye = [336, 296, 334, 293, 300, 383, 265, 261, 448, 449, 450, 451, 452, 453, 464, 413, 441,
                     285, 295, 282, 283, 276, 353, 446, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 442, 443,
                     444, 445, 342, 359,
                     263, 249, 373, 374, 380, 381, 382, 362, 398, 384,
                     258, 257, 259, 260, 467,
                     385, 386, 387, 388, 466, 246]

        left_eye = [107, 55, 221, 180, 244, 233, 232, 231, 230, 229, 228, 31, 25, 156, 70, 63, 105, 66,
                    65, 52, 53, 46, 124, 226, 113, 225, 224, 223, 222,189,
                    28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25, 130, 247, 30, 29, 27,
                    159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33, 256, 161, 160, 390]

        jaw = [136, 150, 149, 156, 148, 152, 377, 400, 378, 379, 365,
               135, 169, 170, 140, 171, 175, 396, 369, 395, 394, 364, 214, 210, 211, 32, 208, 199, 428, 262, 431, 430,
               434,
               212, 202, 204, 194, 201, 200, 418, 424, 422, 432, 176, 421, 182, 335]
        mouth = [426, 436, 52, 43, 106, 83, 18, 313, 406, 273, 287, 410, 322, 291, 293, 164, 167, 165, 92, 186,
                 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 270, 0, 37, 39, 40, 185,
                 76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 206, 408, 304, 303, 302, 11, 72, 73, 74,
                 62, 96, 89, 179, 86, 15, 316, 403, 312, 325, 293, 407, 272, 271, 268, 12, 38, 41, 42, 183,
                 78, 95, 88, 178, 87, 14, 314, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 317, 319,
                 306, 292, 184, 57, 267, 393, 391, 216]

        total_landmarks = left_eye + right_eye + mouth + jaw
