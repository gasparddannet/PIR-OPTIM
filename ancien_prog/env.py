"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):

        ### env_init ###
        # self.long_range = (1, 50)
        # self.lat_range = (1, 30)

        ### paris - chicago ###
        # self.long_range = (-90, 3)
        # self.lat_range = (40, 60)

        ### JFK - SIN ###
        # self.long_range = (-83, 113)
        # self.lat_range = (-9, 90)

        ### TLS - YUL ###
        # self.long_range = (-74, 1.5)
        # self.lat_range = (43.5, 53)

        ### TLS - KRK ###
        # self.long_range = (1, 20)
        # self.lat_range = (40, 55)

        ### CDG - RVN ###
        # self.long_range = (2, 26)
        # self.lat_range = (48, 67)       

        ### CDG - HDN ###
        self.long_range = (2.5, 140)
        self.lat_range = (35, 79)

        ### CDG - LAX ###
        # self.long_range = (-118.5, 2.7)
        # self.lat_range = (33.5, 80)        

        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.obs_ellipse = self.obs_ellipse()

    @staticmethod
    def obs_boundary():
        obs_boundary = [

            ### env_init ###
            # [0, 0, 1, 30],
            # [0, 30, 50, 1],
            # [1, 0, 50, 1],
            # [50, 1, 1, 30]

            ### paris - chicago ###
            # [-91, 39, 1, 21],
            # [-91, 60, 94, 1],
            # [-91, 39, 95, 1],
            # [3, 40, 1, 21]

            ### JFK - SIN ###
            # [-180, -90, 1, 180],
            # [-180, 90, 360, 1],
            # [180, -90, 1, 180],
            # [-180, -90, 360, 1]

        ]


        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [

            ### env_init ###

            # [14, 12, 8, 2],
            # [18, 22, 8, 3],
            # [25, 7, 2, 12],
            # # [26, 7, 2, 12],
            # #[32, 14, 10, 2]
            #------------------------#
            # [6, 10, 2, 10],
            # [25, 12, 3, 16]

            

            # [-94, 35, 4, 35]


        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [

            ### env_init ###

            # [7, 12, 3],
            # [46, 20, 2],
            # [15, 5, 2],
            # # [37, 7, 3],
            # [41, 7, 3],
            # [37, 23, 3]
            #------------------------#
            # [17, 23, 3],
            # [16, 7, 4]

        ]

        return obs_cir

    @staticmethod
    def obs_ellipse():
        obs_ellipse = [

            ### env_init ###

            # ## [27, 13, 2, 6]
            # [32, 12, 2, 6]
            # ## [36, 14, 6, 2]
            #------------------------#
            # [37, 10, 3, 6],
            # [40, 26, 7, 2]


            ### CDG - HDN ###
            ### [63, 68, 8, 4]
            ### [63, 66, 4, 8]
            [68.4, 68.75, 8, 4],
            [105, 61, 5, 9]


            ### CDG - LAX ###
            # [-55, 62, 10, 7],
            # [-104, 48, 2, 20]

            ## [-98, 50, 12, 10]
            ## [-92, 54, 2, 15]

        ]
        return obs_ellipse