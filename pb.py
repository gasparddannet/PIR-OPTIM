    # def Cost(self, x_start:Node, x_end:Node) -> float:
    #     result = self.utils.is_collision(x_start, x_end)
    #     if result != False:
    #         # print(len(result))
    #         # if len(result) == 1:
    #         #     pt = result[0]
    #         #     self.collision_set.add(pt)
    #         #     dist_in_obstcl = self.calc_dist(x_start, pt)
    #         #     return 4*dist_in_obstcl + self.calc_dist(pt, x_end)
    #         # print(len(result))

    #         if not self.utils.is_inside_obs(x_start) and not self.utils.is_inside_obs(x_end):
    #             if len(result)==2:
    #                 pt1, t1 = result[0]
    #                 pt2, t2 = result[1]
    #                 self.collision_set.add((pt1, x_start, x_end))
    #                 self.collision_set.add((pt2, x_start, x_end))

    #                 if t1 < t2:
    #                     dist_in_obstcl = self.calc_dist(pt1, pt2)
    #                     return self.calc_dist(x_start, pt1) + self.cost_in_obstcles * dist_in_obstcl + self.calc_dist(pt2, x_end) 
    #                 else:
    #                     dist_in_obstcl = self.calc_dist(pt2,pt1)
    #                     return self.calc_dist(x_start, pt2) + self.cost_in_obstcles * dist_in_obstcl + self.calc_dist(pt1, x_end)
    #             else:
    #                 print("KWA_1 de fmt.py ??", result)
            
    #         elif self.utils.is_inside_obs(x_start) and not self.utils.is_inside_obs(x_end):
    #             if len(result)==1:
    #                 pt = result[0]
    #                 self.collision_set.add((pt, x_start, x_end))
    #                 return self.cost_in_obstcles * self.calc_dist(x_start, pt) + self.calc_dist(pt, x_end)
    #             else:
    #                 print("x_start : ", x_start)
    #                 print("x_end : ", x_end)
    #                 print("KWA_2 de fmt.py ??", result)
    #                 print("\n")

    #         elif not self.utils.is_inside_obs(x_start) and self.utils.is_inside_obs(x_end):
    #             if len(result) == 1:
    #                 pt = result[0]
    #                 self.collision_set.add((pt, x_start, x_end))
    #                 return self.calc_dist(x_start, pt) + self.cost_in_obstcles * self.calc_dist(pt, x_end)
    #             else:
    #                 print("x_start : ", x_start)
    #                 print("x_end : ", x_end)
    #                 print("KWA_3 de fmt.py ??", result)
    #                 print("\n")

    #     if self.utils.is_inside_obs(x_start) and self.utils.is_inside_obs(x_end):
    #         return self.cost_in_obstcles * self.calc_dist(x_start, x_end)
    #     else:
    #         return self.calc_dist(x_start, x_end)






    def indice(self, x, x0):
        a = x - x0
        b = a // self.step
        return int(b)

    def intersect(self, start, end, indice_x, indice_y):
        l = []
        result1 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+self.step+1])
        if result1 != False:
            l.append(result1)
            # (inter_pt, t) = result
            # self.collision_set.add((inter_pt, start, end))
            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
        
        result2 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1])
        if result2 != False:
            l.append(result2)
            # (inter_pt, t) = result
            # self.collision_set.add((inter_pt, start, end))
            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
        
        result3 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+1])
        if result3 != False:
            l.append(result3)
            # (inter_pt, t) = result
            # self.collision_set.add((inter_pt, start, end))
            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)            
        
        result4 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+1])
        if result4 != False:
            l.append(result4)
            # (inter_pt, t) = result
            # self.collision_set.add((inter_pt, start, end))
            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)                  
       
        if l==[]:
            return False
        else:
            (inter_pt, _) = min(l, key=cmp_key)
            self.collision_set.add((inter_pt, start, end))
            return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end)         


    def intersect2(self, start, end, indice_x, indice_y):
        l = []
        result1 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+self.step+1])
        if result1 != False:
            l.append(result1)
            # (inter_pt1, t1) = result1
            # self.collision_set.add((inter_pt1, start, end))
            # return  self.cost_in_obstcles * self.calc_dist(start, inter_pt1) +self.calc_dist(inter_pt1, end)
        
        result2 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1])
        if result2 != False:
            l.append(result2)
            # (inter_pt2, t2) = result2
            # self.collision_set.add((inter_pt2, start, end))
            # return self.cost_in_obstcles * self.calc_dist(start, inter_pt2) + self.calc_dist(inter_pt2, end)
        
        result3 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+1])
        if result3 != False:
            l.append(result3)
            # (inter_pt3, t3) = result3
            # self.collision_set.add((inter_pt, start, end))
            # return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end)            
        
        result4 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+1])
        if result4 != False:
            l.append(result4)
            # (inter_pt4, t4) = result4
            # self.collision_set.add((inter_pt, start, end))
            # return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end)                  
        
        if l==[]:
            return False
        # if len(l) == 1:
        #     (inter_pt, t) = l[0]
        #     return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end)  

        else:
            (inter_pt, _) = min(l, key=cmp_key)
            self.collision_set.add((inter_pt, start, end))
            return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end) 

    def Cost(self, start:Node, end:Node) -> float:
        start_x = start.x
        start_y = start.y
        end_x = end.x
        end_y = end.y

        indice_start_x = self.indice(start_x, self.x_range[0])
        indice_start_y = self.indice(start_y, self.y_range[0])
        indice_end_x = self.indice(end_x, self.x_range[0])
        indice_end_y = self.indice(end_y, self.y_range[0])

        # print("indice_start_x = ", indice_start_x)
        # print("indice_start_y = ", indice_start_y)
        # print("indice_end_x = ", indice_end_x)
        # print("indice_end_y = ", indice_end_y)

        if self.grid[indice_start_x][indice_start_y] == 1 and self.grid[indice_end_x][indice_end_y] == 1:
            return self.cost_in_obstcles * self.calc_dist(start, end)
        
        elif self.grid[indice_start_x][indice_start_y] == 0 and self.grid[indice_end_x][indice_end_y] == 1:
            # 1 intersect
            # result = self.utils.is_intersect_rec2(start, end, [indice_start_x*self.step+1, indice_start_y*self.step+1], [indice_start_x*self.step+1, indice_start_y*self.step+self.step+1])
            # if result != False:
            #     (inter_pt, t) = result
            #     self.collision_set.add((inter_pt, start, end))
            #     return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
           
            # result = self.utils.is_intersect_rec2(start, end, [indice_start_x*self.step+1, indice_start_y*self.step+self.step+1], [indice_start_x*self.step+self.step+1, indice_start_y*self.step+self.step+1])
            # if result != False:
            #     (inter_pt, t) = result
            #     self.collision_set.add((inter_pt, start, end))
            #     return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
            
            # result = self.utils.is_intersect_rec2(start, end, [indice_start_x*self.step+self.step+1, indice_start_y*self.step+self.step+1], [indice_start_x*self.step+self.step+1, indice_start_y*self.step+1])
            # if result != False:
            #     (inter_pt, t) = result
            #     self.collision_set.add((inter_pt, start, end))
            #     return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)            
            
            # result = self.utils.is_intersect_rec2(start, end, [indice_start_x*self.step+self.step+1, indice_start_y*self.step+1], [indice_start_x*self.step+1, indice_start_y*self.step+1])
            # if result != False:
            #     (inter_pt, t) = result
            #     self.collision_set.add((inter_pt, start, end))
            #     return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)                  
            



            ###############################################################################################################
            if abs(end_x - start_x) < abs(end_y - start_y):
                if end_y - start_y<0:
                    signe = -1
                    limit = end_y - signe * self.step/20
                    y = start_y
                    while y>=limit:
                        y = y + signe * self.step/20
                        x = calcul_x(start_x, end_x, start_y, end_y, y)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 1:
                            # inter_pt = Node([x,y])
                            # self.collision_set.add((inter_pt, start, end))
                            result = self.intersect2(start, end, indice_x, indice_y)
                            if result == None:
                                print("1.1")
                            return self.intersect(start, end, indice_x, indice_y)
                            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
                    print("oups 1.1")
                    return self.intersect(start, end, indice_end_x, indice_end_y)

                else:
                    signe = 1
                    limit = end_y - signe * self.step/20
                    y = start_y
                    while y<=limit:
                        y = y + signe * self.step/20
                        x = calcul_x(start_x, end_x, start_y, end_y, y)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        # print("indice_x : ", indice_x)
                        # print("x : ", x)
                        # print("indice_y : ", indice_y)
                        # print("y : ", y)
                        if self.grid[indice_x][indice_y] == 1:
                            # inter_pt = Node([x,y])
                            # self.collision_set.add((inter_pt, start, end))
                            result = self.intersect2(start, end, indice_x, indice_y)
                            if result == None:
                                print("1.2")
                            return self.intersect(start, end, indice_x, indice_y)
                            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
                    print("oups 1.2")
                    return self.intersect(start, end, indice_end_x, indice_end_y)
            else:
                if end_x - start_x<0:
                    signe = -1

                    limit = end_x - signe * self.step/20
                    x = start_x
                    while x>=limit:
                        x = x + signe * self.step/20
                        y = calcul_y(start_x, end_x, start_y, end_y, x)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        # print("indice_x : ", indice_x)
                        # print("x : ", x)
                        # print("indice_y : ", indice_y)
                        # print("y : ", y)
                        if self.grid[indice_x][indice_y] == 1:
                            # print("trouve")
                            # inter_pt = Node([x,y])
                            # self.collision_set.add((inter_pt, start, end))
                            result = self.intersect2(start, end, indice_x, indice_y)
                            if result == None:
                                print("1.3")
                            return self.intersect(start, end, indice_x, indice_y)

                            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
                    print("oups 1.3")
                    return self.intersect(start, end, indice_end_x, indice_end_y)

                else:
                    signe = 1
                    limit = end_x - signe * self.step/20
                    x = start_x
                    while x<=limit:
                        # if start_x == 35.10204730365133 and start_y == 10.385119510236693 and end_x == 35.940942774480135 and end_y == 9.796519186475688:
                        #     print("x : ", x)
                        #     print("limit : ", limit)
                        x = x + signe * self.step/20
                        y = calcul_y(start_x, end_x, start_y, end_y, x)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 1:
                            # inter_pt = Node([x,y])
                            # self.collision_set.add((inter_pt, start, end))
                            result = self.intersect2(start, end, indice_x, indice_y)
                            if result == None:
                                print("1.4")
                            return self.intersect(start, end, indice_x, indice_y)

                            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
                    print("oups 1.4")
                    return self.intersect(start, end, indice_end_x, indice_end_y)


            print("PROBLEME")
            print("start : ", start)
            print("end : ", end)
            print("\n")
            return 0

        elif self.grid[indice_start_x][indice_start_y] == 1 and self.grid[indice_end_x][indice_end_y] == 0:
            #1 intesect

            # result = self.utils.is_intersect_rec2(start, end, [indice_end_x*self.step+1, indice_end_y*self.step+1], [indice_end_x*self.step+1, indice_end_y*self.step+self.step+1])
            # if result != False:
            #     (inter_pt, t) = result
            #     self.collision_set.add((inter_pt, start, end))
            #     return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end)
           
            # result = self.utils.is_intersect_rec2(start, end, [indice_end_x*self.step+1, indice_end_y*self.step+self.step+1], [indice_end_x*self.step+self.step+1, indice_end_y*self.step+self.step+1])
            # if result != False:
            #     (inter_pt, t) = result
            #     self.collision_set.add((inter_pt, start, end))
            #     return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end)
            
            # result = self.utils.is_intersect_rec2(start, end, [indice_end_x*self.step+self.step+1, indice_end_y*self.step+self.step+1], [indice_end_x*self.step+self.step+1, indice_end_y*self.step+1])
            # if result != False:
            #     (inter_pt, t) = result
            #     self.collision_set.add((inter_pt, start, end))
            #     return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end)            
                
            # result = self.utils.is_intersect_rec2(start, end, [indice_end_x*self.step+self.step+1, indice_end_y*self.step+1], [indice_end_x*self.step+1, indice_end_y*self.step+1])
            # if result != False:
            #     (inter_pt, t) = result
            #     self.collision_set.add((inter_pt, start, end))
            #     return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end)   
            


            if abs(end_x - start_x) < abs(end_y - start_y):
            # limit = (indice_end_y - indice_start_y)*self.step
                if end_y - start_y<0:
                    signe = -1

                    limit = end_y - signe * self.step/20
                    y = start_y
                    while y>=limit:
                        y = y + signe * self.step/20
                        x = calcul_x(start_x, end_x, start_y, end_y, y)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 0:
                            # inter_pt = Node([x,y])
                            # self.collision_set.add((inter_pt, start, end))
                            result = self.intersect2(start, end, indice_x, indice_y)
                            if result == None:
                                print("2.1")
                            # print("result : ", result)
                            return self.intersect2(start, end, indice_x, indice_y)
                            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
                    print("oups 2.1")
                    return self.intersect2(start, end, indice_end_x, indice_end_y)

                else:
                    signe = 1
                    limit = end_y - signe * self.step/20
                    y = start_y
                    while y<=limit:
                        y = y + signe * self.step/20
                        # if start_x == 35.74573595490064 and start_y == 15.46802044393078 and end_x == 36.61457033155867 and end_y == 16.661934771989607:
                        #     print("y : ", y)
                        #     print("limit : ", limit)
                        x = calcul_x(start_x, end_x, start_y, end_y, y)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        # print("indice_x : ", indice_x)
                        # print("x : ", x)
                        # print("indice_y : ", indice_y)
                        # print("y : ", y)
                        if self.grid[indice_x][indice_y] == 0:
                            # inter_pt = Node([x,y])
                            # self.collision_set.add((inter_pt, start, end))
                            result = self.intersect2(start, end, indice_x, indice_y)
                            if result == None:
                                print("2.2")
                            return self.intersect2(start, end, indice_x, indice_y)
                            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
                    print("oups 2.2")
                    return self.intersect2(start, end, indice_end_x, indice_end_y)
            else:
                if end_x - start_x<0:
                    signe = -1

                    limit = end_x - signe * self.step/20
                    x = start_x
                    # print("start : ", start)
                    # print("end : ", end)
                    # print("x : ", x)
                    while x>=limit:
                        x = x + signe * self.step/20
                        y = calcul_y(start_x, end_x, start_y, end_y, x)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        # print("indice_x : ", indice_x)
                        # print("x : ", x)
                        # print("indice_y : ", indice_y)
                        # print("y : ", y)
                        if self.grid[indice_x][indice_y] == 0:
                            # print("trouve")
                            # inter_pt = Node([x,y])
                            # self.collision_set.add((inter_pt, start, end))
                            result = self.intersect2(start, end, indice_x, indice_y)
                            if result == None:
                                print("2.3")
                            return self.intersect2(start, end, indice_x, indice_y)

                            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
                    print("oups 2.3")
                    return self.intersect2(start, end, indice_end_x, indice_end_y)

                else:
                    signe = 1
                    limit = end_x - signe * self.step/20
                    x = start_x
                    while x<=limit:
                        x = x + signe * self.step/20
                        y = calcul_y(start_x, end_x, start_y, end_y, x)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 0:
                            # inter_pt = Node([x,y])
                            # self.collision_set.add((inter_pt, start, end))
                            result = self.intersect2(start, end, indice_x, indice_y)
                            if result == None:
                                print("2.4")
                            return self.intersect2(start, end, indice_x, indice_y)

                            # return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)
                    print("oups 2.4")
                    return self.intersect2(start, end, indice_end_x, indice_end_y)



            print("PROBLEME")
            print("start : ", start)
            print("end : ", end)
            print("\n")
            return 0        
        



        else:
            # cas 1 : ne rentre pasdans un obstalce
            #cas 2 : passe un petit peu dans un obstacle ??

            return self.calc_dist(start, end)






######################################################################################################################
    def intersect_start_not_in_obstacle(self, start, end, indice_x, indice_y):
        l = []
        result1 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+self.step+1])
        if result1 != False:
            l.append(result1)
      
        result2 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1])
        if result2 != False:
            l.append(result2)
      
        result3 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+1])
        if result3 != False:
            l.append(result3)
      
        result4 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+1])
        if result4 != False:
            l.append(result4)
      
        if l==[]:
            return False
        else:
            (inter_pt, _) = min(l, key=cmp_key)
            self.collision_set.add((inter_pt, start, end))
            return self.calc_dist(start, inter_pt) + self.cost_in_obstcles * self.calc_dist(inter_pt, end)         


    def intersect_start_in_obstacle(self, start, end, indice_x, indice_y):
        l = []
        result1 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+self.step+1])
        if result1 != False:
            l.append(result1)
       
        result2 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1])
        if result2 != False:
            l.append(result2)
       
        result3 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+1])
        if result3 != False:
            l.append(result3)
        
        result4 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+1])
        if result4 != False:
            l.append(result4)
       
        if l==[]:
            return False

        else:
            (inter_pt, _) = min(l, key=cmp_key)
            self.collision_set.add((inter_pt, start, end))
            return self.cost_in_obstcles * self.calc_dist(start, inter_pt) + self.calc_dist(inter_pt, end) 

    def fonction_aux_y(self, start, end, pas):
        start_x = start.x
        start_y = start.y
        end_x = end.x
        end_y = end.y
        indice_start_x = self.indice(start_x, self.x_range[0])
        indice_start_y = self.indice(start_y, self.y_range[0])
        indice_end_x = self.indice(end_x, self.x_range[0])
        indice_end_y = self.indice(end_y, self.y_range[0])

        nombre_points = round(abs((end_y - start_y)) / (self.step/pas))
        for y in np.linspace(start_y, end_y, nombre_points):

            # y = y + signe     * self.step/pas
            x = calcul_x(start_x, end_x, start_y, end_y, y)
            indice_y = self.indice(y, self.y_range[0])
            indice_x = self.indice(x, self.x_range[0])
            if self.grid[indice_x][indice_y] == 1:
                # result = self.intersect_start_not_in_obstacle(start, end, indice_x, indice_y)
                result = self.intersect(start, end, indice_x, indice_y, 1, self.cost_in_obstcles)

                if result == None:
                    print("1.1")
                # return self.intersect_start_not_in_obstacle(start, end, indice_x, indice_y)
                return self.intersect(start, end, indice_x, indice_y, 1, self.cost_in_obstcles)

        print("oups 1.1")
        return self.intersect_start_not_in_obstacle(start, end, indice_end_x, indice_end_y)

    def Cost(self, start:Node, end:Node) -> float:
        start_x = start.x
        start_y = start.y
        end_x = end.x
        end_y = end.y

        pas = 5

        indice_start_x = self.indice(start_x, self.x_range[0])
        indice_start_y = self.indice(start_y, self.y_range[0])
        indice_end_x = self.indice(end_x, self.x_range[0])
        indice_end_y = self.indice(end_y, self.y_range[0])

        # print("indice_start_x = ", indice_start_x)
        # print("indice_start_y = ", indice_start_y)
        # print("indice_end_x = ", indice_end_x)
        # print("indice_end_y = ", indice_end_y)

        if self.grid[indice_start_x][indice_start_y] == 1 and self.grid[indice_end_x][indice_end_y] == 1:
            return self.cost_in_obstcles * self.calc_dist(start, end)
        
        elif self.grid[indice_start_x][indice_start_y] == 0 and self.grid[indice_end_x][indice_end_y] == 1:
  
            if abs(end_x - start_x) < abs(end_y - start_y):
                if end_y - start_y<0:   
                    signe = -1
                    limit = end_y - signe * self.step/pas
                    # y = start_y
                    # while y>=limit:
                    # for y in np.arange(start_y, end_y-self.step/pas, -self.step/pas):
                    nombre_points = round(abs((end_y - start_y)) / (self.step/pas))
                    for y in np.linspace(start_y, end_y, nombre_points):

                        # y = y + signe     * self.step/pas
                        x = calcul_x(start_x, end_x, start_y, end_y, y)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 1:
                            # result = self.intersect_start_not_in_obstacle(start, end, indice_x, indice_y)
                            result = self.intersect(start, end, indice_x, indice_y, 1, self.cost_in_obstcles)

                            if result == None:
                                print("1.1")
                            # return self.intersect_start_not_in_obstacle(start, end, indice_x, indice_y)
                            return self.intersect(start, end, indice_x, indice_y, 1, self.cost_in_obstcles)

                    print("oups 1.1")
                    return self.intersect_start_not_in_obstacle(start, end, indice_end_x, indice_end_y)

                else:
                    signe = 1
                    limit = end_y - signe * self.step/pas
                    y = start_y
                    while y<=limit:
                        y = y + signe * self.step/pas
                        x = calcul_x(start_x, end_x, start_y, end_y, y)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 1:
                            result = self.intersect_start_not_in_obstacle(start, end, indice_x, indice_y)
                            if result == None:
                                print("1.2")
                            return self.intersect_start_not_in_obstacle(start, end, indice_x, indice_y)
                    print("oups 1.2")
                    return self.intersect_start_not_in_obstacle(start, end, indice_end_x, indice_end_y)
            
            else:
                if end_x - start_x<0:
                    signe = -1
                    limit = end_x - signe * self.step/pas
                    x = start_x
                    while x>=limit:
                        x = x + signe * self.step/pas
                        y = calcul_y(start_x, end_x, start_y, end_y, x)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 1:
                            result = self.intersect_start_not_in_obstacle(start, end, indice_x, indice_y)
                            if result == None:
                                print("1.3")
                            return self.intersect_start_not_in_obstacle(start, end, indice_x, indice_y)
                    print("oups 1.3")
                    return self.intersect_start_not_in_obstacle(start, end, indice_end_x, indice_end_y)

                else:
                    signe = 1
                    limit = end_x - signe * self.step/pas
                    # x = start_x
                    # while x<=limit:
                    for x in np.arange(start_x, end_x+self.step/pas, self.step/pas):

                        # x = x + signe * self.step/pas
                        y = calcul_y(start_x, end_x, start_y, end_y, x)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 1:
                            result = self.intersect(start, end, indice_x, indice_y, 1, self.cost_in_obstcles)
                            if result == None:
                                print("1.4")
                            return self.intersect(start, end, indice_x, indice_y, 1, self.cost_in_obstcles)
                    print("oups 1.4")   
                    return self.intersect_start_not_in_obstacle(start, end, indice_end_x, indice_end_y)


            print("PROBLEME")
            print("start : ", start)
            print("end : ", end)
            print("\n")
            return 0

        elif self.grid[indice_start_x][indice_start_y] == 1 and self.grid[indice_end_x][indice_end_y] == 0:        
            if abs(end_x - start_x) < abs(end_y - start_y):
                if end_y - start_y<0:
                    signe = -1
                    limit = end_y - signe * self.step/pas
                    y = start_y
                    while y>=limit:
                        y = y + signe * self.step/pas
                        x = calcul_x(start_x, end_x, start_y, end_y, y)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 0:
                            result = self.intersect_start_in_obstacle(start, end, indice_x, indice_y)
                            if result == None:
                                print("2.1")
                            return self.intersect_start_in_obstacle(start, end, indice_x, indice_y)
                    print("oups 2.1")
                    return self.intersect_start_in_obstacle(start, end, indice_end_x, indice_end_y)

                else:
                    signe = 1
                    limit = end_y - signe * self.step/pas
                    y = start_y
                    while y<=limit:
                        y = y + signe * self.step/pas
                        x = calcul_x(start_x, end_x, start_y, end_y, y)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 0:
                            result = self.intersect_start_in_obstacle(start, end, indice_x, indice_y)
                            if result == None:
                                print("2.2")
                            return self.intersect_start_in_obstacle(start, end, indice_x, indice_y)
                    print("oups 2.2")
                    return self.intersect_start_in_obstacle(start, end, indice_end_x, indice_end_y)
            else:
                if end_x - start_x<0:
                    signe = -1
                    limit = end_x - signe * self.step/pas
                    x = start_x
                    while x>=limit:
                        x = x + signe * self.step/pas
                        y = calcul_y(start_x, end_x, start_y, end_y, x)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])

                        if self.grid[indice_x][indice_y] == 0:

                            result = self.intersect_start_in_obstacle(start, end, indice_x, indice_y)
                            if result == None:
                                print("2.3")
                            return self.intersect_start_in_obstacle(start, end, indice_x, indice_y)

                    print("oups 2.3")
                    return self.intersect_start_in_obstacle(start, end, indice_end_x, indice_end_y)

                else:
                    signe = 1
                    limit = end_x - signe * self.step/pas
                    x = start_x
                    while x<=limit:
                        x = x + signe * self.step/pas
                        y = calcul_y(start_x, end_x, start_y, end_y, x)
                        indice_y = self.indice(y, self.y_range[0])
                        indice_x = self.indice(x, self.x_range[0])
                        if self.grid[indice_x][indice_y] == 0:

                            result = self.intersect_start_in_obstacle(start, end, indice_x, indice_y)
                            if result == None:
                                print("2.4")
                            return self.intersect_start_in_obstacle(start, end, indice_x, indice_y)

                    print("oups 2.4")
                    return self.intersect_start_in_obstacle(start, end, indice_end_x, indice_end_y)



            print("PROBLEME")
            print("start : ", start)
            print("end : ", end)
            print("\n")
            return 0        
        



        else:
            # cas 1 : ne rentre pasdans un obstalce
            #cas 2 : passe un petit peu dans un obstacle ??

            return self.calc_dist(start, end)
        
        






##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################


# version 19/11/22

"""
Fast Marching Trees (FMT*)
@author: huiming zhou
"""

import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from scipy.stats import norm
from scipy.stats import uniform
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../../Sampling_based_Planning/")

# from Sampling_based_Planning.rrt_2D import env, plotting, utils

import env, plotting, utils

class Node:

    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = np.inf

    def __repr__(self):
        return "("+str(self.x)+", "+str(self.y)+")"

class FMT:
    def __init__(self, x_start, x_goal, search_radius, cost_in_obstcles, sample_numbers, step):
        self.x_init = Node(x_start)
        self.x_goal = Node(x_goal)
        self.search_radius = search_radius
        self.cost_in_obstcles = cost_in_obstcles

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()




        # self.fig, self.ax1 = plt.subplots()


        self.fig = plt.figure(num = 'MAP', figsize=(15, 15)) #frameon=True)
        self.map = plt.axes(projection=ccrs.PlateCarree())
        self.map.set_extent([-15, 24, 30, 65], ccrs.PlateCarree())  
        self.map.coastlines(resolution='50m')
        self.map.add_feature(cfeature.OCEAN.with_scale('50m'))
        self.map.add_feature(cfeature.LAKES.with_scale('50m'))
        self.map.add_feature(cfeature.LAND.with_scale('50m'))
        self.map.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='dotted', alpha=0.7)

        grid_lines = self.map.gridlines(draw_labels=True)
        grid_lines.xformatter = LONGITUDE_FORMATTER
        grid_lines.yformatter = LATITUDE_FORMATTER


        # # # self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(ncols = 2, nrows = 2)
        # self.fig = plt.figure(constrained_layout = True)
        # widths = [4, 1]
        # heights = [4, 1]
        # spec = self.fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
        # self.ax1 = self.fig.add_subplot(spec[0,0])
        
        # self.ax2 = self.fig.add_subplot(spec[0,1], sharey = self.ax1)
        # self.ax3 = self.fig.add_subplot(spec[1,0], sharex = self.ax1)
        # self.ax1.label_outer()
        
        # # self.ax2.label_outer()
        # # self.ax2.yaxis.set_ticks_position('none')
        # self.ax2.get_yaxis().set_visible(False)
        
        # self.ax3.label_outer()



        self.delta = self.utils.delta
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.step = step

        self.grid = self.quartering(step)
        save_quartering(self.grid, "quartering_map")
        # self.grid = load_quartering("quartering_testtt")

        self.V = set()
        self.V_unvisited = set()
        self.V_open = set()
        self.V_closed = set()
        self.sample_numbers = sample_numbers
        self.rn = self.search_radius * math.sqrt((math.log(self.sample_numbers) / self.sample_numbers))
        self.collision_set = set()

    def Init(self):
        samples = self.SampleFree()

        save_samples(samples, "samples_map.txt")
        # samples = self.load_samples("samples_testtt.txt")

        self.x_init.cost = 0.0
        self.V.add(self.x_init)
        self.V.update(samples)
        self.V_unvisited.update(samples)
        self.V_unvisited.add(self.x_goal)
        self.V_open.add(self.x_init)

    def Planning(self):

        start = time.time()

        self.Init()
        z = self.x_init
        n = self.sample_numbers
        rn = self.search_radius * math.sqrt((math.log(n) / n))
        print("rn = ", rn)
        print("\n")
        Visited = []

        while z is not self.x_goal:
            V_open_new = set()
            X_near = self.Near(self.V_unvisited, z, rn)
            Visited.append(z)

            for x in X_near:
                ###########################################################################

                # self.V.discard(x)
                # Nx = self.Near(self.V,x,rn)
                # self.V.add(x)
                
                # Y_near = Nx.intersection(self.V_open)

                #_________________________________________________________________________#

                Y_near = self.Near(self.V_open, x, rn)

                ###########################################################################
                # print("1-ième fois \n")

                cost_list = {y: y.cost + self.Cost(y, x) for y in Y_near}
                y_min = min(cost_list, key=cost_list.get)

                ###########################################################################
                
                # if not self.utils.is_collision(y_min, x):
                #     x.parent = y_min
                #     V_open_new.add(x)
                #     self.V_unvisited.remove(x)
                #     x.cost = y_min.cost + self.Cost(y_min, x)

                #_________________________________________________________________________#

                x.parent = y_min
                V_open_new.add(x)
                self.V_unvisited.remove(x)
                # print("2-ième fois \n")
                x.cost = y_min.cost + self.Cost(y_min, x)
                
                ###########################################################################


            self.V_open.update(V_open_new)
            self.V_open.remove(z)
            self.V_closed.add(z)

            if not self.V_open:
                print("open set empty!")
                break

            cost_open = {y: y.cost for y in self.V_open}
            z = min(cost_open, key=cost_open.get)

        # node_end = self.ChooseGoalPoint()
        path_x, path_y, path = self.ExtractPath()

        end = time.time()
        print("Temps d'execution : ",end-start)
        print("\n")
        print("Path : ", path)
        print("\n")

        total_cost_verif = self.calc_dist_total(path)
        print("Total cost (en calculant la distance sans aucunes pénalités): ", total_cost_verif)

        print("Total cost : ", self.x_goal.cost)
        self.animation(path_x, path_y, Visited[1: len(Visited)], path)
        # self.plot_nodes()



    def ChooseGoalPoint(self):
        Near = self.Near(self.V, self.x_goal, 2.0)
        cost = {y: y.cost + self.Cost(y, self.x_goal) for y in Near}

        return min(cost, key=cost.get)


    def ExtractPath(self):
        path_x, path_y = [], []
        path = []
        node = self.x_goal

        while node.parent:
            path_x.append(node.x)
            path_y.append(node.y)
            path.append(node)
            node = node.parent

        path_x.append(self.x_init.x)
        path_y.append(self.x_init.y)
        path.append(self.x_init)

        return path_x, path_y, path



    def indice(self, x, x0):
        a = x - x0
        b = a // self.step
        return int(b)



    def intersect(self, start, end, indice_x, indice_y, coef_avant_inter, coef_apres_inter, true_start, true_end):
        l = []
        result1 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+self.step+1])
        if result1 != False:
            l.append(result1)
      
        result2 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1])
        if result2 != False:
            l.append(result2)
      
        result3 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+self.step+1], [indice_x*self.step+self.step+1, indice_y*self.step+1])
        if result3 != False:
            l.append(result3)
      
        result4 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+1, indice_y*self.step+1], [indice_x*self.step+1, indice_y*self.step+1])
        if result4 != False:
            l.append(result4)
      
        if l==[]:
            return False
        else:
            (inter_pt, _) = min(l, key=cmp_key)
            self.collision_set.add((inter_pt, true_start, true_end))
            return coef_avant_inter * self.calc_dist(start, inter_pt) + coef_apres_inter * self.calc_dist(inter_pt, end)         



    def fonction_aux_y(self, start, end, pas, cond, coef_avant_inter, coef_apres_inter):
        start_x = start.x
        start_y = start.y
        end_x = end.x
        end_y = end.y

        indice_end_x = self.indice(end_x, self.x_range[0])
        indice_end_y = self.indice(end_y, self.y_range[0])

        nombre_points = round(abs((end_y - start_y)) / (self.step/pas))
        for y in np.linspace(start_y, end_y, max(2, nombre_points)):
            x = calcul_x(start_x, end_x, start_y, end_y, y)
            indice_y = self.indice(y, self.y_range[0])
            indice_x = self.indice(x, self.x_range[0])
            if self.grid[indice_x][indice_y] == cond:

                return self.intersect(start, end, indice_x, indice_y, coef_avant_inter, coef_apres_inter, start, end)

        # print("oups 1")
        # print("start : ",start)
        # print("end : ", end)
        # return self.intersect(start, end, indice_end_x, indice_end_y, coef_avant_inter, coef_apres_inter)


    def fonction_aux_x(self, start, end, pas, cond, coef_avant_inter, coef_apres_inter):
        start_x = start.x
        start_y = start.y
        end_x = end.x
        end_y = end.y

        indice_end_x = self.indice(end_x, self.x_range[0])
        indice_end_y = self.indice(end_y, self.y_range[0])

        nombre_points = round(abs((end_x - start_x)) / (self.step/pas))
        for x in np.linspace(start_x, end_x, max(2, nombre_points)):
            y = calcul_y(start_x, end_x, start_y, end_y, x)

            indice_y = self.indice(y, self.y_range[0])
            indice_x = self.indice(x, self.x_range[0])
            if self.grid[indice_x][indice_y] == cond:

                return self.intersect(start, end, indice_x, indice_y, coef_avant_inter, coef_apres_inter, start, end)

        # print("oups 2")
        # print("start : ",start)
        # print("end : ", end)
        # return self.intersect(start, end, indice_end_x, indice_end_y, coef_avant_inter, coef_apres_inter)

    def fonction_aux2_y(self, start, end, pas):
        start_x = start.x
        start_y = start.y
        end_x = end.x
        end_y = end.y
        inter_1 = []
        indice_inter_1 = []
        indice_inter_2 = []
        indice_end_x = self.indice(end_x, self.x_range[0])
        indice_end_y = self.indice(end_y, self.y_range[0])

        nombre_points = round(abs((end_y - start_y)) / (self.step/pas))
        for y in np.linspace(start_y, end_y, max(2, nombre_points)):
            x = calcul_x(start_x, end_x, start_y, end_y, y)

            indice_y = self.indice(y, self.y_range[0])
            indice_x = self.indice(x, self.x_range[0])
            if self.grid[indice_x][indice_y] == 1:
                inter_1 = [x, y]
                indice_inter_1 = [indice_x, indice_y]
                break
                # return self.intersect(start, end, indice_x, indice_y, coef_avant_inter, coef_apres_inter)
        if inter_1 == []:
            return None

        for y in np.linspace(inter_1[1], end_y, max(2, nombre_points)):
            x = calcul_x(start_x, end_x, start_y, end_y, y)

            indice_y = self.indice(y, self.y_range[0])
            indice_x = self.indice(x, self.x_range[0])
            if self.grid[indice_x][indice_y] == 0:
                inter_2 = [x, y]
                indice_inter_2 = [indice_x, indice_y]
                break

        return self.intersect(start, Node(inter_1), indice_inter_1[0], indice_inter_1[1], 1, self.cost_in_obstcles, start, end) \
            + self.intersect(Node(inter_1), end, indice_inter_2[0], indice_inter_2[1], self.cost_in_obstcles, 1, start, end)

    def fonction_aux2_x(self, start, end, pas):
        start_x = start.x
        start_y = start.y
        end_x = end.x
        end_y = end.y
        inter_1 = []
        indice_inter_1 = []
        indice_end_x = self.indice(end_x, self.x_range[0])
        indice_end_y = self.indice(end_y, self.y_range[0])

        nombre_points = round(abs((end_x - start_x)) / (self.step/pas))
        for x in np.linspace(start_x, end_x, max(2, nombre_points)):
            y = calcul_y(start_x, end_x, start_y, end_y, x)

            indice_y = self.indice(y, self.y_range[0])
            indice_x = self.indice(x, self.x_range[0])
            if self.grid[indice_x][indice_y] == 1:
                inter_1 = [x, y]
                indice_inter_1 = [indice_x, indice_y]
                break
                # return self.intersect(start, end, indice_x, indice_y, coef_avant_inter, coef_apres_inter)
        if inter_1 == []:
            return None

        for x in np.linspace(inter_1[0], end_x, max(2, nombre_points)):
            y = calcul_y(start_x, end_x, start_y, end_y, x)

            indice_y = self.indice(y, self.y_range[0])
            indice_x = self.indice(x, self.x_range[0])
            if self.grid[indice_x][indice_y] == 0:
                inter_2 = [x, y]
                indice_inter_2 = [indice_x, indice_y]
                break

        return self.intersect(start, Node(inter_1), indice_inter_1[0], indice_inter_1[1], 1, self.cost_in_obstcles, start, end) \
            + self.intersect(Node(inter_1), end, indice_inter_2[0], indice_inter_2[1], self.cost_in_obstcles, 1, start, end)


    def Cost(self, start:Node, end:Node) -> float:
        start_x = start.x
        start_y = start.y
        end_x = end.x
        end_y = end.y

        pas = 5

        indice_start_x = self.indice(start_x, self.x_range[0])
        indice_start_y = self.indice(start_y, self.y_range[0])
        indice_end_x = self.indice(end_x, self.x_range[0])
        indice_end_y = self.indice(end_y, self.y_range[0])

        # print("indice_start_x = ", indice_start_x)
        # print("indice_start_y = ", indice_start_y)
        # print("indice_end_x = ", indice_end_x)
        # print("indice_end_y = ", indice_end_y)

        if self.grid[indice_start_x][indice_start_y] == 1 and self.grid[indice_end_x][indice_end_y] == 1:
            return self.cost_in_obstcles * self.calc_dist(start, end)
        

        elif self.grid[indice_start_x][indice_start_y] == 0 and self.grid[indice_end_x][indice_end_y] == 1:
            
            if abs(end_x - start_x) < abs(end_y - start_y):
                return self.fonction_aux_y(start, end, pas, 1, 1, self.cost_in_obstcles)
            else:
                return self.fonction_aux_x(start, end, pas, 1, 1, self.cost_in_obstcles)


        elif self.grid[indice_start_x][indice_start_y] == 1 and self.grid[indice_end_x][indice_end_y] == 0:        
            
            if abs(end_x - start_x) < abs(end_y - start_y):
                return self.fonction_aux_y(start, end, pas, 0, self.cost_in_obstcles, 1)
            else:
                return self.fonction_aux_x(start, end, pas, 0, self.cost_in_obstcles, 1)


        else:
            # cas 1 : ne rentre pas dans un obstalce
            # cas 2 : passe un petit peu dans un obstacle ??

            
            if abs(end_x - start_x) < abs(end_y - start_y):
                result = self.fonction_aux2_y(start, end, pas)
            else:
                result = self.fonction_aux2_x(start, end, pas)
            if result == None:
                return self.calc_dist(start, end)
            else:
                return result

            # return self.calc_dist(start, end)

    @staticmethod
    def calc_dist(x_start:Node, x_end:Node):
        # return math.hypot(x_start.x - x_end.x, x_start.y - x_end.y)

        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(math.radians, [x_start.x, x_start.y, x_end.x, x_end.y])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + (math.cos(lat1) * math.cos(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6391        # radius of earth in kilometers
        return c * r 


    def calc_dist_total(self, path):
        total_cost = 0
        length = len(path)
        for i in range(length-1):
            total_cost += self.calc_dist(path[i], path[i+1])
        return total_cost


    @staticmethod
    def Near(nodelist, z, rn):
        return {nd for nd in nodelist
                if 0 < (nd.x - z.x) ** 2 + (nd.y - z.y) ** 2 <= rn ** 2}


    def proportion_obstacles(self):
        n = len(self.grid)
        m = len(self.grid[0])
        nombre_points_dans_obstacles = 0
        for i in range(n):
            for j in range(m):
                if self.grid[i][j] == 1:
                    nombre_points_dans_obstacles += 1
        proportion_obstacles = nombre_points_dans_obstacles / (n*m)
        # print("nombre points dans obstacles : ", nombre_points_dans_obstacles)
        # print("Proportion obstacles : ", proportion_obstacles) 
        return proportion_obstacles


    def SampleFree(self):

        start = time.time()

        n = self.sample_numbers
        delta = self.utils.delta
        Sample = set()

        porportion_obstacles = self.proportion_obstacles()
        # porportion_obstacles = 1

        nb_points_dans_obstacles = round(porportion_obstacles * self.sample_numbers)
        nb_points_dans_espace_libre = self.sample_numbers - nb_points_dans_obstacles

        # ind = 0
        # while ind < n:
        #     node = Node((random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
        #                  random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        #     # node = Node((random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
        #     #              random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        #     # if self.utils.is_inside_obs(node):            #si on ne veut pas de points dans les obstacles
        #     #     continue
        #     # else:
        #     #     Sample.add(node)
        #     #     ind += 1


        #     Sample.add(node)                                # si on veut des points dans les obstacles
        #     ind += 1

        ################################################################################################
        cpt_dans_obstacles = 0
        cpt_dans_espace_libre = 0 



        # while cpt_dans_obstacles < nb_points_dans_obstacles or cpt_dans_espace_libre < nb_points_dans_espace_libre:
            
        #     node = Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
        #                  np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        #     indice_node_x = self.indice(node.x, self.x_range[0])
        #     indice_node_y = self.indice(node.y, self.y_range[0])

        #     if self.grid[indice_node_x][indice_node_y] == 1 and cpt_dans_obstacles < nb_points_dans_obstacles:
        #         Sample.add(node)
        #         cpt_dans_obstacles += 1
        #     elif self.grid[indice_node_x][indice_node_y] == 0 and cpt_dans_espace_libre < nb_points_dans_espace_libre:
        #         Sample.add(node)
        #         cpt_dans_espace_libre += 1


################################################################################


        # distributions_x = []
        # distributions_y = []

        # for (x, y, r) in self.obs_circle:
        #     distributions_x.append({"type": np.random.normal, "kwargs": {"loc": x, "scale": r/4}})
        #     distributions_y.append({"type": np.random.normal, "kwargs": {"loc": y, "scale": r/4}})


        # for (x, y, w, h) in self.obs_rectangle:
        #     distributions_x.append({"type": np.random.normal, "kwargs": {"loc": x+w/2, "scale": w/4}})
        #     distributions_y.append({"type": np.random.normal, "kwargs": {"loc": y+h/2, "scale": h/4}})


        # num_distr = len(distributions_x)
        # coefficients = np.ones(num_distr)
        # coefficients /= coefficients.sum()      # in case these did not add up to 1



################################################################################


        # sample_size = 4000

        # num_distr = len(distributions_x)
        # data = np.zeros((sample_size, num_distr))

        # print(data)
        # for idx, distr in enumerate(distributions_x):
        #     data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])


        # random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)


        # sample = data[np.arange(sample_size), random_idx]

        # plt.hist(sample, bins=100, density=True)
        # plt.show()


################################################################################
################################################################################



        distributions = []
        distributions_x = []
        distributions_y = []

        distributions_x_uniform = []
        distributions_y_uniform = []

        coefficients = []
        somme_coefficients = 0
        area = (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0])
        print("area : ", area)
        ecart = 1
        for (x, y, r) in self.obs_circle:
            # mu_x = x 
            # sigma_x = r/4
            # mu_y = y
            # sigma_y = sigma_x
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}}, 
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            #############################################
            xc = x
            yc = y
            radius = r
            distributions.append(("circle", (xc, yc, radius, ecart)))
            p = math.pi * radius**2
            coefficients.append(p)
            somme_coefficients += p


        for (x, y, w, h) in self.obs_rectangle:
            # mu_x = x + w/2
            # sigma_x = w/4
            # mu_y = y + h/2
            # sigma_y = h/4
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            #########################
            ##########################
            area_rect = w * h
            # p = area_rect/4

            coef_sigma = 3

            # mu_x = x
            # sigma_x = ecart
            # mu_y = y+h/2
            # sigma_y = h/coef_sigma
            
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            low_x = x - ecart
            high_x = x + ecart
            low_y = y - ecart
            high_y = y + h + ecart

            distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                  {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
            distributions_x_uniform.append((low_x, high_x))
            distributions_y_uniform.append((low_y, high_y))
            # p = 2*ecart * h
            p = 2*ecart*h * (area_rect/(4*ecart*(h+w)))  *2

            coefficients.append(p)
            somme_coefficients += p
            ##############
            # mu_x = x + w/2
            # sigma_x = w/coef_sigma
            # mu_y = y
            # sigma_y = ecart
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            low_x = x - ecart
            high_x = x + w + ecart
            low_y = y - ecart
            high_y = y + ecart
            distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                  {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
            distributions_x_uniform.append((low_x, high_x))
            distributions_y_uniform.append((low_y, high_y))

            # p = 2*ecart * w
            p = 2 * ecart * w * (area_rect/(4*ecart*(h+w)))  *2

            coefficients.append(p)
            somme_coefficients += p
            ##############
            # mu_x = x + w
            # sigma_x = ecart
            # mu_y = y + h/2
            # sigma_y = h/coef_sigma

            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            low_x = x + w - ecart
            high_x = x + w + ecart
            low_y = y - ecart
            high_y = y + h + ecart
            distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                  {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
            distributions_x_uniform.append((low_x, high_x))
            distributions_y_uniform.append((low_y, high_y))

            # p = 2*ecart * h
            p = 2*ecart*h * (area_rect/(4*ecart*(h+w)))  *2
            coefficients.append(p)
            somme_coefficients += p
            ##############
            # mu_x = x + w/2
            # sigma_x = w/coef_sigma
            # mu_y = y + h
            # sigma_y = ecart
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            low_x = x - ecart
            high_x = x + w + ecart
            low_y = y + h - ecart
            high_y = y + h + ecart
            distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                  {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
            distributions_x_uniform.append((low_x, high_x))
            distributions_y_uniform.append((low_y, high_y))

            # p = 2*ecart * w
            p = 2*ecart*w * (area_rect/(4*ecart*(h+w)))  *2

            coefficients.append(p)
            somme_coefficients += p

        distributions.append(({"type":np.random.uniform, "kwargs":{"low": self.x_range[0], "high": self.x_range[1]}},
                              {"type":np.random.uniform, "kwargs":{"low": self.y_range[0], "high": self.y_range[1]}}))
        p = (area - somme_coefficients)/4
        coefficients.append(p)

        print("coeeficent 1 :", coefficients)
        # somme_coefficients += p
        coefficients = np.array(coefficients)
        coefficients = coefficients / np.sum(coefficients)
        print("coefficents : ", coefficients)
        print("somme coefficents : ", np.sum(coefficients))

        # print(distributions)
        num_distr = len(distributions)

        colors = ["gray", "red", "gold", "olivedrab", "chartreuse", "aqua", "royalblue", "blueviolet", "pink"]
        n = len(colors)
        x = np.linspace(self.x_range[0], self.x_range[1], 1000)
        density_x = 0

        ## for i, (mu_x, sigma_x) in enumerate(distributions_x):
        ##     density_x += norm.pdf(x, mu_x, sigma_x)
            
        ##     # if i==3:
        ##     self.ax3.plot(x, norm.pdf(x, mu_x, sigma_x)/num_distr, color=colors[i%n])
        
        # for i, (low_x, high_x) in enumerate(distributions_x_uniform):
        #     density_x += uniform.pdf(x, low_x, high_x)
        #     self.ax3.plot(x, uniform.pdf(x, low_x, high_x)/num_distr, color=colors[i%n])

        # density_x += uniform.pdf(x, self.x_range[0], self.x_range[1])
        # self.ax3.plot(x, uniform.pdf(x, self.x_range[0], self.x_range[1])/num_distr, linewidth = 1, color="orange")

        # density_x /= num_distr
        # self.ax3.plot(x, density_x, color="black", alpha=0.8)



        y = np.linspace(self.y_range[0], self.y_range[1], 1000)
        density_y = 0
        ## for i, (mu_y, sigma_y) in enumerate(distributions_y):
        ##     density_y += norm.pdf(y, mu_y, sigma_y)
            
        ##     # if i==3:
        ##     self.ax2.plot(norm.pdf(y, mu_y, sigma_y)/num_distr, y, color=colors[i%n])

        # for i, (low_y, high_y) in enumerate(distributions_y_uniform):
        #     density_y += uniform.pdf(y, low_y, high_y)
        #     self.ax2.plot(uniform.pdf(y, low_y, high_y)/num_distr, y, color=colors[i%n])

        # density_y += uniform.pdf(y, self.y_range[0], self.y_range[1])
        # self.ax2.plot(uniform.pdf(y, self.y_range[0], self.y_range[1])/num_distr, y, linewidth = 1, color="orange")

        # density_y /= num_distr
        # self.ax2.plot(density_y, y, color="black", alpha=0.8)

        ###############################""
        # while cpt_dans_espace_libre < nb_points_dans_espace_libre:
            
        #     node = Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
        #                  np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        #     indice_node_x = self.indice(node.x, self.x_range[0])
        #     indice_node_y = self.indice(node.y, self.y_range[0])

        #     if self.grid[indice_node_x][indice_node_y] == 0:
        #         Sample.add(node)
        #         cpt_dans_espace_libre += 1


        while cpt_dans_obstacles < nb_points_dans_obstacles or cpt_dans_espace_libre < nb_points_dans_espace_libre:
            # node = Node((get_sample_pas_bien(distributions_x, num_distr), 
            #              get_sample_pas_bien(distributions_y, num_distr)))

            node = Node(get_sample(distributions, num_distr, coefficients))

            if node.x>self.x_range[0] and node.x < self.x_range[1] and node.y>self.y_range[0] and node.y<self.y_range[1]:
                indice_node_x = self.indice(node.x, self.x_range[0])
                indice_node_y = self.indice(node.y, self.y_range[0])

                if self.grid[indice_node_x][indice_node_y] == 1 and cpt_dans_obstacles < nb_points_dans_obstacles:
                    Sample.add(node)
                    cpt_dans_obstacles += 1

                if self.grid[indice_node_x][indice_node_y] == 0 and cpt_dans_espace_libre < nb_points_dans_espace_libre:
                    Sample.add(node)
                    cpt_dans_espace_libre += 1
                

        print("Proportion obstacles : ", porportion_obstacles)
        print("Nb points dans obstacle : ", cpt_dans_obstacles)
        print("Nb points dans esapce libre : ", cpt_dans_espace_libre)
        
        end = time.time()
        print("Temps d'execution sample: ",end-start)
        return Sample


    def plot_nodes(self):
        self.plot_grid(f"Fast Marching Trees (FMT*) avec n = {self.sample_numbers}, un coût dans l'obstacle de {self.cost_in_obstcles} et step = {self.step}")


        for node in self.V:
            # self.ax1.plot(node.x, node.y, marker='o', color='magenta', markersize=2)
            self.map.plot(node.x, node.y, marker='o', color='magenta', markersize=2, transform=ccrs.PlateCarree())

            # self.ax1.add_patch(
            #     patches.Circle(
            #         (node.x, node.y), self.rn,
            #         edgecolor='black',
            #         facecolor='gray',
            #         fill=False
            #     )
            # )
        plt.show()
    

    def animation(self, path_x, path_y, visited, path):
        self.plot_grid(f"Fast Marching Trees (FMT*) avec n = {self.sample_numbers}, un coût dans l'obstacle de {self.cost_in_obstcles} et step = {self.step}")

        for node in self.V:
            # self.ax1.plot(node.x, node.y, marker='.', color='magenta', markersize=5)  #lightgrey
            self.map.plot(node.x, node.y, marker='.', color='magenta', markersize=5, transform=ccrs.PlateCarree())  #lightgrey

            

        count = 0
        for node in visited:
            count += 1
            # self.ax1.plot([node.x, node.parent.x], [node.y, node.parent.y], '-g')
            self.map.plot([node.x, node.parent.x], [node.y, node.parent.y], '-g', transform=ccrs.Geodetic())

            plt.gcf().canvas.mpl_connect(
                'key_press_event',
                lambda event: [exit(0) if event.key == 'escape' else None])         # key_release_event
            if count % (self.sample_numbers/10) == 0:
                plt.pause(0.01)

        # self.ax1.plot(path_x, path_y, linewidth=2, color='red')
        self.map.plot(path_x, path_y, linewidth=2, color='red', transform=ccrs.Geodetic())

        # plt.pause(0.01)

        for (node, start, end) in self.collision_set:
            for (i, node_i) in enumerate(path):
                if node_i == start and path[i-1] == end:
                    # self.ax1.plot(node.x, node.y, marker="x", color="blue", markersize=6)
                    self.map.plot(node.x, node.y, marker="x", color="blue", markersize=6, transform=ccrs.PlateCarree())

        # plt.pause(0.01)
        plt.show()


    def plot_grid(self, name):

        for (ox, oy, w, h) in self.obs_boundary:
            # self.ax1.add_patch(
            self.map.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            # self.ax1.add_patch(
            self.map.add_patch(

                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            # self.ax1.add_patch(
            self.map.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        # self.ax1.plot(self.x_init.x, self.x_init.y, "bs", linewidth=3)
        # self.ax1.plot(self.x_goal.x, self.x_goal.y, "rs", linewidth=3)

        self.map.plot(self.x_init.x, self.x_init.y, "bs", linewidth=3)
        self.map.plot(self.x_goal.x, self.x_goal.y, "rs", linewidth=3)

        grid_x_ticks = np.arange(self.x_range[0], self.x_range[1], self.step)
        grid_y_ticks = np.arange(self.y_range[0], self.y_range[1], self.step)

        # self.ax1.set_xticks(grid_x_ticks)
        # self.ax1.set_yticks(grid_y_ticks)
        # self.ax1.grid(alpha=0.5, linestyle="--")

        self.map.set_xticks(grid_x_ticks)
        self.map.set_yticks(grid_y_ticks)
        self.map.grid(alpha=0.5, linestyle="--")
        
        # plt.title(name)
        self.fig.suptitle(name)
        # plt.axis("equal")


    def quartering(self, step):
        x_low_limit = self.x_range[0] 
        x_high_limit = self.x_range[1]
        y_low_limit = self.y_range[0]
        y_high_limit = self.y_range[1]


        n = (x_high_limit - x_low_limit) / step
        m = (y_high_limit - y_low_limit) / step
        # n = round(n)
        # m = round(m)
        if n-round(n) == 0.5:
            n = 1+ round(n)
        else:
            n = round(n)
        
        if m-round(m) == 0.5:
            m = 1 + round(m)
        else:
            m = round(m)

        print("n : ", n)
        print("m : ", m)
        tab = [[0]*m for _ in range(n)]

        indice_i = 0
        indice_j = 0
        for i in np.arange(x_low_limit, x_high_limit, step):
            for j in np.arange(y_low_limit, y_high_limit, step):
                # print("j : ", j)    
                point = Node([(i+step/2), (j+step/2)])
                if self.utils.is_inside_obs(point):
                    # print("indice_i : ", indice_i)
                    # print("indice_j : ", indice_j )
                    tab[indice_i][indice_j] = 1
                indice_j += 1
                # print("indice_j = ", indice_j)
            indice_i += 1
            indice_j = 0
            # print("i = ", i)
            # print("indice_i = ", indice_i)
        # print(tab)
        return tab


    def load_samples(self, filename):
        samples = set()
        nb_points = 0
        nb_points_dans_obstacles = 0
        nb_points_dans_espace_libre = 0
        with open(filename, 'r') as f:
            for line in f:
                line.rstrip()
                line = line.split()
                node = Node([float(line[0]), float(line[1])])
                samples.add(node)
                
                indice_node_x = self.indice(node.x, self.x_range[0])
                indice_node_y = self.indice(node.y, self.y_range[0])
                if self.grid[indice_node_x][indice_node_y] == 1:
                    nb_points_dans_obstacles += 1
                else:
                    nb_points_dans_espace_libre += 1
                nb_points += 1
        print("\n")
        print("nb_points : ", nb_points)
        print("nb_points dans obstacles : ", nb_points_dans_obstacles)
        print("nombre_points dans espace libre : ", nb_points_dans_espace_libre)
        return samples

def calcul_x(xa, xb, ya, yb, y):
    m = (yb - ya)/(xb- xa)
    p = ya - m*xa
    x = (y - p)/m
    return x

def calcul_y(xa, xb, ya, yb, x):
    m = (yb - ya)/(xb - xa)
    p = ya - m*xa
    y = m*x + p
    return y


def save_quartering(tab, filename):
    n = len(tab)
    m = len(tab[0])
    with open(filename, 'w') as f:
        # for i in range(n):
        #     for j in range(m):
        for j in range(m-1,-1,-1):
            for i in range(n):
                f.write(str(tab[i][j]) + "  ")
            f.write("\n")

    # i=x_low_limit
    # j=y_low_limit
    # while i<= x_high_limit:
    #     while j<= y_high_limit:
    #         Node((i+pas/2), (j+pas/2))
    #         i += 8
    # return 

def load_quartering(filename):
    m = 0
    with open(filename, 'r') as f:
        for line in f:
            line.rstrip()
            line = line.split()
            # print(line)
            n = len(line)
            m += 1
    print("n : ", n)
    print("m : ", m)

    tab = [[0]*m for _ in range(n)]
    indice_j = m-1

    with open(filename, 'r') as f:
        for line in f:
            line.rstrip()
            line = line.split()
            for i in range(n):
                tab[i][indice_j] = int(line[i])
            indice_j -= 1
        # print("indice_j : ", indice_j)
    return tab

def save_samples(samples, filename):
    with open(filename, 'w') as f:
        for node in samples:
            f.write(str(node.x) + " " + str(node.y) + "\n")


def cmp_key(e):
    return e[1]



def get_sample_pas_bien(distributions, lenght_distributions):

    num_distr = lenght_distributions
    data = np.zeros(num_distr)
    for idx, distr in enumerate(distributions):
        data[idx] = distr["type"](**distr["kwargs"])

    random_idx = np.random.choice(np.arange(num_distr))
    sample = data[random_idx]
    return sample



def get_sample(distributions, lenght_distributions, coefficients):

    num_distr = lenght_distributions
    # data = np.zeros(num_distr)
    data = [(0,0) for _ in range(num_distr)]
    for idx, (distr_x, distr_y) in enumerate(distributions):
        if distr_x == "circle":
            xc, yc, radius, ecart = distr_y
            # r = radius * random.random()
            # r = radius * math.sqrt(random.random())
            r = random.uniform(radius-ecart, radius+ecart)


            theta = 2* math.pi * random.random()
            data[idx] = (xc + r*math.cos(theta), yc + r * math.sin(theta))

        else:
            data[idx] = (distr_x["type"](**distr_x["kwargs"]), distr_y["type"](**distr_y["kwargs"]))

    random_idx = np.random.choice(np.arange(num_distr), p=coefficients)
    sample = data[random_idx]
    return sample


def main():
    x_start = (-9, 36)  # Starting node
    x_goal = (18, 59)  # Goal node

    
    fmt = FMT(x_start, x_goal, search_radius=30, cost_in_obstcles=5, sample_numbers=20000, step=0.1)
    fmt.Planning()
    

    # quartering = fmt.quartering(0.1)
    # save_quartering(quartering, "quartering_test")
    # tab = load_quartering("quartering")
    # save_quartering(tab, "quratering_load")


    # fmt.Init()
    # fmt.plot_nodes()

    # fmt.plot_grid(f"Fast Marching Trees (FMT*) with step = {fmt.step}")
    # plt.show()

if __name__ == '__main__':
    main()

