(define (problem task)
	(:domain reaching-grid)
	(:objects claw - gripper loc0 loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 loc9 loc10 loc11 loc12 loc13 loc14 loc15 loc16 loc17 loc18 loc19 loc20 loc21 loc22 loc23 loc24 loc25 loc26 loc27 loc28 loc29 loc30 loc31 loc32 loc33 loc34 loc35 loc36 loc37 loc38 loc39 loc40 loc41 loc42 loc43 loc44 loc45 loc46 loc47 loc48 loc49 loc50 loc51 loc52 loc53 loc54 loc55 loc56 loc57 loc58 loc59 loc60 loc61 loc62 loc63 goal - location)
	(:init
		(is_goal goal)
		(at claw loc24)
		(neighbors loc0 loc1)
		(neighbors loc0 loc8)
		(neighbors loc1 loc0)
		(neighbors loc1 loc2)
		(neighbors loc1 loc9)
		(neighbors loc2 loc1)
		(neighbors loc2 loc3)
		(neighbors loc2 loc10)
		(neighbors loc3 loc2)
		(neighbors loc3 loc4)
		(neighbors loc3 loc11)
		(neighbors loc4 loc3)
		(neighbors loc4 loc5)
		(neighbors loc4 loc12)
		(neighbors loc5 loc4)
		(neighbors loc5 loc6)
		(neighbors loc5 loc13)
		(neighbors loc6 loc5)
		(neighbors loc6 loc7)
		(neighbors loc6 loc14)
		(neighbors loc7 loc6)
		(neighbors loc7 loc15)
		(neighbors loc8 loc0)
		(neighbors loc8 loc9)
		(neighbors loc8 loc16)
		(neighbors loc9 loc1)
		(neighbors loc9 loc8)
		(neighbors loc9 loc10)
		(neighbors loc9 loc17)
		(neighbors loc10 loc2)
		(neighbors loc10 loc9)
		(neighbors loc10 loc11)
		(neighbors loc10 loc18)
		(neighbors loc11 loc3)
		(neighbors loc11 loc10)
		(neighbors loc11 loc12)
		(neighbors loc11 loc19)
		(neighbors loc12 loc4)
		(neighbors loc12 loc11)
		(neighbors loc12 loc13)
		(neighbors loc12 loc20)
		(neighbors loc13 loc5)
		(neighbors loc13 loc12)
		(neighbors loc13 loc14)
		(neighbors loc13 loc21)
		(neighbors loc14 loc6)
		(neighbors loc14 loc13)
		(neighbors loc14 loc15)
		(neighbors loc15 loc7)
		(neighbors loc15 loc14)
		(neighbors loc15 loc23)
		(neighbors loc16 loc8)
		(neighbors loc16 loc17)
		(neighbors loc16 loc24)
		(neighbors loc17 loc9)
		(neighbors loc17 loc16)
		(neighbors loc17 loc18)
		(neighbors loc17 loc25)
		(neighbors loc18 loc10)
		(neighbors loc18 loc17)
		(neighbors loc18 loc19)
		(neighbors loc18 loc26)
		(neighbors loc19 loc11)
		(neighbors loc19 loc18)
		(neighbors loc19 loc20)
		(neighbors loc19 loc27)
		(neighbors loc20 loc12)
		(neighbors loc20 loc19)
		(neighbors loc20 loc21)
		(neighbors loc20 loc28)
		(neighbors loc21 loc13)
		(neighbors loc21 loc20)
		(neighbors loc21 loc29)
		(neighbors loc23 loc15)
		(neighbors loc23 loc31)
		(neighbors loc24 loc16)
		(neighbors loc24 loc25)
		(neighbors loc24 loc32)
		(neighbors loc25 loc17)
		(neighbors loc25 loc24)
		(neighbors loc25 loc26)
		(neighbors loc25 loc33)
		(neighbors loc26 loc18)
		(neighbors loc26 loc25)
		(neighbors loc26 loc27)
		(neighbors loc26 loc34)
		(neighbors loc27 loc19)
		(neighbors loc27 loc26)
		(neighbors loc27 loc28)
		(neighbors loc27 loc35)
		(neighbors loc28 loc20)
		(neighbors loc28 loc27)
		(neighbors loc28 loc29)
		(neighbors loc28 loc36)
		(neighbors loc29 loc21)
		(neighbors loc29 loc28)
		(neighbors loc29 loc37)
		(neighbors loc31 loc23)
		(neighbors loc31 loc39)
		(neighbors loc32 loc24)
		(neighbors loc32 loc33)
		(neighbors loc32 loc40)
		(neighbors loc33 loc25)
		(neighbors loc33 loc32)
		(neighbors loc33 loc34)
		(neighbors loc33 loc41)
		(neighbors loc34 loc26)
		(neighbors loc34 loc33)
		(neighbors loc34 loc35)
		(neighbors loc34 loc42)
		(neighbors loc35 loc27)
		(neighbors loc35 loc34)
		(neighbors loc35 loc36)
		(neighbors loc35 loc43)
		(neighbors loc36 loc28)
		(neighbors loc36 loc35)
		(neighbors loc36 loc37)
		(neighbors loc36 loc44)
		(neighbors loc37 loc29)
		(neighbors loc37 loc36)
		(neighbors loc37 loc45)
		(neighbors loc39 loc31)
		(neighbors loc39 loc47)
		(neighbors loc40 loc32)
		(neighbors loc40 loc41)
		(neighbors loc40 loc48)
		(neighbors loc41 loc33)
		(neighbors loc41 loc40)
		(neighbors loc41 loc42)
		(neighbors loc41 loc49)
		(neighbors loc42 loc34)
		(neighbors loc42 loc41)
		(neighbors loc42 loc43)
		(neighbors loc42 loc50)
		(neighbors loc43 loc35)
		(neighbors loc43 loc42)
		(neighbors loc43 loc44)
		(neighbors loc43 loc51)
		(neighbors loc44 loc36)
		(neighbors loc44 loc43)
		(neighbors loc44 loc45)
		(neighbors loc44 loc52)
		(neighbors loc45 loc37)
		(neighbors loc45 loc44)
		(neighbors loc45 loc53)
		(neighbors loc47 loc39)
		(neighbors loc47 loc55)
		(neighbors loc48 loc40)
		(neighbors loc48 loc49)
		(neighbors loc48 loc56)
		(neighbors loc49 loc41)
		(neighbors loc49 loc48)
		(neighbors loc49 loc50)
		(neighbors loc49 loc57)
		(neighbors loc50 loc42)
		(neighbors loc50 loc49)
		(neighbors loc50 loc51)
		(neighbors loc50 loc58)
		(neighbors loc51 loc43)
		(neighbors loc51 loc50)
		(neighbors loc51 loc52)
		(neighbors loc51 loc59)
		(neighbors loc52 loc44)
		(neighbors loc52 loc51)
		(neighbors loc52 loc53)
		(neighbors loc52 loc60)
		(neighbors loc53 loc45)
		(neighbors loc53 loc52)
		(neighbors loc53 loc54)
		(neighbors loc53 loc61)
		(neighbors loc54 loc53)
		(neighbors loc54 loc55)
		(neighbors loc54 loc62)
		(neighbors loc55 loc47)
		(neighbors loc55 loc54)
		(neighbors loc55 loc63)
		(neighbors loc56 loc48)
		(neighbors loc56 loc57)
		(neighbors loc57 loc49)
		(neighbors loc57 loc56)
		(neighbors loc57 loc58)
		(neighbors loc58 loc50)
		(neighbors loc58 loc57)
		(neighbors loc58 loc59)
		(neighbors loc59 loc51)
		(neighbors loc59 loc58)
		(neighbors loc59 loc60)
		(neighbors loc60 loc52)
		(neighbors loc60 loc59)
		(neighbors loc60 loc61)
		(neighbors loc61 loc53)
		(neighbors loc61 loc60)
		(neighbors loc61 loc62)
		(neighbors loc62 loc54)
		(neighbors loc62 loc61)
		(neighbors loc62 loc63)
		(neighbors loc63 loc55)
		(neighbors loc63 loc62)
		(neighbors goal loc31)
		(neighbors loc31 goal)
		
	)
	(:goal (and 
		(at claw goal))
	)
)