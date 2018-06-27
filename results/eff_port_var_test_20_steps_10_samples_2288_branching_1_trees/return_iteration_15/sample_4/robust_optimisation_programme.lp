\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.22764 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.92611 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.821158 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.12503 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.94464 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.43967 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.937705 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.747287 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.846302 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.878792 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.16874 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.912666 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.955109 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.36977 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.873212 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.01257 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.932422 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.6998 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.04717 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.626061 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.15664 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.858981 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.00384 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.26242 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.14998 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.04117 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.972673 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.915597 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.30441 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.833571 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.818138 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.04458 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.10056 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.832482 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.983995 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.800075 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.23603 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.11518 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.29842 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.890532 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.872771 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.07438 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.01071 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.56186 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.954705 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.654983 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.928008 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.604222 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.14837 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.25847 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.902287 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.52443 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.912402 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.43751 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.33468 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.629513 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.14926 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.554497 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.59369 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.827781 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.958969 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.5124 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.07599 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.01543 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.759435 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.75185 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.5928 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.553865 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.939467 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.07141 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.59464 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.75221 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.23173 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.770487 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.28392 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.875999 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.40186 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.924972 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.770167 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.10323 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.924114 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.52811 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.974147 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.523786 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.03457 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.684951 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.19014 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.42968 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.825081 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.60108 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.70537 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.42928 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.24299 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.659668 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.11021 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.544617 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.86804 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.864094 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.744349 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.5675 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.18868 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.13902 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.820877 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.683454 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.62005 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.54419 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.33026 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.24468 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.5642 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.708974 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.971719 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.829915 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.70473 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.17092 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.27466 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.938279 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.813949 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.07861 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.934286 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.35101 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.965982 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.738499 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.891711 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.825622 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.22717 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.937194 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.911744 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.31904 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.847178 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.08721 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.29002 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.688921 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.07708 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.706174 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.4221 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.950183 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.820289 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.17004 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.21915 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.979769 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.944564 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.776443 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.43454 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.710314 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.11767 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.0389 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.26204 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.920543 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.06609 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.827611 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.44402 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.10331 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.09451 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.0099 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.15735 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.0129 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.00951 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.20173 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.939826 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.609508 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.21341 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.479915 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.74777 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 2.06876 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.696848 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.47073 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.883737 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.90369 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.915115 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.978038 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.04162 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.52735 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.3112 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.759371 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.36107 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.98137 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.784093 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.28704 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.606685 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.899066 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.1713 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.523121 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.04017 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.57508 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.93234 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.555321 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.17503 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.879361 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.00633 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.732451 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.17498 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.92605 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.83581 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.05195 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.0068 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.37499 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.05267 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.833031 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.942037 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.00043 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.19456 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.15719 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.03828 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.33471 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.82489 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.08948 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.19455 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.983986 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.124 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.823972 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.24967 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.940154 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.907077 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.19774 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.12196 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.07024 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.901707 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.847011 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.46238 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.752668 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.15526 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.994292 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.21966 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.898009 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.02072 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.715213 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.10355 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.928014 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15233 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.977636 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.918261 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.02612 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.02512 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.2463 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.04481 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.693966 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.07466 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.680522 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.890498 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.69878 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.758246 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.4542 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.831923 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.59097 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.963603 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.979085 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.13275 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.614462 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.54875 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.883648 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.968765 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.45748 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.867842 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.01584 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.654935 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.758851 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.26212 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.556504 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.16944 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.28051 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.50711 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.65017 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.07796 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.841975 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.22586 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.951741 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.05202 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.03364 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.34836 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.00676 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.963018 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.22275 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.805294 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.538606 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.909346 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.387543 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.928776 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 2.07897 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.59233 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.34591 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.02698 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 2.40884 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.0967 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.814022 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.993013 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.541696 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.48903 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.865126 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.16355 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.92476 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.73559 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.18294 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.584417 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.722966 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.28962 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.50496 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.859131 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.41616 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 2.53148 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.576991 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.38255 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.816659 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.09153 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.593427 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18255 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.94374 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.868914 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.03742 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.992619 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.35323 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.04218 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.80381 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.918503 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.893778 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.14468 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.02336 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.919622 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.18782 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.934738 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.23281 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.24262 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.87021 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.10796 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.833522 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.47235 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.05076 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.773197 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.2194 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.15503 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.991865 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.869974 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.815449 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.26908 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.707811 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.13232 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.924604 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.23391 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.807557 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.990518 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.820672 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.28069 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.982816 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11028 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.958272 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.903786 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.08032 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.04232 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.1794 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.951411 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.808244 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.00994 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.812353 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.12785 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.19572 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.888244 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.1769 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.909691 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.15644 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.11788 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.849312 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.01432 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.72323 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.24653 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.924732 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.929568 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.16604 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.986514 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.978422 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.850933 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.834494 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.15932 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.703815 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.08795 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.11177 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.34223 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.90253 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.08036 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.873585 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.14227 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.976039 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.131000388892161 z_k1_s0000
                                 - 0.219000293506375 z_k1_s0001
                                 - 0.0760001254579443 z_k1_s0002
                                 - 0.157000473664658 z_k1_s0003
                                 - 0.14999972445593 z_k1_s0004
                                 - 0.100000478971303 z_k1_s0005
                                 - 0.0830000794658536 z_k1_s0006
                                 - 0.0780000555172885 z_k1_s0007
                                 - 0.0409998886491736 z_k1_s0010
                                 - 0.0240000228678855 z_k1_s0011
                                 - 0.0210000019593841 z_k1_s0012
                                 - 0.0570001445613065 z_k1_s0013
                                 - 0.0409998886491736 z_k1_s0014
                                 - 0.0610000280391828 z_k1_s0015
                                 - 0.0360000343018283 z_k1_s0016
                                 - 0.080000196559723 z_k1_s0017
                                 - 0.0659999286187586 z_k1_s0020
                                 - 0.0859998795517614 z_k1_s0021
                                 - 0.0849998166152201 z_k1_s0022
                                 - 0.206999245686989 z_k1_s0023
                                 - 0.269999890894618 z_k1_s0024
                                 - 0.0789998413953025 z_k1_s0025
                                 - 0.10600003168443 z_k1_s0026
                                 - 0.106999994021138 z_k1_s0027
                                 - 0.103000382522477 z_k1_s0030
                                 - 0.0709999910877947 z_k1_s0031
                                 - 0.0440000654967956 z_k1_s0032
                                 - 0.041000061031105 z_k1_s0033
                                 - 0.0340000506111603 z_k1_s0034
                                 - 0.104000269410824 z_k1_s0035
                                 - 0.0550000818709945 z_k1_s0036
                                 - 0.121000065515972 z_k1_s0037
                                 - 0.0560000019775996 z_k1_s0040
                                 - 0.0359999512713123 z_k1_s0041
                                 - 0.0729998525779371 z_k1_s0042
                                 - 0.032999996165371 z_k1_s0043
                                 - 0.0449998515891373 z_k1_s0044
                                 - 0.0680001024013745 z_k1_s0045
                                 - 0.0180000106356574 z_k1_s0046
                                 - 0.0210000007415999 z_k1_s0047
                                 - 0.0260000201298786 z_k1_s0050
                                 - 0.0150000007287687 z_k1_s0051
                                 - 0.0670000126885066 z_k1_s0052
                                 - 0.0479999457320213 z_k1_s0053
                                 - 0.0300000014575374 z_k1_s0054
                                 - 0.0220000105022005 z_k1_s0055
                                 - 0.0450001436864026 z_k1_s0056
                                 - 0.0300000014575374 z_k1_s0057
                                 - 0.119999408722507 z_k1_s0060
                                 - 0.0599998261608602 z_k1_s0061
                                 - 0.0599998261608602 z_k1_s0062
                                 - 0.0459998748432999 z_k1_s0063
                                 - 0.0579998157155506 z_k1_s0064
                                 - 0.0879996069963737 z_k1_s0065
                                 - 0.0859996574508676 z_k1_s0066
                                 - 0.090999531314633 z_k1_s0067
                                 - 0.114999487893258 z_k1_s0070
                                 - 0.0509998595915974 z_k1_s0071
                                 - 0.0960000927010261 z_k1_s0072
                                 - 0.2079997673532 z_k1_s0073
                                 - 0.0429998952164142 z_k1_s0074
                                 - 0.0779998260576453 z_k1_s0075
                                 - 0.120999981373473 z_k1_s0076
                                 - 0.154999309769174 z_k1_s0077
                                 - 0.457999482541856 z_k1_s0100
                                 - 0.574000739306441 z_k1_s0101
                                 - 0.764000971296351 z_k1_s0102
                                 - 0.38400050731653 z_k1_s0103
                                 - 0.790003079488772 z_k1_s0104
                                 - 0.33000065287247 z_k1_s0105
                                 - 0.166000337270413 z_k1_s0106
                                 - 0.187000382137759 z_k1_s0107
                                 - 0.199999763353136 z_k1_s0110
                                 - 0.377998603973069 z_k1_s0111
                                 - 0.158999656893011 z_k1_s0112
                                 - 0.586998852353238 z_k1_s0113
                                 - 0.128999775199898 z_k1_s0114
                                 - 0.296000283849544 z_k1_s0115
                                 - 0.497999550343067 z_k1_s0116
                                 - 0.118999893502057 z_k1_s0117
                                 - 0.284999998845236 z_k1_s0120
                                 - 0.136999908402854 z_k1_s0121
                                 - 0.33600005150172 z_k1_s0122
                                 - 0.100000011342181 z_k1_s0123
                                 - 0.209000445552899 z_k1_s0124
                                 - 0.135999861417461 z_k1_s0125
                                 - 0.0959999908005061 z_k1_s0126
                                 - 0.374999205533693 z_k1_s0127
                                 - 0.137999924338969 z_k1_s0130
                                 - 0.322000016924202 z_k1_s0131
                                 - 0.233999796000884 z_k1_s0132
                                 - 0.329999377462888 z_k1_s0133
                                 - 0.10699989927733 z_k1_s0134
                                 - 0.119999984677349 z_k1_s0135
                                 - 0.0860000180387579 z_k1_s0136
                                 - 0.113999956423491 z_k1_s0137
                                 - 0.206999661698498 z_k1_s0140
                                 - 0.279999528302335 z_k1_s0141
                                 - 0.777997835091233 z_k1_s0142
                                 - 0.464998331128309 z_k1_s0143
                                 - 0.206999661698498 z_k1_s0144
                                 - 0.518000148307917 z_k1_s0145
                                 - 0.256999598302457 z_k1_s0146
                                 - 0.204999769246195 z_k1_s0147
                                 - 0.135999860098557 z_k1_s0150
                                 - 0.233000714506737 z_k1_s0151
                                 - 0.225999217075745 z_k1_s0152
                                 - 0.344999973411661 z_k1_s0153
                                 - 0.351000171210052 z_k1_s0154
                                 - 0.138999958997753 z_k1_s0155
                                 - 0.102999912206989 z_k1_s0156
                                 - 0.367000065339326 z_k1_s0157
                                 - 0.240001127602988 z_k1_s0160
                                 - 0.290000158768563 z_k1_s0161
                                 - 0.240001127602988 z_k1_s0162
                                 - 0.261999930914744 z_k1_s0163
                                 - 0.107000237897622 z_k1_s0164
                                 - 0.509000152479816 z_k1_s0165
                                 - 0.114000102260802 z_k1_s0166
                                 - 0.164000289028023 z_k1_s0167
                                 - 0.767998797895163 z_k1_s0170
                                 - 0.492000424823888 z_k1_s0171
                                 - 0.206000115698598 z_k1_s0172
                                 - 0.283999988414511 z_k1_s0173
                                 - 0.4240008256614 z_k1_s0174
                                 - 0.70799915157532 z_k1_s0175
                                 - 0.23900008742457 z_k1_s0176
                                 - 0.204000127487936 z_k1_s0177
                                 - 0.310999596459951 z_k1_s1000
                                 - 0.243999614840039 z_k1_s1001
                                 - 0.241000464259076 z_k1_s1002
                                 - 0.354998983300577 z_k1_s1003
                                 - 0.571999135920048 z_k1_s1004
                                 - 0.20699966742004 z_k1_s1005
                                 - 0.320999134060461 z_k1_s1006
                                 - 0.117999755480085 z_k1_s1007
                                 - 0.540997628222956 z_k1_s1010
                                 - 1.24199634424479 z_k1_s1011
                                 - 0.944993913245025 z_k1_s1012
                                 - 0.400997943393073 z_k1_s1013
                                 - 0.670993760831527 z_k1_s1014
                                 - 0.645995966476962 z_k1_s1015
                                 - 0.693996039417799 z_k1_s1016
                                 - 1.19499719667345 z_k1_s1017
                                 - 1.94199759825844 z_k1_s1020
                                 - 0.578000897870287 z_k1_s1021
                                 - 1.46800225796743 z_k1_s1022
                                 - 1.98400443564356 z_k1_s1023
                                 - 1.47200085066938 z_k1_s1024
                                 - 1.52299911813024 z_k1_s1025
                                 - 1.25900554374289 z_k1_s1026
                                 - 0.581000923097482 z_k1_s1027
                                 - 1.18899878362617 z_k1_s1030
                                 - 1.21300083335742 z_k1_s1031
                                 - 1.46899471050859 z_k1_s1032
                                 - 0.99699917776547 z_k1_s1033
                                 - 1.28600400691693 z_k1_s1034
                                 - 0.739999880253899 z_k1_s1035
                                 - 2.33699362833673 z_k1_s1036
                                 - 1.26399600591718 z_k1_s1037
                                 - 1.82199688061293 z_k1_s1040
                                 - 1.79599756645957 z_k1_s1041
                                 - 1.23399975118232 z_k1_s1042
                                 - 0.692997660140097 z_k1_s1043
                                 - 0.579998323580478 z_k1_s1044
                                 - 0.644998434758429 z_k1_s1045
                                 - 1.45399886346533 z_k1_s1046
                                 - 0.904997053675701 z_k1_s1047
                                 - 1.09899964875489 z_k1_s1050
                                 - 0.878000539842924 z_k1_s1051
                                 - 1.31600560083057 z_k1_s1052
                                 - 1.35799815336708 z_k1_s1053
                                 - 1.09600002070227 z_k1_s1054
                                 - 1.13799929383934 z_k1_s1055
                                 - 2.14900163898333 z_k1_s1056
                                 - 2.16700164749925 z_k1_s1057
                                 - 0.306000071559936 z_k1_s1060
                                 - 1.12100411028428 z_k1_s1061
                                 - 0.769998651930173 z_k1_s1062
                                 - 0.410000193126165 z_k1_s1063
                                 - 0.649000888057671 z_k1_s1064
                                 - 1.48100044358594 z_k1_s1065
                                 - 1.16799896727962 z_k1_s1066
                                 - 1.18000097055869 z_k1_s1067
                                 - 0.325999804052393 z_k1_s1070
                                 - 0.406000144882991 z_k1_s1071
                                 - 0.474999158025797 z_k1_s1072
                                 - 0.617000131797562 z_k1_s1073
                                 - 0.370999871175871 z_k1_s1074
                                 - 0.536000250195272 z_k1_s1075
                                 - 0.987998408173638 z_k1_s1076
                                 - 0.268000125097636 z_k1_s1077
                                 - 0.0690000319037434 z_k1_s1100
                                 - 0.150000216616978 z_k1_s1101
                                 - 0.151999677584671 z_k1_s1102
                                 - 0.125999910997995 z_k1_s1103
                                 - 0.117999809125001 z_k1_s1104
                                 - 0.113000592205213 z_k1_s1105
                                 - 0.284000229487968 z_k1_s1106
                                 - 0.11700064314171 z_k1_s1107
                                 - 0.0979997424710144 z_k1_s1110
                                 - 0.324998807215975 z_k1_s1111
                                 - 0.291000015172117 z_k1_s1112
                                 - 0.215998865202887 z_k1_s1113
                                 - 0.105999648974958 z_k1_s1114
                                 - 0.286999521521543 z_k1_s1115
                                 - 0.125999646834218 z_k1_s1116
                                 - 0.0949997582320854 z_k1_s1117
                                 - 0.0750001346101707 z_k1_s1120
                                 - 0.174999937089737 z_k1_s1121
                                 - 0.0740001548554071 z_k1_s1122
                                 - 0.0360000541728636 z_k1_s1123
                                 - 0.150000443220647 z_k1_s1124
                                 - 0.0620001367977859 z_k1_s1125
                                 - 0.163000006032268 z_k1_s1126
                                 - 0.134999876897666 z_k1_s1127
                                 - 0.486001707122131 z_k1_s1130
                                 - 0.126000442587219 z_k1_s1131
                                 - 0.216000758720947 z_k1_s1132
                                 - 0.314002102958963 z_k1_s1133
                                 - 0.203000663057011 z_k1_s1134
                                 - 0.216000758720947 z_k1_s1135
                                 - 0.163000472552956 z_k1_s1136
                                 - 0.526002347627766 z_k1_s1137
                                 - 0.40399986268847 z_k1_s1140
                                 - 0.226000399811621 z_k1_s1141
                                 - 0.311999811066214 z_k1_s1142
                                 - 0.268000696378729 z_k1_s1143
                                 - 0.134000138889038 z_k1_s1144
                                 - 0.3360002795336 z_k1_s1145
                                 - 0.241000954229146 z_k1_s1146
                                 - 0.172000287611473 z_k1_s1147
                                 - 0.553000480447573 z_k1_s1150
                                 - 0.2490005526278 z_k1_s1151
                                 - 0.763003764069372 z_k1_s1152
                                 - 0.550001175940796 z_k1_s1153
                                 - 0.219000454740024 z_k1_s1154
                                 - 0.171000446600004 z_k1_s1155
                                 - 0.36400086599743 z_k1_s1156
                                 - 0.843001302962386 z_k1_s1157
                                 - 0.116000461141577 z_k1_s1160
                                 - 0.0480000548239166 z_k1_s1161
                                 - 0.204999940705913 z_k1_s1162
                                 - 0.0770000762091837 z_k1_s1163
                                 - 0.0939996613378992 z_k1_s1164
                                 - 0.0670000882625662 z_k1_s1165
                                 - 0.200000697933622 z_k1_s1166
                                 - 0.132000103815707 z_k1_s1167
                                 - 0.197999749251304 z_k1_s1170
                                 - 0.264000265667645 z_k1_s1171
                                 - 0.116999851830316 z_k1_s1172
                                 - 0.112999896895895 z_k1_s1173
                                 - 0.212000131521091 z_k1_s1174
                                 - 0.170999783444308 z_k1_s1175
                                 - 0.199999546718742 z_k1_s1176
                                 - 0.524999935135273 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.850203 w_k1_ADM_s000
                                 + 0.938894 w_k1_AXP_s000
                                 + 1.1151 w_k1_DIS_s000
                                 + 0.979983 w_k1_FDX_s000
                                 + 1.13354 w_k1_IBM_s000
                                 + 1.15649 w_k1_INTC_s000
                                 + 1.04647 w_k1_KO_s000 + 1.15443 w_k1_MAT_s000
                                 + 1.02273 w_k1_MSFT_s000
                                 + 0.993915 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.718286 w_k1_ADM_s000
                                 + 0.786233 w_k1_AXP_s000
                                 + 0.774017 w_k1_DIS_s000
                                 + 0.800445 w_k1_FDX_s000
                                 + 0.763147 w_k1_IBM_s000
                                 + 0.731016 w_k1_INTC_s000
                                 + 0.741322 w_k1_KO_s000
                                 + 1.03052 w_k1_MAT_s000
                                 + 0.733653 w_k1_MSFT_s000
                                 + 0.979522 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.10452 w_k1_ADM_s000
                                 + 1.27232 w_k1_AXP_s000 + 1.3835 w_k1_DIS_s000
                                 + 1.124 w_k1_FDX_s000 + 1.51075 w_k1_IBM_s000
                                 + 1.30573 w_k1_INTC_s000
                                 + 1.36645 w_k1_KO_s000 + 1.27551 w_k1_MAT_s000
                                 + 1.1172 w_k1_MSFT_s000 + 1.22754 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.61095 w_k1_ADM_s000
                                 + 1.39058 w_k1_AXP_s000
                                 + 1.65725 w_k1_DIS_s000 + 1.3385 w_k1_FDX_s000
                                 + 1.23469 w_k1_IBM_s000
                                 + 1.12014 w_k1_INTC_s000
                                 + 1.39249 w_k1_KO_s000 + 1.527 w_k1_MAT_s000
                                 + 1.16919 w_k1_MSFT_s000
                                 + 1.28879 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.909764 w_k1_ADM_s000
                                 + 0.993679 w_k1_AXP_s000
                                 + 0.763349 w_k1_DIS_s000
                                 + 0.860509 w_k1_FDX_s000
                                 + 1.0811 w_k1_IBM_s000
                                 + 0.900108 w_k1_INTC_s000
                                 + 0.967241 w_k1_KO_s000
                                 + 1.17685 w_k1_MAT_s000
                                 + 0.759613 w_k1_MSFT_s000
                                 + 0.975459 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.18936 w_k1_ADM_s000
                                 + 1.21004 w_k1_AXP_s000
                                 + 0.970118 w_k1_DIS_s000
                                 + 1.1033 w_k1_FDX_s000
                                 + 0.853023 w_k1_IBM_s000
                                 + 1.07875 w_k1_INTC_s000
                                 + 0.929469 w_k1_KO_s000
                                 + 0.90664 w_k1_MAT_s000
                                 + 0.927448 w_k1_MSFT_s000
                                 + 1.0249 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.994354 w_k1_ADM_s000
                                 + 0.845414 w_k1_AXP_s000
                                 + 1.07346 w_k1_DIS_s000
                                 + 1.10151 w_k1_FDX_s000
                                 + 0.695993 w_k1_IBM_s000
                                 + 1.05597 w_k1_INTC_s000
                                 + 0.935983 w_k1_KO_s000
                                 + 0.65319 w_k1_MAT_s000
                                 + 1.09035 w_k1_MSFT_s000 + 1.0845 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.16915 w_k1_ADM_s000
                                 + 1.07326 w_k1_AXP_s000
                                 + 1.40295 w_k1_DIS_s000
                                 + 1.29475 w_k1_FDX_s000
                                 + 0.888961 w_k1_IBM_s000
                                 + 1.27004 w_k1_INTC_s000
                                 + 1.17836 w_k1_KO_s000
                                 + 0.964429 w_k1_MAT_s000
                                 + 1.15008 w_k1_MSFT_s000 + 1.1814 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.63828 w_k1_ADM_s001
                                 + 1.2545 w_k1_AXP_s001 + 1.73833 w_k1_DIS_s001
                                 + 1.25989 w_k1_FDX_s001 + 1.2235 w_k1_IBM_s001
                                 + 0.981366 w_k1_INTC_s001
                                 + 1.2435 w_k1_KO_s001 + 1.56445 w_k1_MAT_s001
                                 + 1.16821 w_k1_MSFT_s001
                                 + 1.24678 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.08807 w_k1_ADM_s001
                                 + 1.01762 w_k1_AXP_s001
                                 + 1.07359 w_k1_DIS_s001
                                 + 0.923286 w_k1_FDX_s001
                                 + 1.7347 w_k1_IBM_s001
                                 + 0.906979 w_k1_INTC_s001
                                 + 1.01659 w_k1_KO_s001 + 1.47809 w_k1_MAT_s001
                                 + 0.8502 w_k1_MSFT_s001 + 1.00229 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.27639 w_k1_ADM_s001
                                 + 1.37681 w_k1_AXP_s001
                                 + 1.68439 w_k1_DIS_s001
                                 + 1.06461 w_k1_FDX_s001
                                 + 1.66391 w_k1_IBM_s001
                                 + 1.11254 w_k1_INTC_s001
                                 + 1.45172 w_k1_KO_s001 + 1.49196 w_k1_MAT_s001
                                 + 1.1192 w_k1_MSFT_s001 + 1.2141 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.30661 w_k1_ADM_s001
                                 + 1.25045 w_k1_AXP_s001
                                 + 1.26185 w_k1_DIS_s001
                                 + 1.21549 w_k1_FDX_s001
                                 + 0.859102 w_k1_IBM_s001
                                 + 1.18888 w_k1_INTC_s001
                                 + 0.973432 w_k1_KO_s001
                                 + 0.658362 w_k1_MAT_s001
                                 + 0.913693 w_k1_MSFT_s001
                                 + 1.04223 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.21135 w_k1_ADM_s001
                                 + 1.27091 w_k1_AXP_s001
                                 + 1.16567 w_k1_DIS_s001
                                 + 1.16064 w_k1_FDX_s001 + 1.3324 w_k1_IBM_s001
                                 + 1.09932 w_k1_INTC_s001
                                 + 1.13441 w_k1_KO_s001 + 1.64219 w_k1_MAT_s001
                                 + 0.996888 w_k1_MSFT_s001
                                 + 1.1144 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.806809 w_k1_ADM_s001
                                 + 0.730272 w_k1_AXP_s001
                                 + 0.757012 w_k1_DIS_s001
                                 + 0.827366 w_k1_FDX_s001
                                 + 0.739624 w_k1_IBM_s001
                                 + 0.856226 w_k1_INTC_s001
                                 + 0.754936 w_k1_KO_s001
                                 + 0.785355 w_k1_MAT_s001
                                 + 0.733082 w_k1_MSFT_s001
                                 + 0.886866 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.737588 w_k1_ADM_s001
                                 + 0.758332 w_k1_AXP_s001
                                 + 0.748505 w_k1_DIS_s001
                                 + 0.708901 w_k1_FDX_s001
                                 + 0.949881 w_k1_IBM_s001
                                 + 0.817628 w_k1_INTC_s001
                                 + 0.852298 w_k1_KO_s001
                                 + 0.631047 w_k1_MAT_s001
                                 + 0.750293 w_k1_MSFT_s001
                                 + 0.863534 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.954029 w_k1_ADM_s001
                                 + 0.916657 w_k1_AXP_s001
                                 + 0.826358 w_k1_DIS_s001
                                 + 0.871241 w_k1_FDX_s001
                                 + 1.09417 w_k1_IBM_s001
                                 + 0.941233 w_k1_INTC_s001
                                 + 0.949267 w_k1_KO_s001
                                 + 0.939014 w_k1_MAT_s001
                                 + 0.84061 w_k1_MSFT_s001
                                 + 1.01723 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.64842 w_k1_ADM_s002
                                 + 1.4573 w_k1_AXP_s002 + 1.35668 w_k1_DIS_s002
                                 + 1.27871 w_k1_FDX_s002 + 1.6653 w_k1_IBM_s002
                                 + 1.22307 w_k1_INTC_s002
                                 + 1.42353 w_k1_KO_s002 + 1.54813 w_k1_MAT_s002
                                 + 1.20314 w_k1_MSFT_s002
                                 + 1.15218 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.06507 w_k1_ADM_s002
                                 + 1.24932 w_k1_AXP_s002
                                 + 1.05996 w_k1_DIS_s002
                                 + 1.12218 w_k1_FDX_s002
                                 + 1.07496 w_k1_IBM_s002
                                 + 1.00964 w_k1_INTC_s002
                                 + 1.14851 w_k1_KO_s002 + 1.44018 w_k1_MAT_s002
                                 + 0.943747 w_k1_MSFT_s002
                                 + 1.10089 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.983766 w_k1_ADM_s002
                                 + 0.854373 w_k1_AXP_s002
                                 + 1.0602 w_k1_DIS_s002
                                 + 0.951065 w_k1_FDX_s002
                                 + 1.35581 w_k1_IBM_s002
                                 + 1.1628 w_k1_INTC_s002 + 1.38529 w_k1_KO_s002
                                 + 1.65371 w_k1_MAT_s002
                                 + 1.26455 w_k1_MSFT_s002
                                 + 1.03584 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.833691 w_k1_ADM_s002
                                 + 1.03772 w_k1_AXP_s002
                                 + 0.737894 w_k1_DIS_s002
                                 + 0.838128 w_k1_FDX_s002
                                 + 1.07874 w_k1_IBM_s002
                                 + 1.22541 w_k1_INTC_s002
                                 + 1.19242 w_k1_KO_s002 + 1.41328 w_k1_MAT_s002
                                 + 0.988021 w_k1_MSFT_s002
                                 + 1.03697 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.918783 w_k1_ADM_s002
                                 + 0.904771 w_k1_AXP_s002
                                 + 0.798915 w_k1_DIS_s002
                                 + 0.912954 w_k1_FDX_s002
                                 + 0.819473 w_k1_IBM_s002
                                 + 1.05362 w_k1_INTC_s002
                                 + 0.91263 w_k1_KO_s002 + 1.23322 w_k1_MAT_s002
                                 + 0.835058 w_k1_MSFT_s002
                                 + 0.934089 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.75599 w_k1_ADM_s002
                                 + 0.721629 w_k1_AXP_s002
                                 + 0.689651 w_k1_DIS_s002
                                 + 0.869293 w_k1_FDX_s002
                                 + 0.643038 w_k1_IBM_s002
                                 + 0.891618 w_k1_INTC_s002
                                 + 0.73384 w_k1_KO_s002
                                 + 0.745466 w_k1_MAT_s002
                                 + 0.881901 w_k1_MSFT_s002
                                 + 0.85223 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.19955 w_k1_ADM_s002
                                 + 1.45066 w_k1_AXP_s002
                                 + 1.39569 w_k1_DIS_s002
                                 + 1.37633 w_k1_FDX_s002
                                 + 1.19152 w_k1_IBM_s002
                                 + 1.42005 w_k1_INTC_s002
                                 + 1.37323 w_k1_KO_s002 + 1.71607 w_k1_MAT_s002
                                 + 1.39266 w_k1_MSFT_s002 + 1.2656 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.13602 w_k1_ADM_s002
                                 + 1.48523 w_k1_AXP_s002
                                 + 1.28727 w_k1_DIS_s002
                                 + 1.32198 w_k1_FDX_s002
                                 + 0.947121 w_k1_IBM_s002
                                 + 1.65964 w_k1_INTC_s002
                                 + 1.15494 w_k1_KO_s002 + 1.19353 w_k1_MAT_s002
                                 + 1.50518 w_k1_MSFT_s002
                                 + 1.13645 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.23479 w_k1_ADM_s003
                                 + 1.19633 w_k1_AXP_s003
                                 + 1.10327 w_k1_DIS_s003
                                 + 1.06753 w_k1_FDX_s003
                                 + 1.05345 w_k1_IBM_s003
                                 + 0.859705 w_k1_INTC_s003
                                 + 1.11048 w_k1_KO_s003 + 1.30852 w_k1_MAT_s003
                                 + 1.0042 w_k1_MSFT_s003 + 1.0682 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.748652 w_k1_ADM_s003
                                 + 0.836293 w_k1_AXP_s003
                                 + 0.647209 w_k1_DIS_s003
                                 + 0.7748 w_k1_FDX_s003 + 0.79653 w_k1_IBM_s003
                                 + 0.800107 w_k1_INTC_s003
                                 + 0.715122 w_k1_KO_s003
                                 + 0.810905 w_k1_MAT_s003
                                 + 0.816938 w_k1_MSFT_s003
                                 + 0.8154 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.819279 w_k1_ADM_s003
                                 + 0.927994 w_k1_AXP_s003
                                 + 0.740712 w_k1_DIS_s003
                                 + 0.971088 w_k1_FDX_s003
                                 + 0.547998 w_k1_IBM_s003
                                 + 0.915609 w_k1_INTC_s003
                                 + 0.771271 w_k1_KO_s003
                                 + 0.901299 w_k1_MAT_s003
                                 + 0.935441 w_k1_MSFT_s003
                                 + 0.900547 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.17 w_k1_ADM_s003
                                 + 1.17089 w_k1_AXP_s003
                                 + 1.60666 w_k1_DIS_s003
                                 + 1.30578 w_k1_FDX_s003
                                 + 0.820244 w_k1_IBM_s003
                                 + 1.15723 w_k1_INTC_s003 + 1.1615 w_k1_KO_s003
                                 + 1.1106 w_k1_MAT_s003
                                 + 1.32105 w_k1_MSFT_s003 + 1.1559 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.78493 w_k1_ADM_s003
                                 + 0.533192 w_k1_AXP_s003
                                 + 0.553758 w_k1_DIS_s003
                                 + 0.602314 w_k1_FDX_s003
                                 + 0.703215 w_k1_IBM_s003
                                 + 0.514012 w_k1_INTC_s003
                                 + 0.644804 w_k1_KO_s003
                                 + 0.47151 w_k1_MAT_s003
                                 + 0.573072 w_k1_MSFT_s003
                                 + 0.765129 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.15898 w_k1_ADM_s003
                                 + 1.44226 w_k1_AXP_s003
                                 + 1.26874 w_k1_DIS_s003
                                 + 1.21349 w_k1_FDX_s003
                                 + 1.45684 w_k1_IBM_s003
                                 + 1.05852 w_k1_INTC_s003
                                 + 1.26617 w_k1_KO_s003 + 1.15595 w_k1_MAT_s003
                                 + 1.12558 w_k1_MSFT_s003
                                 + 1.19121 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.78534 w_k1_ADM_s003
                                 + 0.748387 w_k1_AXP_s003
                                 + 0.772712 w_k1_DIS_s003
                                 + 0.794044 w_k1_FDX_s003
                                 + 1.46422 w_k1_IBM_s003
                                 + 0.823839 w_k1_INTC_s003
                                 + 1.1342 w_k1_KO_s003 + 1.56815 w_k1_MAT_s003
                                 + 0.914587 w_k1_MSFT_s003
                                 + 1.0354 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.864101 w_k1_ADM_s003
                                 + 0.797948 w_k1_AXP_s003
                                 + 1.15462 w_k1_DIS_s003
                                 + 0.922523 w_k1_FDX_s003
                                 + 1.10822 w_k1_IBM_s003
                                 + 0.994314 w_k1_INTC_s003
                                 + 1.09462 w_k1_KO_s003
                                 + 0.994632 w_k1_MAT_s003
                                 + 1.05971 w_k1_MSFT_s003
                                 + 1.11259 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.201 w_k1_ADM_s004
                                 + 1.09251 w_k1_AXP_s004
                                 + 0.968051 w_k1_DIS_s004
                                 + 1.05905 w_k1_FDX_s004
                                 + 0.907715 w_k1_IBM_s004
                                 + 0.93997 w_k1_INTC_s004
                                 + 1.00146 w_k1_KO_s004
                                 + 0.689688 w_k1_MAT_s004
                                 + 0.953289 w_k1_MSFT_s004
                                 + 0.838559 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.883729 w_k1_ADM_s004
                                 + 0.719196 w_k1_AXP_s004
                                 + 0.69655 w_k1_DIS_s004
                                 + 0.77938 w_k1_FDX_s004
                                 + 0.876311 w_k1_IBM_s004
                                 + 0.81803 w_k1_INTC_s004
                                 + 1.08844 w_k1_KO_s004 + 0.93991 w_k1_MAT_s004
                                 + 0.850412 w_k1_MSFT_s004
                                 + 0.96407 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.948814 w_k1_ADM_s004
                                 + 1.129 w_k1_AXP_s004 + 1.02102 w_k1_DIS_s004
                                 + 0.921678 w_k1_FDX_s004
                                 + 1.34463 w_k1_IBM_s004
                                 + 0.863617 w_k1_INTC_s004 + 1.447 w_k1_KO_s004
                                 + 1.25725 w_k1_MAT_s004
                                 + 1.05814 w_k1_MSFT_s004
                                 + 1.10103 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.99793 w_k1_ADM_s004
                                 + 0.967897 w_k1_AXP_s004
                                 + 0.899767 w_k1_DIS_s004
                                 + 0.832498 w_k1_FDX_s004
                                 + 1.24224 w_k1_IBM_s004
                                 + 0.873224 w_k1_INTC_s004
                                 + 1.12412 w_k1_KO_s004
                                 + 0.945244 w_k1_MAT_s004
                                 + 0.85803 w_k1_MSFT_s004
                                 + 0.849246 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.20998 w_k1_ADM_s004
                                 + 1.34564 w_k1_AXP_s004
                                 + 1.61436 w_k1_DIS_s004
                                 + 1.33219 w_k1_FDX_s004
                                 + 1.39715 w_k1_IBM_s004
                                 + 1.17112 w_k1_INTC_s004
                                 + 1.54195 w_k1_KO_s004 + 1.17321 w_k1_MAT_s004
                                 + 1.28518 w_k1_MSFT_s004 + 1.2983 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.03733 w_k1_ADM_s004
                                 + 1.11294 w_k1_AXP_s004 + 1.3016 w_k1_DIS_s004
                                 + 1.07617 w_k1_FDX_s004
                                 + 1.22564 w_k1_IBM_s004
                                 + 0.923289 w_k1_INTC_s004
                                 + 1.2505 w_k1_KO_s004 + 0.966937 w_k1_MAT_s004
                                 + 1.02381 w_k1_MSFT_s004
                                 + 1.02953 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.0708 w_k1_ADM_s004
                                 + 0.823926 w_k1_AXP_s004
                                 + 0.681939 w_k1_DIS_s004
                                 + 0.916656 w_k1_FDX_s004
                                 + 0.560956 w_k1_IBM_s004
                                 + 0.963726 w_k1_INTC_s004
                                 + 0.833193 w_k1_KO_s004
                                 + 0.406046 w_k1_MAT_s004
                                 + 0.917894 w_k1_MSFT_s004
                                 + 0.877222 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.952348 w_k1_ADM_s004
                                 + 1.04702 w_k1_AXP_s004
                                 + 0.679128 w_k1_DIS_s004
                                 + 0.877069 w_k1_FDX_s004
                                 + 0.74451 w_k1_IBM_s004
                                 + 0.956556 w_k1_INTC_s004
                                 + 0.962351 w_k1_KO_s004
                                 + 0.705822 w_k1_MAT_s004
                                 + 0.90591 w_k1_MSFT_s004
                                 + 1.01967 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.58741 w_k1_ADM_s005
                                 + 0.788511 w_k1_AXP_s005
                                 + 1.18613 w_k1_DIS_s005
                                 + 1.41379 w_k1_FDX_s005
                                 + 0.791301 w_k1_IBM_s005
                                 + 1.1362 w_k1_INTC_s005
                                 + 0.995505 w_k1_KO_s005
                                 + 0.504642 w_k1_MAT_s005
                                 + 1.28443 w_k1_MSFT_s005
                                 + 1.02079 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.83504 w_k1_ADM_s005
                                 + 1.04462 w_k1_AXP_s005
                                 + 1.65949 w_k1_DIS_s005
                                 + 1.41354 w_k1_FDX_s005
                                 + 1.02561 w_k1_IBM_s005
                                 + 1.0862 w_k1_INTC_s005 + 1.34815 w_k1_KO_s005
                                 + 0.834917 w_k1_MAT_s005
                                 + 1.37332 w_k1_MSFT_s005
                                 + 1.19581 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.947434 w_k1_ADM_s005
                                 + 0.781818 w_k1_AXP_s005
                                 + 0.717256 w_k1_DIS_s005
                                 + 0.931596 w_k1_FDX_s005
                                 + 0.784736 w_k1_IBM_s005
                                 + 0.7833 w_k1_INTC_s005
                                 + 0.754425 w_k1_KO_s005
                                 + 0.875504 w_k1_MAT_s005
                                 + 0.893369 w_k1_MSFT_s005
                                 + 1.05085 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.965637 w_k1_ADM_s005
                                 + 1.12287 w_k1_AXP_s005
                                 + 1.05698 w_k1_DIS_s005
                                 + 1.15313 w_k1_FDX_s005
                                 + 1.24477 w_k1_IBM_s005
                                 + 1.16593 w_k1_INTC_s005
                                 + 1.13055 w_k1_KO_s005 + 1.38121 w_k1_MAT_s005
                                 + 1.36254 w_k1_MSFT_s005
                                 + 1.09167 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.19628 w_k1_ADM_s005
                                 + 1.32765 w_k1_AXP_s005
                                 + 0.754815 w_k1_DIS_s005
                                 + 1.14016 w_k1_FDX_s005
                                 + 0.793638 w_k1_IBM_s005
                                 + 0.965015 w_k1_INTC_s005
                                 + 0.694764 w_k1_KO_s005
                                 + 0.744343 w_k1_MAT_s005
                                 + 1.0215 w_k1_MSFT_s005
                                 + 0.989322 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.68081 w_k1_ADM_s005
                                 + 1.56663 w_k1_AXP_s005
                                 + 1.36537 w_k1_DIS_s005
                                 + 1.58208 w_k1_FDX_s005
                                 + 1.17522 w_k1_IBM_s005
                                 + 1.18074 w_k1_INTC_s005
                                 + 1.18263 w_k1_KO_s005 + 1.12203 w_k1_MAT_s005
                                 + 1.19531 w_k1_MSFT_s005
                                 + 1.18107 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.21733 w_k1_ADM_s005
                                 + 1.07364 w_k1_AXP_s005
                                 + 0.890039 w_k1_DIS_s005
                                 + 1.09414 w_k1_FDX_s005
                                 + 1.01488 w_k1_IBM_s005
                                 + 0.860834 w_k1_INTC_s005
                                 + 0.899994 w_k1_KO_s005
                                 + 1.14469 w_k1_MAT_s005
                                 + 1.0096 w_k1_MSFT_s005 + 1.04482 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.58233 w_k1_ADM_s005
                                 + 1.61145 w_k1_AXP_s005
                                 + 1.75951 w_k1_DIS_s005
                                 + 1.59548 w_k1_FDX_s005
                                 + 1.74431 w_k1_IBM_s005
                                 + 1.40595 w_k1_INTC_s005
                                 + 1.60147 w_k1_KO_s005 + 1.42299 w_k1_MAT_s005
                                 + 1.51732 w_k1_MSFT_s005 + 1.2922 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.934108 w_k1_ADM_s006
                                 + 0.842732 w_k1_AXP_s006
                                 + 0.785118 w_k1_DIS_s006
                                 + 0.972147 w_k1_FDX_s006
                                 + 0.785732 w_k1_IBM_s006
                                 + 0.990572 w_k1_INTC_s006
                                 + 0.97287 w_k1_KO_s006
                                 + 0.701548 w_k1_MAT_s006
                                 + 0.975953 w_k1_MSFT_s006
                                 + 1.04707 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.820165 w_k1_ADM_s006
                                 + 0.677963 w_k1_AXP_s006
                                 + 0.890955 w_k1_DIS_s006
                                 + 0.836655 w_k1_FDX_s006
                                 + 0.914715 w_k1_IBM_s006
                                 + 0.889218 w_k1_INTC_s006
                                 + 1.0693 w_k1_KO_s006 + 1.08985 w_k1_MAT_s006
                                 + 0.984478 w_k1_MSFT_s006
                                 + 0.982287 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.913192 w_k1_ADM_s006
                                 + 0.874768 w_k1_AXP_s006
                                 + 1.06409 w_k1_DIS_s006 + 1.0796 w_k1_FDX_s006
                                 + 1.43009 w_k1_IBM_s006
                                 + 0.99039 w_k1_INTC_s006
                                 + 1.20542 w_k1_KO_s006 + 1.14572 w_k1_MAT_s006
                                 + 0.910002 w_k1_MSFT_s006
                                 + 1.10514 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.950042 w_k1_ADM_s006
                                 + 1.26732 w_k1_AXP_s006
                                 + 0.732102 w_k1_DIS_s006
                                 + 0.881748 w_k1_FDX_s006
                                 + 1.01852 w_k1_IBM_s006
                                 + 1.01165 w_k1_INTC_s006
                                 + 1.01937 w_k1_KO_s006 + 1.2616 w_k1_MAT_s006
                                 + 0.921267 w_k1_MSFT_s006
                                 + 1.09532 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.982409 w_k1_ADM_s006
                                 + 1.43691 w_k1_AXP_s006
                                 + 1.70197 w_k1_DIS_s006
                                 + 1.34971 w_k1_FDX_s006
                                 + 1.49064 w_k1_IBM_s006
                                 + 1.47709 w_k1_INTC_s006
                                 + 1.57741 w_k1_KO_s006 + 1.91707 w_k1_MAT_s006
                                 + 1.18063 w_k1_MSFT_s006
                                 + 1.41886 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.971441 w_k1_ADM_s006
                                 + 0.803846 w_k1_AXP_s006
                                 + 1.29137 w_k1_DIS_s006
                                 + 1.11349 w_k1_FDX_s006
                                 + 1.04747 w_k1_IBM_s006
                                 + 0.950781 w_k1_INTC_s006
                                 + 1.16364 w_k1_KO_s006 + 0.98443 w_k1_MAT_s006
                                 + 1.18734 w_k1_MSFT_s006
                                 + 1.14427 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.590745 w_k1_ADM_s006
                                 + 0.634905 w_k1_AXP_s006
                                 + 0.761609 w_k1_DIS_s006
                                 + 0.737501 w_k1_FDX_s006
                                 + 1.12251 w_k1_IBM_s006
                                 + 0.637728 w_k1_INTC_s006
                                 + 0.800323 w_k1_KO_s006
                                 + 0.938129 w_k1_MAT_s006
                                 + 0.745865 w_k1_MSFT_s006
                                 + 0.888607 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.13042 w_k1_ADM_s006
                                 + 1.24124 w_k1_AXP_s006
                                 + 1.07383 w_k1_DIS_s006 + 1.2087 w_k1_FDX_s006
                                 + 1.05241 w_k1_IBM_s006 + 1.079 w_k1_INTC_s006
                                 + 1.16967 w_k1_KO_s006 + 1.58122 w_k1_MAT_s006
                                 + 1.0271 w_k1_MSFT_s006 + 1.34181 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.735006 w_k1_ADM_s007
                                 + 0.706602 w_k1_AXP_s007
                                 + 0.724596 w_k1_DIS_s007
                                 + 0.816905 w_k1_FDX_s007
                                 + 0.882194 w_k1_IBM_s007
                                 + 0.929779 w_k1_INTC_s007
                                 + 0.849187 w_k1_KO_s007
                                 + 1.02551 w_k1_MAT_s007
                                 + 0.879836 w_k1_MSFT_s007
                                 + 0.905334 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.21282 w_k1_ADM_s007
                                 + 1.09604 w_k1_AXP_s007
                                 + 0.826468 w_k1_DIS_s007
                                 + 1.03573 w_k1_FDX_s007
                                 + 0.833968 w_k1_IBM_s007
                                 + 0.950425 w_k1_INTC_s007
                                 + 0.878536 w_k1_KO_s007
                                 + 0.985883 w_k1_MAT_s007
                                 + 1.02418 w_k1_MSFT_s007
                                 + 1.16379 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.0181 w_k1_ADM_s007
                                 + 0.870637 w_k1_AXP_s007
                                 + 1.04541 w_k1_DIS_s007
                                 + 0.999967 w_k1_FDX_s007
                                 + 0.777602 w_k1_IBM_s007
                                 + 0.793051 w_k1_INTC_s007
                                 + 1.01698 w_k1_KO_s007
                                 + 0.675353 w_k1_MAT_s007
                                 + 1.09848 w_k1_MSFT_s007
                                 + 0.973902 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.09871 w_k1_ADM_s007
                                 + 1.0113 w_k1_AXP_s007 + 1.01557 w_k1_DIS_s007
                                 + 0.925333 w_k1_FDX_s007
                                 + 1.1125 w_k1_IBM_s007
                                 + 1.02192 w_k1_INTC_s007
                                 + 1.02917 w_k1_KO_s007 + 1.0681 w_k1_MAT_s007
                                 + 0.945824 w_k1_MSFT_s007
                                 + 1.04483 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.557631 w_k1_ADM_s007
                                 + 1.15627 w_k1_AXP_s007
                                 + 0.811784 w_k1_DIS_s007
                                 + 0.858705 w_k1_FDX_s007
                                 + 1.42241 w_k1_IBM_s007
                                 + 0.953701 w_k1_INTC_s007
                                 + 1.01242 w_k1_KO_s007 + 1.8065 w_k1_MAT_s007
                                 + 0.980777 w_k1_MSFT_s007
                                 + 1.14354 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.871786 w_k1_ADM_s007
                                 + 0.743157 w_k1_AXP_s007
                                 + 0.785003 w_k1_DIS_s007
                                 + 0.883321 w_k1_FDX_s007
                                 + 0.635278 w_k1_IBM_s007
                                 + 0.722354 w_k1_INTC_s007
                                 + 0.723675 w_k1_KO_s007
                                 + 0.625398 w_k1_MAT_s007
                                 + 0.921882 w_k1_MSFT_s007
                                 + 0.957823 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.00483 w_k1_ADM_s007
                                 + 1.76003 w_k1_AXP_s007
                                 + 1.48299 w_k1_DIS_s007
                                 + 1.30014 w_k1_FDX_s007
                                 + 1.62522 w_k1_IBM_s007
                                 + 1.40919 w_k1_INTC_s007
                                 + 1.33413 w_k1_KO_s007 + 1.69345 w_k1_MAT_s007
                                 + 1.20849 w_k1_MSFT_s007
                                 + 1.42355 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.917368 w_k1_ADM_s007
                                 + 1.30629 w_k1_AXP_s007
                                 + 1.35719 w_k1_DIS_s007
                                 + 1.21556 w_k1_FDX_s007
                                 + 1.00363 w_k1_IBM_s007
                                 + 1.23866 w_k1_INTC_s007
                                 + 1.15247 w_k1_KO_s007 + 1.38137 w_k1_MAT_s007
                                 + 1.40956 w_k1_MSFT_s007
                                 + 1.34303 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.975286 w_k1_ADM_s010
                                 + 1.05487 w_k1_AXP_s010
                                 + 1.05908 w_k1_DIS_s010
                                 + 1.09486 w_k1_FDX_s010
                                 + 0.641817 w_k1_IBM_s010
                                 + 1.02106 w_k1_INTC_s010
                                 + 0.932472 w_k1_KO_s010
                                 + 0.745184 w_k1_MAT_s010
                                 + 1.00482 w_k1_MSFT_s010
                                 + 0.951334 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.09808 w_k1_ADM_s010
                                 + 1.02357 w_k1_AXP_s010
                                 + 1.09163 w_k1_DIS_s010
                                 + 1.10268 w_k1_FDX_s010
                                 + 0.964961 w_k1_IBM_s010
                                 + 1.17934 w_k1_INTC_s010
                                 + 1.07679 w_k1_KO_s010 + 1.09473 w_k1_MAT_s010
                                 + 1.08524 w_k1_MSFT_s010
                                 + 0.897059 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.982117 w_k1_ADM_s010
                                 + 0.815945 w_k1_AXP_s010
                                 + 0.875167 w_k1_DIS_s010
                                 + 0.890347 w_k1_FDX_s010
                                 + 0.848851 w_k1_IBM_s010
                                 + 0.977179 w_k1_INTC_s010
                                 + 1.01449 w_k1_KO_s010
                                 + 0.731332 w_k1_MAT_s010
                                 + 0.85247 w_k1_MSFT_s010
                                 + 0.965294 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.28481 w_k1_ADM_s010
                                 + 1.45027 w_k1_AXP_s010
                                 + 1.81537 w_k1_DIS_s010
                                 + 1.35608 w_k1_FDX_s010 + 1.3729 w_k1_IBM_s010
                                 + 1.51025 w_k1_INTC_s010
                                 + 1.38907 w_k1_KO_s010 + 1.4037 w_k1_MAT_s010
                                 + 1.31722 w_k1_MSFT_s010
                                 + 1.30692 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.0537 w_k1_ADM_s010
                                 + 1.06279 w_k1_AXP_s010
                                 + 1.17625 w_k1_DIS_s010
                                 + 0.919781 w_k1_FDX_s010
                                 + 1.45748 w_k1_IBM_s010
                                 + 1.26958 w_k1_INTC_s010
                                 + 1.30762 w_k1_KO_s010 + 1.45062 w_k1_MAT_s010
                                 + 0.957394 w_k1_MSFT_s010
                                 + 1.10384 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.728858 w_k1_ADM_s010
                                 + 0.667321 w_k1_AXP_s010
                                 + 0.745163 w_k1_DIS_s010
                                 + 0.808992 w_k1_FDX_s010
                                 + 0.732069 w_k1_IBM_s010
                                 + 0.781833 w_k1_INTC_s010
                                 + 0.808121 w_k1_KO_s010
                                 + 0.585614 w_k1_MAT_s010
                                 + 0.886598 w_k1_MSFT_s010
                                 + 0.762246 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.980415 w_k1_ADM_s010
                                 + 0.879822 w_k1_AXP_s010
                                 + 0.516396 w_k1_DIS_s010
                                 + 0.772561 w_k1_FDX_s010
                                 + 0.591068 w_k1_IBM_s010
                                 + 0.908545 w_k1_INTC_s010
                                 + 0.662814 w_k1_KO_s010
                                 + 0.636815 w_k1_MAT_s010
                                 + 0.724882 w_k1_MSFT_s010
                                 + 0.731345 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.001 w_k1_ADM_s010
                                 + 1.62844 w_k1_AXP_s010
                                 + 1.03574 w_k1_DIS_s010
                                 + 1.15833 w_k1_FDX_s010
                                 + 1.27617 w_k1_IBM_s010
                                 + 1.14908 w_k1_INTC_s010
                                 + 1.11963 w_k1_KO_s010 + 1.64368 w_k1_MAT_s010
                                 + 1.03718 w_k1_MSFT_s010
                                 + 1.03368 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.29257 w_k1_ADM_s011
                                 + 0.903666 w_k1_AXP_s011
                                 + 0.807696 w_k1_DIS_s011
                                 + 1.00933 w_k1_FDX_s011
                                 + 0.630977 w_k1_IBM_s011
                                 + 1.34205 w_k1_INTC_s011
                                 + 0.77253 w_k1_KO_s011
                                 + 0.487492 w_k1_MAT_s011
                                 + 1.08511 w_k1_MSFT_s011
                                 + 0.895563 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.902738 w_k1_ADM_s011
                                 + 0.73946 w_k1_AXP_s011
                                 + 0.915024 w_k1_DIS_s011
                                 + 0.912507 w_k1_FDX_s011
                                 + 0.925444 w_k1_IBM_s011
                                 + 0.857494 w_k1_INTC_s011
                                 + 0.941119 w_k1_KO_s011
                                 + 0.919629 w_k1_MAT_s011
                                 + 0.896102 w_k1_MSFT_s011
                                 + 0.950645 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.818102 w_k1_ADM_s011
                                 + 0.752148 w_k1_AXP_s011
                                 + 0.726937 w_k1_DIS_s011
                                 + 0.81691 w_k1_FDX_s011
                                 + 0.701183 w_k1_IBM_s011
                                 + 0.886923 w_k1_INTC_s011
                                 + 0.792316 w_k1_KO_s011
                                 + 0.72071 w_k1_MAT_s011
                                 + 0.854634 w_k1_MSFT_s011
                                 + 0.792648 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.23788 w_k1_ADM_s011
                                 + 1.07969 w_k1_AXP_s011
                                 + 1.35736 w_k1_DIS_s011
                                 + 1.25108 w_k1_FDX_s011
                                 + 0.928955 w_k1_IBM_s011
                                 + 1.43637 w_k1_INTC_s011
                                 + 1.17482 w_k1_KO_s011
                                 + 0.829319 w_k1_MAT_s011
                                 + 1.21889 w_k1_MSFT_s011
                                 + 1.18244 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.734896 w_k1_ADM_s011
                                 + 0.570074 w_k1_AXP_s011
                                 + 0.491205 w_k1_DIS_s011
                                 + 0.676025 w_k1_FDX_s011
                                 + 0.6463 w_k1_IBM_s011
                                 + 0.793189 w_k1_INTC_s011
                                 + 0.638122 w_k1_KO_s011
                                 + 0.621299 w_k1_MAT_s011
                                 + 0.740484 w_k1_MSFT_s011
                                 + 0.851438 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.806526 w_k1_ADM_s011
                                 + 0.753381 w_k1_AXP_s011
                                 + 0.620358 w_k1_DIS_s011
                                 + 0.87817 w_k1_FDX_s011
                                 + 0.774336 w_k1_IBM_s011
                                 + 0.971224 w_k1_INTC_s011
                                 + 0.786567 w_k1_KO_s011
                                 + 0.872744 w_k1_MAT_s011
                                 + 0.811709 w_k1_MSFT_s011
                                 + 0.986255 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.2477 w_k1_ADM_s011
                                 + 1.15726 w_k1_AXP_s011
                                 + 1.22909 w_k1_DIS_s011
                                 + 1.14166 w_k1_FDX_s011
                                 + 1.33595 w_k1_IBM_s011
                                 + 1.21188 w_k1_INTC_s011
                                 + 1.17951 w_k1_KO_s011 + 1.52444 w_k1_MAT_s011
                                 + 1.09719 w_k1_MSFT_s011
                                 + 1.16938 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.28934 w_k1_ADM_s011
                                 + 1.48278 w_k1_AXP_s011 + 1.8996 w_k1_DIS_s011
                                 + 1.28341 w_k1_FDX_s011
                                 + 1.82285 w_k1_IBM_s011
                                 + 1.60216 w_k1_INTC_s011
                                 + 1.69719 w_k1_KO_s011 + 1.91992 w_k1_MAT_s011
                                 + 1.43302 w_k1_MSFT_s011 + 1.4049 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.98371 w_k1_ADM_s012
                                 + 0.929849 w_k1_AXP_s012
                                 + 1.0291 w_k1_DIS_s012
                                 + 0.974974 w_k1_FDX_s012
                                 + 0.82578 w_k1_IBM_s012
                                 + 1.05797 w_k1_INTC_s012
                                 + 1.09447 w_k1_KO_s012
                                 + 0.700865 w_k1_MAT_s012
                                 + 0.972934 w_k1_MSFT_s012
                                 + 0.954721 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.26056 w_k1_ADM_s012
                                 + 0.857526 w_k1_AXP_s012
                                 + 1.11423 w_k1_DIS_s012
                                 + 1.11166 w_k1_FDX_s012
                                 + 0.624751 w_k1_IBM_s012
                                 + 1.0252 w_k1_INTC_s012 + 1.0575 w_k1_KO_s012
                                 + 0.629555 w_k1_MAT_s012
                                 + 1.15257 w_k1_MSFT_s012
                                 + 0.941248 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.10093 w_k1_ADM_s012
                                 + 0.90775 w_k1_AXP_s012
                                 + 0.824072 w_k1_DIS_s012
                                 + 0.898963 w_k1_FDX_s012
                                 + 0.863275 w_k1_IBM_s012
                                 + 0.924119 w_k1_INTC_s012
                                 + 0.981601 w_k1_KO_s012
                                 + 1.19325 w_k1_MAT_s012
                                 + 0.892367 w_k1_MSFT_s012
                                 + 0.904499 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.58136 w_k1_ADM_s012
                                 + 1.46887 w_k1_AXP_s012
                                 + 1.06593 w_k1_DIS_s012
                                 + 1.21673 w_k1_FDX_s012
                                 + 0.846767 w_k1_IBM_s012
                                 + 0.844475 w_k1_INTC_s012
                                 + 0.982139 w_k1_KO_s012
                                 + 1.19559 w_k1_MAT_s012
                                 + 1.04166 w_k1_MSFT_s012
                                 + 0.958881 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.34056 w_k1_ADM_s012
                                 + 0.776735 w_k1_AXP_s012
                                 + 1.03408 w_k1_DIS_s012
                                 + 0.851575 w_k1_FDX_s012
                                 + 1.00818 w_k1_IBM_s012
                                 + 0.932565 w_k1_INTC_s012
                                 + 1.18818 w_k1_KO_s012 + 1.04077 w_k1_MAT_s012
                                 + 0.906382 w_k1_MSFT_s012
                                 + 0.948264 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.954253 w_k1_ADM_s012
                                 + 0.809564 w_k1_AXP_s012
                                 + 0.595831 w_k1_DIS_s012
                                 + 0.72668 w_k1_FDX_s012
                                 + 0.916582 w_k1_IBM_s012
                                 + 0.779396 w_k1_INTC_s012
                                 + 0.836463 w_k1_KO_s012
                                 + 0.83439 w_k1_MAT_s012
                                 + 0.717155 w_k1_MSFT_s012
                                 + 0.74533 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.18082 w_k1_ADM_s012
                                 + 1.30699 w_k1_AXP_s012
                                 + 0.952111 w_k1_DIS_s012
                                 + 0.850515 w_k1_FDX_s012
                                 + 1.5418 w_k1_IBM_s012
                                 + 0.910199 w_k1_INTC_s012
                                 + 1.25406 w_k1_KO_s012 + 1.62543 w_k1_MAT_s012
                                 + 0.825218 w_k1_MSFT_s012
                                 + 0.989738 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.59394 w_k1_ADM_s012
                                 + 1.29136 w_k1_AXP_s012 + 1.3467 w_k1_DIS_s012
                                 + 1.17675 w_k1_FDX_s012
                                 + 1.24586 w_k1_IBM_s012
                                 + 1.0135 w_k1_INTC_s012 + 1.38119 w_k1_KO_s012
                                 + 1.08554 w_k1_MAT_s012
                                 + 1.21577 w_k1_MSFT_s012
                                 + 0.935716 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.1875 w_k1_ADM_s013
                                 + 0.915974 w_k1_AXP_s013
                                 + 1.12401 w_k1_DIS_s013 + 1.0299 w_k1_FDX_s013
                                 + 0.838922 w_k1_IBM_s013
                                 + 0.925458 w_k1_INTC_s013
                                 + 0.945251 w_k1_KO_s013
                                 + 0.678554 w_k1_MAT_s013
                                 + 1.07863 w_k1_MSFT_s013
                                 + 1.00716 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.871721 w_k1_ADM_s013
                                 + 1.0907 w_k1_AXP_s013 + 1.00012 w_k1_DIS_s013
                                 + 0.969813 w_k1_FDX_s013
                                 + 1.20496 w_k1_IBM_s013
                                 + 0.68698 w_k1_INTC_s013
                                 + 0.908782 w_k1_KO_s013
                                 + 1.48068 w_k1_MAT_s013
                                 + 0.805033 w_k1_MSFT_s013
                                 + 0.94308 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.956472 w_k1_ADM_s013
                                 + 0.892354 w_k1_AXP_s013
                                 + 0.870356 w_k1_DIS_s013
                                 + 0.918846 w_k1_FDX_s013
                                 + 0.967519 w_k1_IBM_s013
                                 + 0.759611 w_k1_INTC_s013
                                 + 0.84148 w_k1_KO_s013 + 1.14618 w_k1_MAT_s013
                                 + 0.892211 w_k1_MSFT_s013
                                 + 1.00563 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.00179 w_k1_ADM_s013
                                 + 1.36045 w_k1_AXP_s013
                                 + 1.19022 w_k1_DIS_s013
                                 + 1.16977 w_k1_FDX_s013
                                 + 1.37317 w_k1_IBM_s013
                                 + 1.21972 w_k1_INTC_s013
                                 + 1.16598 w_k1_KO_s013 + 1.55013 w_k1_MAT_s013
                                 + 1.18525 w_k1_MSFT_s013
                                 + 1.09347 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.07086 w_k1_ADM_s013
                                 + 0.880444 w_k1_AXP_s013
                                 + 0.924208 w_k1_DIS_s013
                                 + 0.944165 w_k1_FDX_s013
                                 + 0.646275 w_k1_IBM_s013
                                 + 0.803143 w_k1_INTC_s013
                                 + 0.779499 w_k1_KO_s013
                                 + 0.89108 w_k1_MAT_s013
                                 + 0.909446 w_k1_MSFT_s013
                                 + 0.891026 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.76878 w_k1_ADM_s013
                                 + 0.771271 w_k1_AXP_s013
                                 + 0.672156 w_k1_DIS_s013
                                 + 0.764931 w_k1_FDX_s013
                                 + 0.880311 w_k1_IBM_s013
                                 + 0.681447 w_k1_INTC_s013
                                 + 0.661677 w_k1_KO_s013
                                 + 0.796006 w_k1_MAT_s013
                                 + 0.654245 w_k1_MSFT_s013
                                 + 0.757917 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.770993 w_k1_ADM_s013
                                 + 0.764885 w_k1_AXP_s013
                                 + 0.643636 w_k1_DIS_s013
                                 + 0.732081 w_k1_FDX_s013
                                 + 0.593221 w_k1_IBM_s013
                                 + 0.652526 w_k1_INTC_s013
                                 + 0.561789 w_k1_KO_s013
                                 + 0.736276 w_k1_MAT_s013
                                 + 0.750431 w_k1_MSFT_s013
                                 + 0.812882 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.41023 w_k1_ADM_s013
                                 + 1.35258 w_k1_AXP_s013
                                 + 1.92322 w_k1_DIS_s013
                                 + 1.38906 w_k1_FDX_s013
                                 + 1.21182 w_k1_IBM_s013
                                 + 1.00913 w_k1_INTC_s013
                                 + 1.23587 w_k1_KO_s013 + 1.19957 w_k1_MAT_s013
                                 + 1.12982 w_k1_MSFT_s013
                                 + 1.12366 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.926836 w_k1_ADM_s014
                                 + 1.03578 w_k1_AXP_s014
                                 + 0.926335 w_k1_DIS_s014
                                 + 1.13977 w_k1_FDX_s014
                                 + 0.772815 w_k1_IBM_s014
                                 + 0.997211 w_k1_INTC_s014
                                 + 0.830499 w_k1_KO_s014
                                 + 1.05897 w_k1_MAT_s014
                                 + 1.03676 w_k1_MSFT_s014
                                 + 0.997808 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.06274 w_k1_ADM_s014
                                 + 1.22046 w_k1_AXP_s014
                                 + 1.40629 w_k1_DIS_s014
                                 + 1.20835 w_k1_FDX_s014 + 1.1362 w_k1_IBM_s014
                                 + 1.12467 w_k1_INTC_s014
                                 + 1.13694 w_k1_KO_s014 + 1.90784 w_k1_MAT_s014
                                 + 1.32055 w_k1_MSFT_s014 + 1.0952 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.770748 w_k1_ADM_s014
                                 + 0.790036 w_k1_AXP_s014
                                 + 0.85062 w_k1_DIS_s014
                                 + 0.871617 w_k1_FDX_s014
                                 + 1.04031 w_k1_IBM_s014
                                 + 0.964921 w_k1_INTC_s014
                                 + 0.935568 w_k1_KO_s014
                                 + 1.33315 w_k1_MAT_s014
                                 + 0.975596 w_k1_MSFT_s014
                                 + 0.992969 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.942129 w_k1_ADM_s014
                                 + 0.632794 w_k1_AXP_s014
                                 + 0.826297 w_k1_DIS_s014
                                 + 0.841737 w_k1_FDX_s014
                                 + 0.650894 w_k1_IBM_s014
                                 + 0.763337 w_k1_INTC_s014
                                 + 0.826879 w_k1_KO_s014
                                 + 0.694691 w_k1_MAT_s014
                                 + 0.898719 w_k1_MSFT_s014
                                 + 0.948201 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.12476 w_k1_ADM_s014
                                 + 1.23787 w_k1_AXP_s014
                                 + 1.65488 w_k1_DIS_s014
                                 + 1.25082 w_k1_FDX_s014 + 1.5936 w_k1_IBM_s014
                                 + 1.03722 w_k1_INTC_s014
                                 + 1.36854 w_k1_KO_s014 + 1.78025 w_k1_MAT_s014
                                 + 1.12416 w_k1_MSFT_s014 + 1.2271 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.874884 w_k1_ADM_s014
                                 + 0.969894 w_k1_AXP_s014
                                 + 1.07589 w_k1_DIS_s014
                                 + 0.954862 w_k1_FDX_s014
                                 + 1.3698 w_k1_IBM_s014
                                 + 0.916551 w_k1_INTC_s014
                                 + 1.2074 w_k1_KO_s014 + 1.46169 w_k1_MAT_s014
                                 + 1.05571 w_k1_MSFT_s014
                                 + 0.946032 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.723222 w_k1_ADM_s014
                                 + 0.784087 w_k1_AXP_s014
                                 + 1.35702 w_k1_DIS_s014
                                 + 1.04335 w_k1_FDX_s014
                                 + 0.885101 w_k1_IBM_s014
                                 + 1.02147 w_k1_INTC_s014
                                 + 1.12696 w_k1_KO_s014 + 1.32494 w_k1_MAT_s014
                                 + 1.19224 w_k1_MSFT_s014
                                 + 1.07871 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.43028 w_k1_ADM_s014
                                 + 1.28078 w_k1_AXP_s014
                                 + 1.44264 w_k1_DIS_s014
                                 + 1.33769 w_k1_FDX_s014
                                 + 1.03018 w_k1_IBM_s014
                                 + 1.17064 w_k1_INTC_s014
                                 + 1.25972 w_k1_KO_s014 + 1.37378 w_k1_MAT_s014
                                 + 1.23764 w_k1_MSFT_s014
                                 + 1.25659 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.00953 w_k1_ADM_s015
                                 + 0.90982 w_k1_AXP_s015
                                 + 1.22896 w_k1_DIS_s015
                                 + 1.14526 w_k1_FDX_s015
                                 + 0.919963 w_k1_IBM_s015
                                 + 1.22198 w_k1_INTC_s015
                                 + 1.01672 w_k1_KO_s015
                                 + 0.579463 w_k1_MAT_s015
                                 + 0.975593 w_k1_MSFT_s015
                                 + 1.13041 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.13335 w_k1_ADM_s015
                                 + 1.02768 w_k1_AXP_s015
                                 + 1.26846 w_k1_DIS_s015
                                 + 0.935801 w_k1_FDX_s015
                                 + 1.27615 w_k1_IBM_s015
                                 + 1.05963 w_k1_INTC_s015
                                 + 1.25933 w_k1_KO_s015
                                 + 0.986436 w_k1_MAT_s015
                                 + 0.872233 w_k1_MSFT_s015
                                 + 1.03353 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.836706 w_k1_ADM_s015
                                 + 1.28359 w_k1_AXP_s015
                                 + 0.931612 w_k1_DIS_s015
                                 + 0.914235 w_k1_FDX_s015
                                 + 1.43948 w_k1_IBM_s015
                                 + 0.806872 w_k1_INTC_s015
                                 + 1.09736 w_k1_KO_s015 + 1.66324 w_k1_MAT_s015
                                 + 0.821432 w_k1_MSFT_s015
                                 + 1.09975 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.0073 w_k1_ADM_s015
                                 + 1.16353 w_k1_AXP_s015
                                 + 0.807422 w_k1_DIS_s015
                                 + 1.00919 w_k1_FDX_s015
                                 + 0.940929 w_k1_IBM_s015
                                 + 0.957013 w_k1_INTC_s015
                                 + 0.876435 w_k1_KO_s015
                                 + 0.928446 w_k1_MAT_s015
                                 + 0.821919 w_k1_MSFT_s015
                                 + 1.02398 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.57777 w_k1_ADM_s015
                                 + 1.25381 w_k1_AXP_s015
                                 + 1.47334 w_k1_DIS_s015
                                 + 1.36118 w_k1_FDX_s015
                                 + 1.07075 w_k1_IBM_s015
                                 + 0.923136 w_k1_INTC_s015
                                 + 1.16708 w_k1_KO_s015 + 1.15194 w_k1_MAT_s015
                                 + 1.1653 w_k1_MSFT_s015 + 1.11524 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.34699 w_k1_ADM_s015
                                 + 1.08682 w_k1_AXP_s015
                                 + 0.793365 w_k1_DIS_s015
                                 + 0.979726 w_k1_FDX_s015
                                 + 0.556198 w_k1_IBM_s015
                                 + 0.803202 w_k1_INTC_s015
                                 + 0.769878 w_k1_KO_s015
                                 + 0.899357 w_k1_MAT_s015
                                 + 0.853685 w_k1_MSFT_s015
                                 + 0.929295 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.88132 w_k1_ADM_s015
                                 + 0.779923 w_k1_AXP_s015
                                 + 0.767472 w_k1_DIS_s015
                                 + 0.761331 w_k1_FDX_s015
                                 + 1.24444 w_k1_IBM_s015
                                 + 0.759844 w_k1_INTC_s015
                                 + 1.03907 w_k1_KO_s015 + 1.37059 w_k1_MAT_s015
                                 + 0.746568 w_k1_MSFT_s015
                                 + 0.991185 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.89988 w_k1_ADM_s015
                                 + 0.831473 w_k1_AXP_s015
                                 + 0.751743 w_k1_DIS_s015
                                 + 0.80978 w_k1_FDX_s015
                                 + 0.726252 w_k1_IBM_s015
                                 + 0.82179 w_k1_INTC_s015
                                 + 0.868475 w_k1_KO_s015
                                 + 0.792051 w_k1_MAT_s015
                                 + 0.792912 w_k1_MSFT_s015
                                 + 0.870645 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.11314 w_k1_ADM_s016
                                 + 1.2518 w_k1_AXP_s016 + 1.03244 w_k1_DIS_s016
                                 + 1.00615 w_k1_FDX_s016
                                 + 1.15122 w_k1_IBM_s016
                                 + 0.903006 w_k1_INTC_s016
                                 + 0.889008 w_k1_KO_s016
                                 + 1.17988 w_k1_MAT_s016
                                 + 0.955506 w_k1_MSFT_s016
                                 + 0.939076 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.867044 w_k1_ADM_s016
                                 + 0.681282 w_k1_AXP_s016
                                 + 0.720556 w_k1_DIS_s016
                                 + 0.904657 w_k1_FDX_s016
                                 + 0.549962 w_k1_IBM_s016
                                 + 0.956228 w_k1_INTC_s016
                                 + 0.653736 w_k1_KO_s016
                                 + 0.637538 w_k1_MAT_s016
                                 + 0.897191 w_k1_MSFT_s016
                                 + 0.90486 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.1366 w_k1_ADM_s016
                                 + 1.28389 w_k1_AXP_s016
                                 + 1.60855 w_k1_DIS_s016
                                 + 1.44887 w_k1_FDX_s016
                                 + 1.11405 w_k1_IBM_s016
                                 + 1.49775 w_k1_INTC_s016
                                 + 1.11752 w_k1_KO_s016 + 1.09936 w_k1_MAT_s016
                                 + 1.36752 w_k1_MSFT_s016
                                 + 1.23765 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.10165 w_k1_ADM_s016
                                 + 0.987807 w_k1_AXP_s016
                                 + 1.13509 w_k1_DIS_s016
                                 + 1.15046 w_k1_FDX_s016
                                 + 0.972445 w_k1_IBM_s016
                                 + 1.21311 w_k1_INTC_s016
                                 + 1.09401 w_k1_KO_s016 + 1.05396 w_k1_MAT_s016
                                 + 1.18141 w_k1_MSFT_s016
                                 + 1.11864 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.864112 w_k1_ADM_s016
                                 + 0.522424 w_k1_AXP_s016
                                 + 0.712597 w_k1_DIS_s016
                                 + 0.74201 w_k1_FDX_s016
                                 + 1.04822 w_k1_IBM_s016
                                 + 0.812558 w_k1_INTC_s016
                                 + 0.882295 w_k1_KO_s016
                                 + 0.879407 w_k1_MAT_s016
                                 + 0.776795 w_k1_MSFT_s016
                                 + 0.918531 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.968067 w_k1_ADM_s016
                                 + 0.850505 w_k1_AXP_s016
                                 + 0.900011 w_k1_DIS_s016
                                 + 0.896092 w_k1_FDX_s016
                                 + 0.856138 w_k1_IBM_s016
                                 + 1.09809 w_k1_INTC_s016
                                 + 0.87555 w_k1_KO_s016 + 1.01068 w_k1_MAT_s016
                                 + 0.992974 w_k1_MSFT_s016
                                 + 0.928864 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.757739 w_k1_ADM_s016
                                 + 0.853127 w_k1_AXP_s016
                                 + 0.683171 w_k1_DIS_s016
                                 + 0.763095 w_k1_FDX_s016
                                 + 1.05269 w_k1_IBM_s016
                                 + 0.840347 w_k1_INTC_s016
                                 + 0.885494 w_k1_KO_s016
                                 + 1.35675 w_k1_MAT_s016
                                 + 0.808656 w_k1_MSFT_s016
                                 + 0.895764 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.41819 w_k1_ADM_s016
                                 + 1.53928 w_k1_AXP_s016
                                 + 1.22817 w_k1_DIS_s016
                                 + 1.24662 w_k1_FDX_s016
                                 + 1.56728 w_k1_IBM_s016
                                 + 1.22751 w_k1_INTC_s016
                                 + 1.19801 w_k1_KO_s016 + 1.80595 w_k1_MAT_s016
                                 + 1.02897 w_k1_MSFT_s016
                                 + 1.24097 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.16671 w_k1_ADM_s017
                                 + 0.914393 w_k1_AXP_s017
                                 + 0.850869 w_k1_DIS_s017
                                 + 1.00167 w_k1_FDX_s017
                                 + 0.596274 w_k1_IBM_s017
                                 + 1.11937 w_k1_INTC_s017
                                 + 0.764237 w_k1_KO_s017
                                 + 0.418705 w_k1_MAT_s017
                                 + 0.960798 w_k1_MSFT_s017
                                 + 0.900814 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.53272 w_k1_ADM_s017
                                 + 1.5214 w_k1_AXP_s017 + 1.25682 w_k1_DIS_s017
                                 + 1.24783 w_k1_FDX_s017
                                 + 1.17238 w_k1_IBM_s017
                                 + 1.23953 w_k1_INTC_s017
                                 + 1.10411 w_k1_KO_s017 + 1.15535 w_k1_MAT_s017
                                 + 1.06076 w_k1_MSFT_s017
                                 + 1.11714 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.01368 w_k1_ADM_s017
                                 + 1.04247 w_k1_AXP_s017
                                 + 1.32525 w_k1_DIS_s017
                                 + 1.00745 w_k1_FDX_s017
                                 + 0.98754 w_k1_IBM_s017
                                 + 1.01883 w_k1_INTC_s017
                                 + 1.16082 w_k1_KO_s017 + 1.06787 w_k1_MAT_s017
                                 + 1.05723 w_k1_MSFT_s017
                                 + 1.10404 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.33772 w_k1_ADM_s017
                                 + 1.38904 w_k1_AXP_s017
                                 + 1.52598 w_k1_DIS_s017
                                 + 1.11016 w_k1_FDX_s017 + 1.5936 w_k1_IBM_s017
                                 + 1.33744 w_k1_INTC_s017
                                 + 1.53036 w_k1_KO_s017 + 1.42167 w_k1_MAT_s017
                                 + 1.09939 w_k1_MSFT_s017
                                 + 1.16203 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.884008 w_k1_ADM_s017
                                 + 0.922737 w_k1_AXP_s017
                                 + 0.947095 w_k1_DIS_s017
                                 + 0.841125 w_k1_FDX_s017
                                 + 1.41875 w_k1_IBM_s017
                                 + 0.962748 w_k1_INTC_s017
                                 + 1.11515 w_k1_KO_s017 + 1.1757 w_k1_MAT_s017
                                 + 0.823172 w_k1_MSFT_s017
                                 + 0.920561 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.987191 w_k1_ADM_s017
                                 + 0.904265 w_k1_AXP_s017
                                 + 0.971034 w_k1_DIS_s017
                                 + 0.91387 w_k1_FDX_s017
                                 + 1.03789 w_k1_IBM_s017
                                 + 1.10197 w_k1_INTC_s017
                                 + 1.02948 w_k1_KO_s017 + 0.93877 w_k1_MAT_s017
                                 + 0.941124 w_k1_MSFT_s017
                                 + 0.911335 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.31724 w_k1_ADM_s017
                                 + 1.34624 w_k1_AXP_s017
                                 + 1.59503 w_k1_DIS_s017
                                 + 1.32589 w_k1_FDX_s017
                                 + 0.898784 w_k1_IBM_s017
                                 + 1.27152 w_k1_INTC_s017
                                 + 1.23666 w_k1_KO_s017
                                 + 0.939869 w_k1_MAT_s017
                                 + 1.30586 w_k1_MSFT_s017
                                 + 1.20793 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.767763 w_k1_ADM_s017
                                 + 0.66962 w_k1_AXP_s017
                                 + 1.02514 w_k1_DIS_s017
                                 + 0.82821 w_k1_FDX_s017
                                 + 1.24387 w_k1_IBM_s017
                                 + 0.817148 w_k1_INTC_s017
                                 + 1.13029 w_k1_KO_s017
                                 + 0.902375 w_k1_MAT_s017
                                 + 0.833306 w_k1_MSFT_s017
                                 + 1.00551 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.657321 w_k1_ADM_s100
                                 + 0.673004 w_k1_AXP_s100
                                 + 0.733026 w_k1_DIS_s100
                                 + 0.709231 w_k1_FDX_s100
                                 + 0.842167 w_k1_IBM_s100
                                 + 0.865326 w_k1_INTC_s100
                                 + 0.777999 w_k1_KO_s100
                                 + 0.827824 w_k1_MAT_s100
                                 + 0.89879 w_k1_MSFT_s100
                                 + 0.862576 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.07784 w_k1_ADM_s100
                                 + 0.689161 w_k1_AXP_s100
                                 + 0.811229 w_k1_DIS_s100
                                 + 0.872268 w_k1_FDX_s100
                                 + 0.799377 w_k1_IBM_s100
                                 + 0.737904 w_k1_INTC_s100
                                 + 0.730999 w_k1_KO_s100
                                 + 0.561706 w_k1_MAT_s100
                                 + 0.94326 w_k1_MSFT_s100
                                 + 0.842593 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.55025 w_k1_ADM_s100
                                 + 1.11665 w_k1_AXP_s100
                                 + 1.29953 w_k1_DIS_s100
                                 + 1.27912 w_k1_FDX_s100 + 0.8273 w_k1_IBM_s100
                                 + 0.914498 w_k1_INTC_s100
                                 + 0.86507 w_k1_KO_s100
                                 + 0.746631 w_k1_MAT_s100
                                 + 1.03214 w_k1_MSFT_s100
                                 + 1.01542 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.25751 w_k1_ADM_s100
                                 + 1.11837 w_k1_AXP_s100
                                 + 0.891091 w_k1_DIS_s100
                                 + 1.03821 w_k1_FDX_s100
                                 + 0.755281 w_k1_IBM_s100
                                 + 1.05384 w_k1_INTC_s100
                                 + 0.886662 w_k1_KO_s100
                                 + 0.857782 w_k1_MAT_s100
                                 + 0.977531 w_k1_MSFT_s100
                                 + 1.01812 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.14969 w_k1_ADM_s100
                                 + 1.08455 w_k1_AXP_s100
                                 + 1.15224 w_k1_DIS_s100
                                 + 1.10692 w_k1_FDX_s100
                                 + 1.11944 w_k1_IBM_s100
                                 + 0.944579 w_k1_INTC_s100
                                 + 1.03742 w_k1_KO_s100 + 1.11316 w_k1_MAT_s100
                                 + 0.979999 w_k1_MSFT_s100
                                 + 1.07074 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.31873 w_k1_ADM_s100
                                 + 1.49533 w_k1_AXP_s100 + 1.6422 w_k1_DIS_s100
                                 + 1.39668 w_k1_FDX_s100
                                 + 1.29414 w_k1_IBM_s100
                                 + 1.1504 w_k1_INTC_s100 + 1.32291 w_k1_KO_s100
                                 + 1.29345 w_k1_MAT_s100
                                 + 1.1536 w_k1_MSFT_s100 + 1.19148 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.676338 w_k1_ADM_s100
                                 + 0.921581 w_k1_AXP_s100
                                 + 0.844158 w_k1_DIS_s100
                                 + 0.79755 w_k1_FDX_s100
                                 + 1.31184 w_k1_IBM_s100
                                 + 0.893173 w_k1_INTC_s100
                                 + 0.951825 w_k1_KO_s100
                                 + 1.40905 w_k1_MAT_s100
                                 + 0.864732 w_k1_MSFT_s100
                                 + 0.926142 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.806868 w_k1_ADM_s100
                                 + 1.05916 w_k1_AXP_s100
                                 + 1.14693 w_k1_DIS_s100
                                 + 0.996563 w_k1_FDX_s100
                                 + 1.87153 w_k1_IBM_s100
                                 + 1.04378 w_k1_INTC_s100
                                 + 1.21481 w_k1_KO_s100 + 1.59459 w_k1_MAT_s100
                                 + 1.01766 w_k1_MSFT_s100
                                 + 1.09376 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.818886 w_k1_ADM_s101
                                 + 1.05443 w_k1_AXP_s101
                                 + 0.913297 w_k1_DIS_s101
                                 + 0.916502 w_k1_FDX_s101
                                 + 1.30476 w_k1_IBM_s101
                                 + 1.02472 w_k1_INTC_s101
                                 + 0.997261 w_k1_KO_s101
                                 + 1.15792 w_k1_MAT_s101
                                 + 0.850275 w_k1_MSFT_s101
                                 + 0.991387 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.849534 w_k1_ADM_s101
                                 + 0.70861 w_k1_AXP_s101
                                 + 0.763678 w_k1_DIS_s101
                                 + 0.849238 w_k1_FDX_s101
                                 + 0.868524 w_k1_IBM_s101
                                 + 1.00911 w_k1_INTC_s101
                                 + 0.845259 w_k1_KO_s101
                                 + 0.696325 w_k1_MAT_s101
                                 + 0.819992 w_k1_MSFT_s101
                                 + 0.977415 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.861131 w_k1_ADM_s101
                                 + 0.949112 w_k1_AXP_s101
                                 + 1.09248 w_k1_DIS_s101
                                 + 1.14359 w_k1_FDX_s101
                                 + 0.720449 w_k1_IBM_s101
                                 + 1.19572 w_k1_INTC_s101
                                 + 0.972323 w_k1_KO_s101
                                 + 0.757998 w_k1_MAT_s101
                                 + 1.07337 w_k1_MSFT_s101 + 1.0955 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.626325 w_k1_ADM_s101
                                 + 0.721951 w_k1_AXP_s101
                                 + 0.628133 w_k1_DIS_s101
                                 + 0.696377 w_k1_FDX_s101
                                 + 1.1605 w_k1_IBM_s101
                                 + 0.810287 w_k1_INTC_s101
                                 + 0.819162 w_k1_KO_s101
                                 + 1.04491 w_k1_MAT_s101
                                 + 0.652113 w_k1_MSFT_s101
                                 + 0.84515 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.26063 w_k1_ADM_s101
                                 + 1.39193 w_k1_AXP_s101
                                 + 1.65546 w_k1_DIS_s101
                                 + 1.37909 w_k1_FDX_s101 + 1.2017 w_k1_IBM_s101
                                 + 1.17957 w_k1_INTC_s101
                                 + 1.36753 w_k1_KO_s101 + 1.18956 w_k1_MAT_s101
                                 + 1.137 w_k1_MSFT_s101 + 1.11887 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.800164 w_k1_ADM_s101
                                 + 0.712155 w_k1_AXP_s101
                                 + 0.591578 w_k1_DIS_s101
                                 + 0.796231 w_k1_FDX_s101
                                 + 0.612619 w_k1_IBM_s101
                                 + 0.68581 w_k1_INTC_s101
                                 + 0.606465 w_k1_KO_s101
                                 + 0.543084 w_k1_MAT_s101
                                 + 0.65079 w_k1_MSFT_s101
                                 + 0.794291 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.20977 w_k1_ADM_s101
                                 + 1.25642 w_k1_AXP_s101
                                 + 1.48835 w_k1_DIS_s101
                                 + 1.14103 w_k1_FDX_s101
                                 + 1.59906 w_k1_IBM_s101
                                 + 1.02271 w_k1_INTC_s101
                                 + 1.64882 w_k1_KO_s101 + 1.93634 w_k1_MAT_s101
                                 + 1.17849 w_k1_MSFT_s101
                                 + 1.27218 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.907019 w_k1_ADM_s101
                                 + 1.05775 w_k1_AXP_s101
                                 + 1.06419 w_k1_DIS_s101
                                 + 1.10445 w_k1_FDX_s101
                                 + 0.906432 w_k1_IBM_s101
                                 + 1.11393 w_k1_INTC_s101
                                 + 1.06257 w_k1_KO_s101 + 1.18043 w_k1_MAT_s101
                                 + 1.09971 w_k1_MSFT_s101
                                 + 1.04022 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.02603 w_k1_ADM_s102
                                 + 0.87793 w_k1_AXP_s102
                                 + 0.987796 w_k1_DIS_s102
                                 + 1.05672 w_k1_FDX_s102
                                 + 0.824723 w_k1_IBM_s102
                                 + 0.822125 w_k1_INTC_s102
                                 + 0.892781 w_k1_KO_s102
                                 + 0.707559 w_k1_MAT_s102
                                 + 0.973825 w_k1_MSFT_s102
                                 + 1.00172 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.11809 w_k1_ADM_s102
                                 + 1.04641 w_k1_AXP_s102
                                 + 0.765764 w_k1_DIS_s102
                                 + 1.04614 w_k1_FDX_s102
                                 + 0.673915 w_k1_IBM_s102
                                 + 0.905497 w_k1_INTC_s102
                                 + 0.691376 w_k1_KO_s102
                                 + 0.676213 w_k1_MAT_s102
                                 + 0.860821 w_k1_MSFT_s102
                                 + 0.894108 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.903356 w_k1_ADM_s102
                                 + 1.00735 w_k1_AXP_s102
                                 + 1.05858 w_k1_DIS_s102
                                 + 0.908015 w_k1_FDX_s102
                                 + 1.32661 w_k1_IBM_s102
                                 + 0.999848 w_k1_INTC_s102
                                 + 1.15427 w_k1_KO_s102 + 1.21609 w_k1_MAT_s102
                                 + 0.980131 w_k1_MSFT_s102
                                 + 1.06368 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.896026 w_k1_ADM_s102
                                 + 0.933215 w_k1_AXP_s102
                                 + 0.618653 w_k1_DIS_s102
                                 + 0.78858 w_k1_FDX_s102
                                 + 0.914057 w_k1_IBM_s102
                                 + 0.789558 w_k1_INTC_s102
                                 + 0.902806 w_k1_KO_s102
                                 + 0.912951 w_k1_MAT_s102
                                 + 0.795683 w_k1_MSFT_s102
                                 + 0.882825 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.24044 w_k1_ADM_s102
                                 + 1.33691 w_k1_AXP_s102
                                 + 1.44078 w_k1_DIS_s102 + 1.2515 w_k1_FDX_s102
                                 + 1.43123 w_k1_IBM_s102
                                 + 1.01684 w_k1_INTC_s102
                                 + 1.35309 w_k1_KO_s102 + 1.56505 w_k1_MAT_s102
                                 + 1.24305 w_k1_MSFT_s102
                                 + 1.19101 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.90104 w_k1_ADM_s102
                                 + 0.739723 w_k1_AXP_s102
                                 + 0.77002 w_k1_DIS_s102
                                 + 0.85651 w_k1_FDX_s102
                                 + 0.777719 w_k1_IBM_s102
                                 + 1.00526 w_k1_INTC_s102
                                 + 0.807424 w_k1_KO_s102
                                 + 0.938308 w_k1_MAT_s102
                                 + 0.899493 w_k1_MSFT_s102
                                 + 0.917174 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.22555 w_k1_ADM_s102
                                 + 1.37223 w_k1_AXP_s102
                                 + 1.30314 w_k1_DIS_s102
                                 + 1.27551 w_k1_FDX_s102
                                 + 0.861955 w_k1_IBM_s102
                                 + 1.1829 w_k1_INTC_s102 + 1.11908 w_k1_KO_s102
                                 + 1.44015 w_k1_MAT_s102 + 1.115 w_k1_MSFT_s102
                                 + 1.14484 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.26231 w_k1_ADM_s102
                                 + 1.10737 w_k1_AXP_s102
                                 + 1.59731 w_k1_DIS_s102
                                 + 1.16443 w_k1_FDX_s102
                                 + 0.976087 w_k1_IBM_s102
                                 + 1.20706 w_k1_INTC_s102
                                 + 1.28733 w_k1_KO_s102 + 1.16533 w_k1_MAT_s102
                                 + 1.25472 w_k1_MSFT_s102
                                 + 1.18415 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.878055 w_k1_ADM_s103
                                 + 0.750369 w_k1_AXP_s103
                                 + 0.636689 w_k1_DIS_s103
                                 + 0.77319 w_k1_FDX_s103
                                 + 0.706632 w_k1_IBM_s103
                                 + 0.803398 w_k1_INTC_s103
                                 + 0.760774 w_k1_KO_s103
                                 + 0.527094 w_k1_MAT_s103
                                 + 0.767728 w_k1_MSFT_s103
                                 + 0.795089 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.1131 w_k1_ADM_s103
                                 + 0.978955 w_k1_AXP_s103
                                 + 0.860877 w_k1_DIS_s103
                                 + 0.861552 w_k1_FDX_s103
                                 + 1.28799 w_k1_IBM_s103
                                 + 0.873077 w_k1_INTC_s103
                                 + 1.19406 w_k1_KO_s103 + 1.1841 w_k1_MAT_s103
                                 + 0.911251 w_k1_MSFT_s103
                                 + 0.984092 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.84117 w_k1_ADM_s103
                                 + 0.779858 w_k1_AXP_s103
                                 + 0.875268 w_k1_DIS_s103
                                 + 0.994478 w_k1_FDX_s103
                                 + 0.704965 w_k1_IBM_s103
                                 + 0.973833 w_k1_INTC_s103
                                 + 0.934143 w_k1_KO_s103
                                 + 0.718494 w_k1_MAT_s103
                                 + 1.03286 w_k1_MSFT_s103
                                 + 0.978999 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.29003 w_k1_ADM_s103
                                 + 1.43767 w_k1_AXP_s103
                                 + 1.59134 w_k1_DIS_s103
                                 + 1.38956 w_k1_FDX_s103 + 1.1234 w_k1_IBM_s103
                                 + 1.16509 w_k1_INTC_s103
                                 + 1.41466 w_k1_KO_s103 + 1.30611 w_k1_MAT_s103
                                 + 1.37739 w_k1_MSFT_s103
                                 + 1.14053 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.28704 w_k1_ADM_s103
                                 + 1.14189 w_k1_AXP_s103
                                 + 1.18917 w_k1_DIS_s103
                                 + 1.10177 w_k1_FDX_s103
                                 + 1.15986 w_k1_IBM_s103
                                 + 0.917313 w_k1_INTC_s103
                                 + 1.29591 w_k1_KO_s103 + 1.21354 w_k1_MAT_s103
                                 + 1.13758 w_k1_MSFT_s103
                                 + 1.15639 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.19262 w_k1_ADM_s103
                                 + 1.82243 w_k1_AXP_s103
                                 + 1.38363 w_k1_DIS_s103
                                 + 1.19826 w_k1_FDX_s103
                                 + 1.56812 w_k1_IBM_s103
                                 + 0.955905 w_k1_INTC_s103
                                 + 1.39288 w_k1_KO_s103 + 1.98487 w_k1_MAT_s103
                                 + 1.09676 w_k1_MSFT_s103
                                 + 1.16486 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.1558 w_k1_ADM_s103
                                 + 1.21456 w_k1_AXP_s103
                                 + 1.02277 w_k1_DIS_s103
                                 + 1.16432 w_k1_FDX_s103
                                 + 0.828153 w_k1_IBM_s103
                                 + 0.99673 w_k1_INTC_s103 + 1.0355 w_k1_KO_s103
                                 + 0.981493 w_k1_MAT_s103
                                 + 1.04332 w_k1_MSFT_s103
                                 + 1.09523 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.807791 w_k1_ADM_s103
                                 + 0.806129 w_k1_AXP_s103
                                 + 0.683489 w_k1_DIS_s103
                                 + 0.788535 w_k1_FDX_s103
                                 + 1.03015 w_k1_IBM_s103
                                 + 0.919667 w_k1_INTC_s103
                                 + 0.964045 w_k1_KO_s103
                                 + 1.03104 w_k1_MAT_s103
                                 + 0.877647 w_k1_MSFT_s103
                                 + 0.920301 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.929564 w_k1_ADM_s104
                                 + 1.00527 w_k1_AXP_s104
                                 + 1.11417 w_k1_DIS_s104
                                 + 1.01775 w_k1_FDX_s104
                                 + 0.927794 w_k1_IBM_s104
                                 + 0.981662 w_k1_INTC_s104
                                 + 1.13255 w_k1_KO_s104 + 1.212 w_k1_MAT_s104
                                 + 1.1351 w_k1_MSFT_s104
                                 + 0.914942 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.863901 w_k1_ADM_s104
                                 + 0.869716 w_k1_AXP_s104
                                 + 0.646566 w_k1_DIS_s104
                                 + 0.864189 w_k1_FDX_s104
                                 + 0.794168 w_k1_IBM_s104
                                 + 0.869608 w_k1_INTC_s104
                                 + 0.836407 w_k1_KO_s104
                                 + 0.797158 w_k1_MAT_s104
                                 + 0.88357 w_k1_MSFT_s104
                                 + 0.835145 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.2506 w_k1_ADM_s104
                                 + 1.49543 w_k1_AXP_s104
                                 + 1.52418 w_k1_DIS_s104
                                 + 1.31216 w_k1_FDX_s104
                                 + 1.32024 w_k1_IBM_s104
                                 + 1.43492 w_k1_INTC_s104
                                 + 1.38146 w_k1_KO_s104 + 1.30335 w_k1_MAT_s104
                                 + 1.29369 w_k1_MSFT_s104
                                 + 1.11116 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.814246 w_k1_ADM_s104
                                 + 0.910901 w_k1_AXP_s104
                                 + 1.22863 w_k1_DIS_s104
                                 + 0.994587 w_k1_FDX_s104
                                 + 1.34047 w_k1_IBM_s104
                                 + 1.06127 w_k1_INTC_s104
                                 + 1.43393 w_k1_KO_s104 + 1.08221 w_k1_MAT_s104
                                 + 1.12814 w_k1_MSFT_s104
                                 + 1.07731 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.670514 w_k1_ADM_s104
                                 + 0.667409 w_k1_AXP_s104
                                 + 0.580916 w_k1_DIS_s104
                                 + 0.680713 w_k1_FDX_s104
                                 + 0.797663 w_k1_IBM_s104
                                 + 0.608457 w_k1_INTC_s104
                                 + 0.785287 w_k1_KO_s104
                                 + 0.815205 w_k1_MAT_s104
                                 + 0.754903 w_k1_MSFT_s104
                                 + 0.772211 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.15007 w_k1_ADM_s104
                                 + 1.47693 w_k1_AXP_s104
                                 + 1.00939 w_k1_DIS_s104
                                 + 1.14053 w_k1_FDX_s104
                                 + 1.26402 w_k1_IBM_s104
                                 + 0.990755 w_k1_INTC_s104
                                 + 1.09871 w_k1_KO_s104 + 1.34713 w_k1_MAT_s104
                                 + 1.08695 w_k1_MSFT_s104
                                 + 0.996365 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.922324 w_k1_ADM_s104
                                 + 0.984491 w_k1_AXP_s104
                                 + 0.865793 w_k1_DIS_s104
                                 + 0.842565 w_k1_FDX_s104
                                 + 1.37702 w_k1_IBM_s104
                                 + 0.886512 w_k1_INTC_s104
                                 + 1.25201 w_k1_KO_s104 + 1.27961 w_k1_MAT_s104
                                 + 0.920826 w_k1_MSFT_s104
                                 + 0.994242 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.10884 w_k1_ADM_s104
                                 + 1.22928 w_k1_AXP_s104
                                 + 0.977015 w_k1_DIS_s104
                                 + 1.12728 w_k1_FDX_s104
                                 + 0.734847 w_k1_IBM_s104
                                 + 1.15758 w_k1_INTC_s104
                                 + 0.904031 w_k1_KO_s104
                                 + 0.786575 w_k1_MAT_s104
                                 + 1.00371 w_k1_MSFT_s104
                                 + 0.917164 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.919327 w_k1_ADM_s105
                                 + 0.972439 w_k1_AXP_s105
                                 + 1.25704 w_k1_DIS_s105
                                 + 1.01245 w_k1_FDX_s105
                                 + 0.975588 w_k1_IBM_s105
                                 + 1.06934 w_k1_INTC_s105
                                 + 1.04192 w_k1_KO_s105
                                 + 0.925187 w_k1_MAT_s105
                                 + 0.980456 w_k1_MSFT_s105
                                 + 0.916265 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.992893 w_k1_ADM_s105
                                 + 1.22313 w_k1_AXP_s105
                                 + 1.45064 w_k1_DIS_s105
                                 + 1.11284 w_k1_FDX_s105
                                 + 1.72311 w_k1_IBM_s105
                                 + 1.16056 w_k1_INTC_s105
                                 + 1.33631 w_k1_KO_s105 + 1.62707 w_k1_MAT_s105
                                 + 1.02291 w_k1_MSFT_s105
                                 + 1.24205 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.43887 w_k1_ADM_s105
                                 + 1.37174 w_k1_AXP_s105
                                 + 1.64357 w_k1_DIS_s105
                                 + 1.39457 w_k1_FDX_s105
                                 + 1.13292 w_k1_IBM_s105
                                 + 1.22561 w_k1_INTC_s105
                                 + 1.32507 w_k1_KO_s105 + 1.07587 w_k1_MAT_s105
                                 + 1.29852 w_k1_MSFT_s105
                                 + 1.17162 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.00788 w_k1_ADM_s105
                                 + 1.23188 w_k1_AXP_s105
                                 + 1.05636 w_k1_DIS_s105
                                 + 1.05803 w_k1_FDX_s105
                                 + 1.28018 w_k1_IBM_s105
                                 + 0.849908 w_k1_INTC_s105
                                 + 1.01785 w_k1_KO_s105 + 1.76845 w_k1_MAT_s105
                                 + 0.916549 w_k1_MSFT_s105
                                 + 0.99093 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.673459 w_k1_ADM_s105
                                 + 0.642343 w_k1_AXP_s105
                                 + 0.581528 w_k1_DIS_s105
                                 + 0.673046 w_k1_FDX_s105
                                 + 1.08221 w_k1_IBM_s105
                                 + 0.747864 w_k1_INTC_s105
                                 + 0.87691 w_k1_KO_s105
                                 + 0.977346 w_k1_MAT_s105
                                 + 0.749996 w_k1_MSFT_s105
                                 + 0.75366 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.840217 w_k1_ADM_s105
                                 + 0.84782 w_k1_AXP_s105
                                 + 0.921616 w_k1_DIS_s105
                                 + 1.02769 w_k1_FDX_s105
                                 + 0.671721 w_k1_IBM_s105
                                 + 1.01017 w_k1_INTC_s105
                                 + 0.809576 w_k1_KO_s105
                                 + 0.695928 w_k1_MAT_s105
                                 + 0.887288 w_k1_MSFT_s105
                                 + 0.820278 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.890592 w_k1_ADM_s105
                                 + 1.02585 w_k1_AXP_s105
                                 + 1.02308 w_k1_DIS_s105
                                 + 1.06436 w_k1_FDX_s105
                                 + 0.958679 w_k1_IBM_s105
                                 + 1.03586 w_k1_INTC_s105
                                 + 0.981022 w_k1_KO_s105
                                 + 1.20097 w_k1_MAT_s105
                                 + 0.981375 w_k1_MSFT_s105
                                 + 1.01432 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.882548 w_k1_ADM_s105
                                 + 0.783323 w_k1_AXP_s105
                                 + 0.666027 w_k1_DIS_s105
                                 + 0.874098 w_k1_FDX_s105
                                 + 0.772347 w_k1_IBM_s105
                                 + 0.785606 w_k1_INTC_s105
                                 + 0.778187 w_k1_KO_s105
                                 + 1.00224 w_k1_MAT_s105
                                 + 0.785885 w_k1_MSFT_s105
                                 + 0.828223 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.39273 w_k1_ADM_s106
                                 + 0.938559 w_k1_AXP_s106
                                 + 1.65652 w_k1_DIS_s106
                                 + 1.30916 w_k1_FDX_s106
                                 + 0.939756 w_k1_IBM_s106
                                 + 0.867671 w_k1_INTC_s106
                                 + 1.27119 w_k1_KO_s106 + 0.69929 w_k1_MAT_s106
                                 + 1.21025 w_k1_MSFT_s106 + 1.1303 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.11533 w_k1_ADM_s106
                                 + 1.29913 w_k1_AXP_s106
                                 + 1.25613 w_k1_DIS_s106
                                 + 1.12863 w_k1_FDX_s106
                                 + 1.54197 w_k1_IBM_s106
                                 + 0.84673 w_k1_INTC_s106
                                 + 1.13589 w_k1_KO_s106 + 1.66864 w_k1_MAT_s106
                                 + 0.972412 w_k1_MSFT_s106
                                 + 1.12922 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.949087 w_k1_ADM_s106
                                 + 0.964318 w_k1_AXP_s106
                                 + 0.738273 w_k1_DIS_s106
                                 + 0.783939 w_k1_FDX_s106
                                 + 1.18352 w_k1_IBM_s106
                                 + 0.752962 w_k1_INTC_s106
                                 + 0.916007 w_k1_KO_s106
                                 + 1.17254 w_k1_MAT_s106
                                 + 0.761281 w_k1_MSFT_s106
                                 + 0.971863 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.51305 w_k1_ADM_s106
                                 + 1.36538 w_k1_AXP_s106
                                 + 1.58569 w_k1_DIS_s106
                                 + 1.38352 w_k1_FDX_s106 + 1.2406 w_k1_IBM_s106
                                 + 1.01808 w_k1_INTC_s106
                                 + 1.31622 w_k1_KO_s106 + 1.42625 w_k1_MAT_s106
                                 + 1.4173 w_k1_MSFT_s106 + 1.21159 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.28078 w_k1_ADM_s106
                                 + 1.44635 w_k1_AXP_s106 + 1.0408 w_k1_DIS_s106
                                 + 1.12166 w_k1_FDX_s106
                                 + 1.00109 w_k1_IBM_s106
                                 + 0.763044 w_k1_INTC_s106
                                 + 0.849022 w_k1_KO_s106
                                 + 1.40958 w_k1_MAT_s106
                                 + 0.921543 w_k1_MSFT_s106
                                 + 1.03141 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.983262 w_k1_ADM_s106
                                 + 0.916848 w_k1_AXP_s106
                                 + 0.924862 w_k1_DIS_s106
                                 + 1.02964 w_k1_FDX_s106
                                 + 0.861355 w_k1_IBM_s106
                                 + 0.807203 w_k1_INTC_s106
                                 + 0.889532 w_k1_KO_s106
                                 + 1.02641 w_k1_MAT_s106
                                 + 1.02426 w_k1_MSFT_s106
                                 + 0.93497 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.817668 w_k1_ADM_s106
                                 + 0.780006 w_k1_AXP_s106
                                 + 0.617338 w_k1_DIS_s106
                                 + 0.73808 w_k1_FDX_s106
                                 + 0.73579 w_k1_IBM_s106
                                 + 0.743551 w_k1_INTC_s106
                                 + 0.669623 w_k1_KO_s106
                                 + 0.696354 w_k1_MAT_s106
                                 + 0.733876 w_k1_MSFT_s106
                                 + 0.834816 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.956589 w_k1_ADM_s106
                                 + 0.797551 w_k1_AXP_s106
                                 + 0.925182 w_k1_DIS_s106
                                 + 0.897127 w_k1_FDX_s106
                                 + 1.10121 w_k1_IBM_s106
                                 + 0.911379 w_k1_INTC_s106
                                 + 0.923629 w_k1_KO_s106
                                 + 0.709871 w_k1_MAT_s106
                                 + 0.933057 w_k1_MSFT_s106
                                 + 0.932282 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.19457 w_k1_ADM_s107
                                 + 1.40889 w_k1_AXP_s107
                                 + 1.28199 w_k1_DIS_s107
                                 + 1.16682 w_k1_FDX_s107
                                 + 1.65943 w_k1_IBM_s107
                                 + 1.05579 w_k1_INTC_s107 + 1.3755 w_k1_KO_s107
                                 + 1.56712 w_k1_MAT_s107
                                 + 1.23112 w_k1_MSFT_s107
                                 + 1.14792 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.630778 w_k1_ADM_s107
                                 + 0.64378 w_k1_AXP_s107
                                 + 0.466514 w_k1_DIS_s107
                                 + 0.698467 w_k1_FDX_s107
                                 + 0.700431 w_k1_IBM_s107
                                 + 0.747599 w_k1_INTC_s107
                                 + 0.587355 w_k1_KO_s107
                                 + 0.473004 w_k1_MAT_s107
                                 + 0.664187 w_k1_MSFT_s107
                                 + 0.764244 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.09455 w_k1_ADM_s107
                                 + 0.915796 w_k1_AXP_s107
                                 + 0.863575 w_k1_DIS_s107
                                 + 1.0293 w_k1_FDX_s107
                                 + 0.624471 w_k1_IBM_s107
                                 + 0.934666 w_k1_INTC_s107
                                 + 0.751694 w_k1_KO_s107
                                 + 0.567569 w_k1_MAT_s107
                                 + 0.946373 w_k1_MSFT_s107
                                 + 0.955572 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.06929 w_k1_ADM_s107
                                 + 1.61196 w_k1_AXP_s107
                                 + 1.41636 w_k1_DIS_s107
                                 + 1.19558 w_k1_FDX_s107
                                 + 1.31705 w_k1_IBM_s107
                                 + 1.12862 w_k1_INTC_s107
                                 + 1.34724 w_k1_KO_s107 + 1.75439 w_k1_MAT_s107
                                 + 1.22211 w_k1_MSFT_s107
                                 + 1.28705 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.692866 w_k1_ADM_s107
                                 + 0.935125 w_k1_AXP_s107
                                 + 0.747564 w_k1_DIS_s107
                                 + 0.769608 w_k1_FDX_s107
                                 + 0.864086 w_k1_IBM_s107
                                 + 0.909276 w_k1_INTC_s107
                                 + 0.876616 w_k1_KO_s107
                                 + 0.798307 w_k1_MAT_s107
                                 + 0.887563 w_k1_MSFT_s107
                                 + 0.846512 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.988643 w_k1_ADM_s107
                                 + 0.832142 w_k1_AXP_s107
                                 + 0.998317 w_k1_DIS_s107
                                 + 1.00996 w_k1_FDX_s107
                                 + 0.944553 w_k1_IBM_s107
                                 + 1.03085 w_k1_INTC_s107
                                 + 1.11383 w_k1_KO_s107
                                 + 0.612965 w_k1_MAT_s107
                                 + 1.05547 w_k1_MSFT_s107
                                 + 1.05744 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.983216 w_k1_ADM_s107
                                 + 1.11028 w_k1_AXP_s107
                                 + 1.06097 w_k1_DIS_s107
                                 + 1.03391 w_k1_FDX_s107
                                 + 1.07557 w_k1_IBM_s107
                                 + 0.911857 w_k1_INTC_s107
                                 + 1.0904 w_k1_KO_s107 + 0.916422 w_k1_MAT_s107
                                 + 1.02874 w_k1_MSFT_s107
                                 + 1.04577 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.33089 w_k1_ADM_s107
                                 + 1.19337 w_k1_AXP_s107
                                 + 1.96821 w_k1_DIS_s107
                                 + 1.45853 w_k1_FDX_s107
                                 + 1.05026 w_k1_IBM_s107
                                 + 1.0759 w_k1_INTC_s107 + 1.40694 w_k1_KO_s107
                                 + 0.839768 w_k1_MAT_s107
                                 + 1.49577 w_k1_MSFT_s107
                                 + 1.28083 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.982902 w_k1_ADM_s110
                                 + 1.15625 w_k1_AXP_s110
                                 + 0.787408 w_k1_DIS_s110
                                 + 1.02157 w_k1_FDX_s110
                                 + 0.82382 w_k1_IBM_s110
                                 + 1.08723 w_k1_INTC_s110
                                 + 0.764392 w_k1_KO_s110
                                 + 0.970197 w_k1_MAT_s110
                                 + 0.993554 w_k1_MSFT_s110
                                 + 0.917417 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.17687 w_k1_ADM_s110
                                 + 0.923521 w_k1_AXP_s110
                                 + 0.931808 w_k1_DIS_s110
                                 + 0.860571 w_k1_FDX_s110
                                 + 0.861094 w_k1_IBM_s110
                                 + 0.802966 w_k1_INTC_s110
                                 + 0.991138 w_k1_KO_s110
                                 + 0.78609 w_k1_MAT_s110
                                 + 0.920494 w_k1_MSFT_s110
                                 + 0.993057 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.85627 w_k1_ADM_s110
                                 + 0.777341 w_k1_AXP_s110
                                 + 0.79642 w_k1_DIS_s110
                                 + 0.75235 w_k1_FDX_s110
                                 + 1.14649 w_k1_IBM_s110
                                 + 0.801371 w_k1_INTC_s110
                                 + 0.915003 w_k1_KO_s110
                                 + 1.14735 w_k1_MAT_s110
                                 + 0.833755 w_k1_MSFT_s110
                                 + 0.942846 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.47141 w_k1_ADM_s110
                                 + 1.47233 w_k1_AXP_s110
                                 + 1.64028 w_k1_DIS_s110
                                 + 1.31687 w_k1_FDX_s110
                                 + 1.40262 w_k1_IBM_s110
                                 + 0.991965 w_k1_INTC_s110
                                 + 1.50668 w_k1_KO_s110 + 2.01656 w_k1_MAT_s110
                                 + 1.2197 w_k1_MSFT_s110 + 1.35583 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.906874 w_k1_ADM_s110
                                 + 0.739022 w_k1_AXP_s110
                                 + 0.607231 w_k1_DIS_s110
                                 + 0.717697 w_k1_FDX_s110
                                 + 0.713803 w_k1_IBM_s110
                                 + 0.581311 w_k1_INTC_s110
                                 + 0.825936 w_k1_KO_s110
                                 + 1.16474 w_k1_MAT_s110
                                 + 0.722261 w_k1_MSFT_s110
                                 + 0.875641 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.906092 w_k1_ADM_s110
                                 + 0.977045 w_k1_AXP_s110
                                 + 0.974557 w_k1_DIS_s110
                                 + 0.980532 w_k1_FDX_s110
                                 + 0.676745 w_k1_IBM_s110
                                 + 0.873748 w_k1_INTC_s110
                                 + 0.893132 w_k1_KO_s110
                                 + 0.933423 w_k1_MAT_s110
                                 + 0.941773 w_k1_MSFT_s110
                                 + 0.882583 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.46188 w_k1_ADM_s110
                                 + 1.17782 w_k1_AXP_s110
                                 + 1.23389 w_k1_DIS_s110
                                 + 1.15315 w_k1_FDX_s110
                                 + 0.97927 w_k1_IBM_s110
                                 + 0.972148 w_k1_INTC_s110
                                 + 1.25868 w_k1_KO_s110 + 1.30825 w_k1_MAT_s110
                                 + 1.01511 w_k1_MSFT_s110
                                 + 1.05485 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.56814 w_k1_ADM_s110
                                 + 1.08765 w_k1_AXP_s110
                                 + 1.08856 w_k1_DIS_s110
                                 + 0.979706 w_k1_FDX_s110
                                 + 1.29719 w_k1_IBM_s110
                                 + 0.945265 w_k1_INTC_s110
                                 + 1.16888 w_k1_KO_s110 + 1.1995 w_k1_MAT_s110
                                 + 0.935444 w_k1_MSFT_s110
                                 + 1.06654 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.660115 w_k1_ADM_s111
                                 + 0.728741 w_k1_AXP_s111
                                 + 0.60826 w_k1_DIS_s111
                                 + 0.672495 w_k1_FDX_s111
                                 + 0.812523 w_k1_IBM_s111
                                 + 0.79387 w_k1_INTC_s111
                                 + 0.67934 w_k1_KO_s111
                                 + 0.690739 w_k1_MAT_s111
                                 + 0.734522 w_k1_MSFT_s111
                                 + 0.763317 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.931898 w_k1_ADM_s111
                                 + 0.973034 w_k1_AXP_s111
                                 + 0.99186 w_k1_DIS_s111
                                 + 1.03289 w_k1_FDX_s111
                                 + 0.827889 w_k1_IBM_s111
                                 + 1.21592 w_k1_INTC_s111
                                 + 0.94601 w_k1_KO_s111
                                 + 0.970022 w_k1_MAT_s111
                                 + 1.28197 w_k1_MSFT_s111
                                 + 1.04028 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.637593 w_k1_ADM_s111
                                 + 0.705934 w_k1_AXP_s111
                                 + 0.769494 w_k1_DIS_s111
                                 + 0.809604 w_k1_FDX_s111
                                 + 0.938893 w_k1_IBM_s111
                                 + 0.905727 w_k1_INTC_s111
                                 + 0.79861 w_k1_KO_s111 + 1.18242 w_k1_MAT_s111
                                 + 0.876523 w_k1_MSFT_s111
                                 + 0.915565 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.988683 w_k1_ADM_s111
                                 + 1.44919 w_k1_AXP_s111
                                 + 1.64792 w_k1_DIS_s111
                                 + 1.40986 w_k1_FDX_s111
                                 + 1.12298 w_k1_IBM_s111
                                 + 1.09593 w_k1_INTC_s111
                                 + 1.20075 w_k1_KO_s111 + 1.37527 w_k1_MAT_s111
                                 + 1.38025 w_k1_MSFT_s111
                                 + 1.24106 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.775327 w_k1_ADM_s111
                                 + 0.704888 w_k1_AXP_s111
                                 + 0.754184 w_k1_DIS_s111
                                 + 0.790971 w_k1_FDX_s111
                                 + 0.611517 w_k1_IBM_s111
                                 + 0.911592 w_k1_INTC_s111
                                 + 0.845916 w_k1_KO_s111
                                 + 0.608551 w_k1_MAT_s111
                                 + 0.974308 w_k1_MSFT_s111
                                 + 0.908516 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.860683 w_k1_ADM_s111
                                 + 1.06317 w_k1_AXP_s111
                                 + 1.17684 w_k1_DIS_s111
                                 + 1.01741 w_k1_FDX_s111
                                 + 1.07271 w_k1_IBM_s111
                                 + 1.1894 w_k1_INTC_s111 + 1.24711 w_k1_KO_s111
                                 + 1.21891 w_k1_MAT_s111
                                 + 1.37121 w_k1_MSFT_s111
                                 + 1.11815 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.774571 w_k1_ADM_s111
                                 + 1.163 w_k1_AXP_s111 + 1.45817 w_k1_DIS_s111
                                 + 1.04126 w_k1_FDX_s111
                                 + 1.59005 w_k1_IBM_s111
                                 + 1.22407 w_k1_INTC_s111
                                 + 1.44839 w_k1_KO_s111 + 1.38692 w_k1_MAT_s111
                                 + 1.32553 w_k1_MSFT_s111 + 1.053 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.854828 w_k1_ADM_s111
                                 + 0.850221 w_k1_AXP_s111
                                 + 0.83753 w_k1_DIS_s111
                                 + 0.826334 w_k1_FDX_s111
                                 + 1.28082 w_k1_IBM_s111
                                 + 0.849143 w_k1_INTC_s111
                                 + 1.03113 w_k1_KO_s111 + 1.0558 w_k1_MAT_s111
                                 + 0.981392 w_k1_MSFT_s111
                                 + 0.963753 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.07065 w_k1_ADM_s112
                                 + 1.15399 w_k1_AXP_s112
                                 + 1.96547 w_k1_DIS_s112
                                 + 1.30234 w_k1_FDX_s112
                                 + 1.71041 w_k1_IBM_s112
                                 + 1.19535 w_k1_INTC_s112
                                 + 1.84156 w_k1_KO_s112 + 1.47131 w_k1_MAT_s112
                                 + 1.39787 w_k1_MSFT_s112
                                 + 1.35648 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.96387 w_k1_ADM_s112
                                 + 0.934125 w_k1_AXP_s112
                                 + 0.76842 w_k1_DIS_s112
                                 + 0.823332 w_k1_FDX_s112
                                 + 0.858614 w_k1_IBM_s112
                                 + 0.930273 w_k1_INTC_s112
                                 + 0.814059 w_k1_KO_s112
                                 + 0.645522 w_k1_MAT_s112
                                 + 0.790656 w_k1_MSFT_s112
                                 + 0.81013 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.711761 w_k1_ADM_s112
                                 + 0.912528 w_k1_AXP_s112
                                 + 1.08832 w_k1_DIS_s112
                                 + 0.831165 w_k1_FDX_s112
                                 + 1.62084 w_k1_IBM_s112
                                 + 1.00929 w_k1_INTC_s112
                                 + 1.32795 w_k1_KO_s112 + 1.91954 w_k1_MAT_s112
                                 + 1.00584 w_k1_MSFT_s112
                                 + 0.980281 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.0701 w_k1_ADM_s112
                                 + 1.10651 w_k1_AXP_s112
                                 + 0.733042 w_k1_DIS_s112
                                 + 0.771862 w_k1_FDX_s112
                                 + 1.33595 w_k1_IBM_s112
                                 + 0.852354 w_k1_INTC_s112
                                 + 1.14284 w_k1_KO_s112
                                 + 0.856914 w_k1_MAT_s112
                                 + 0.720468 w_k1_MSFT_s112
                                 + 0.897608 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.933669 w_k1_ADM_s112
                                 + 0.936642 w_k1_AXP_s112
                                 + 1.43019 w_k1_DIS_s112
                                 + 1.09938 w_k1_FDX_s112
                                 + 1.06601 w_k1_IBM_s112
                                 + 1.12354 w_k1_INTC_s112
                                 + 1.31773 w_k1_KO_s112 + 1.13379 w_k1_MAT_s112
                                 + 1.05969 w_k1_MSFT_s112
                                 + 1.16845 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.2389 w_k1_ADM_s112
                                 + 1.6062 w_k1_AXP_s112 + 1.18823 w_k1_DIS_s112
                                 + 1.12399 w_k1_FDX_s112 + 1.1438 w_k1_IBM_s112
                                 + 1.08988 w_k1_INTC_s112
                                 + 1.10284 w_k1_KO_s112 + 1.52168 w_k1_MAT_s112
                                 + 1.03766 w_k1_MSFT_s112
                                 + 1.12992 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.07116 w_k1_ADM_s112
                                 + 0.784241 w_k1_AXP_s112
                                 + 0.799244 w_k1_DIS_s112
                                 + 0.769692 w_k1_FDX_s112
                                 + 1.04954 w_k1_IBM_s112
                                 + 0.678709 w_k1_INTC_s112
                                 + 1.0129 w_k1_KO_s112 + 0.86113 w_k1_MAT_s112
                                 + 0.715479 w_k1_MSFT_s112
                                 + 0.923776 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.04037 w_k1_ADM_s112
                                 + 0.933259 w_k1_AXP_s112
                                 + 1.17789 w_k1_DIS_s112
                                 + 1.11274 w_k1_FDX_s112
                                 + 0.752408 w_k1_IBM_s112
                                 + 1.2145 w_k1_INTC_s112
                                 + 0.975957 w_k1_KO_s112
                                 + 0.73533 w_k1_MAT_s112
                                 + 1.14348 w_k1_MSFT_s112
                                 + 0.929926 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.20315 w_k1_ADM_s113
                                 + 1.05107 w_k1_AXP_s113
                                 + 1.07943 w_k1_DIS_s113
                                 + 1.08589 w_k1_FDX_s113
                                 + 0.944047 w_k1_IBM_s113
                                 + 0.998548 w_k1_INTC_s113
                                 + 1.12648 w_k1_KO_s113
                                 + 0.811576 w_k1_MAT_s113
                                 + 1.04046 w_k1_MSFT_s113
                                 + 1.00656 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.834811 w_k1_ADM_s113
                                 + 0.712715 w_k1_AXP_s113
                                 + 0.738045 w_k1_DIS_s113
                                 + 0.872039 w_k1_FDX_s113
                                 + 0.555346 w_k1_IBM_s113
                                 + 0.883737 w_k1_INTC_s113
                                 + 0.875808 w_k1_KO_s113
                                 + 0.633222 w_k1_MAT_s113
                                 + 0.970888 w_k1_MSFT_s113
                                 + 0.783942 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.694671 w_k1_ADM_s113
                                 + 0.665442 w_k1_AXP_s113
                                 + 0.560314 w_k1_DIS_s113
                                 + 0.628685 w_k1_FDX_s113
                                 + 0.876822 w_k1_IBM_s113
                                 + 0.740854 w_k1_INTC_s113
                                 + 0.760556 w_k1_KO_s113
                                 + 0.784039 w_k1_MAT_s113
                                 + 0.760044 w_k1_MSFT_s113
                                 + 0.815572 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.59201 w_k1_ADM_s113
                                 + 1.47639 w_k1_AXP_s113
                                 + 1.42858 w_k1_DIS_s113 + 1.2926 w_k1_FDX_s113
                                 + 1.49205 w_k1_IBM_s113
                                 + 1.18824 w_k1_INTC_s113 + 1.4572 w_k1_KO_s113
                                 + 1.93056 w_k1_MAT_s113
                                 + 1.24929 w_k1_MSFT_s113
                                 + 1.14537 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.09011 w_k1_ADM_s113
                                 + 0.732859 w_k1_AXP_s113
                                 + 0.887404 w_k1_DIS_s113
                                 + 0.861399 w_k1_FDX_s113
                                 + 0.749285 w_k1_IBM_s113
                                 + 0.826382 w_k1_INTC_s113
                                 + 1.03678 w_k1_KO_s113
                                 + 0.788942 w_k1_MAT_s113
                                 + 1.00373 w_k1_MSFT_s113
                                 + 0.886069 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.54592 w_k1_ADM_s113
                                 + 1.44971 w_k1_AXP_s113
                                 + 0.983438 w_k1_DIS_s113
                                 + 1.04987 w_k1_FDX_s113
                                 + 1.36459 w_k1_IBM_s113
                                 + 0.756559 w_k1_INTC_s113
                                 + 1.13482 w_k1_KO_s113 + 1.27583 w_k1_MAT_s113
                                 + 1.10819 w_k1_MSFT_s113
                                 + 1.12392 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.10798 w_k1_ADM_s113
                                 + 1.09711 w_k1_AXP_s113
                                 + 1.48838 w_k1_DIS_s113 + 1.296 w_k1_FDX_s113
                                 + 0.859665 w_k1_IBM_s113
                                 + 1.11623 w_k1_INTC_s113
                                 + 1.20793 w_k1_KO_s113
                                 + 0.712164 w_k1_MAT_s113
                                 + 1.31233 w_k1_MSFT_s113
                                 + 1.07064 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.880794 w_k1_ADM_s113
                                 + 1.02252 w_k1_AXP_s113
                                 + 0.889874 w_k1_DIS_s113
                                 + 0.870218 w_k1_FDX_s113
                                 + 1.07432 w_k1_IBM_s113
                                 + 0.78051 w_k1_INTC_s113
                                 + 1.23037 w_k1_KO_s113 + 1.19655 w_k1_MAT_s113
                                 + 1.01306 w_k1_MSFT_s113
                                 + 0.916585 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.985094 w_k1_ADM_s114
                                 + 0.884011 w_k1_AXP_s114
                                 + 1.01383 w_k1_DIS_s114
                                 + 0.921416 w_k1_FDX_s114
                                 + 0.939295 w_k1_IBM_s114
                                 + 0.908256 w_k1_INTC_s114
                                 + 1.09436 w_k1_KO_s114
                                 + 0.939675 w_k1_MAT_s114
                                 + 0.916566 w_k1_MSFT_s114
                                 + 1.00966 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.798107 w_k1_ADM_s114
                                 + 0.810582 w_k1_AXP_s114
                                 + 0.920742 w_k1_DIS_s114
                                 + 0.74756 w_k1_FDX_s114
                                 + 1.12011 w_k1_IBM_s114
                                 + 0.790139 w_k1_INTC_s114
                                 + 0.970281 w_k1_KO_s114
                                 + 1.11899 w_k1_MAT_s114
                                 + 0.846489 w_k1_MSFT_s114
                                 + 0.934782 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.879892 w_k1_ADM_s114
                                 + 0.695131 w_k1_AXP_s114
                                 + 0.768672 w_k1_DIS_s114
                                 + 0.781902 w_k1_FDX_s114
                                 + 0.668536 w_k1_IBM_s114
                                 + 0.786113 w_k1_INTC_s114
                                 + 0.739762 w_k1_KO_s114
                                 + 0.636534 w_k1_MAT_s114
                                 + 0.819513 w_k1_MSFT_s114
                                 + 0.823361 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.04213 w_k1_ADM_s114
                                 + 1.14182 w_k1_AXP_s114
                                 + 1.08865 w_k1_DIS_s114
                                 + 1.07626 w_k1_FDX_s114
                                 + 0.772981 w_k1_IBM_s114
                                 + 0.805098 w_k1_INTC_s114
                                 + 0.935989 w_k1_KO_s114 + 1.1666 w_k1_MAT_s114
                                 + 0.930431 w_k1_MSFT_s114
                                 + 0.97515 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.14507 w_k1_ADM_s114
                                 + 1.74883 w_k1_AXP_s114 + 1.7673 w_k1_DIS_s114
                                 + 1.38831 w_k1_FDX_s114
                                 + 1.26697 w_k1_IBM_s114
                                 + 1.03996 w_k1_INTC_s114
                                 + 1.40453 w_k1_KO_s114 + 1.79349 w_k1_MAT_s114
                                 + 1.17504 w_k1_MSFT_s114
                                 + 1.26803 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.12959 w_k1_ADM_s114
                                 + 1.05134 w_k1_AXP_s114
                                 + 1.59465 w_k1_DIS_s114
                                 + 1.15926 w_k1_FDX_s114
                                 + 1.11749 w_k1_IBM_s114
                                 + 0.895041 w_k1_INTC_s114
                                 + 1.17369 w_k1_KO_s114 + 1.05104 w_k1_MAT_s114
                                 + 1.14138 w_k1_MSFT_s114
                                 + 1.14154 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.887145 w_k1_ADM_s114
                                 + 1.03326 w_k1_AXP_s114
                                 + 1.10842 w_k1_DIS_s114
                                 + 0.924358 w_k1_FDX_s114
                                 + 1.31907 w_k1_IBM_s114
                                 + 0.966216 w_k1_INTC_s114
                                 + 1.08352 w_k1_KO_s114 + 1.37097 w_k1_MAT_s114
                                 + 0.940334 w_k1_MSFT_s114
                                 + 1.01546 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.31353 w_k1_ADM_s114
                                 + 1.23544 w_k1_AXP_s114
                                 + 1.52541 w_k1_DIS_s114
                                 + 1.32166 w_k1_FDX_s114
                                 + 1.06804 w_k1_IBM_s114
                                 + 1.20606 w_k1_INTC_s114
                                 + 1.27128 w_k1_KO_s114
                                 + 0.867426 w_k1_MAT_s114
                                 + 1.2718 w_k1_MSFT_s114 + 1.23094 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.780892 w_k1_ADM_s115
                                 + 1.09072 w_k1_AXP_s115
                                 + 1.04408 w_k1_DIS_s115
                                 + 0.911772 w_k1_FDX_s115
                                 + 1.54599 w_k1_IBM_s115
                                 + 1.44345 w_k1_INTC_s115
                                 + 1.37203 w_k1_KO_s115 + 1.16915 w_k1_MAT_s115
                                 + 0.979498 w_k1_MSFT_s115
                                 + 0.985512 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.02275 w_k1_ADM_s115
                                 + 1.17708 w_k1_AXP_s115
                                 + 0.740928 w_k1_DIS_s115
                                 + 0.830029 w_k1_FDX_s115
                                 + 1.32119 w_k1_IBM_s115
                                 + 0.945828 w_k1_INTC_s115
                                 + 0.950178 w_k1_KO_s115
                                 + 1.31357 w_k1_MAT_s115
                                 + 0.749747 w_k1_MSFT_s115
                                 + 1.05244 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.0404 w_k1_ADM_s115
                                 + 1.13446 w_k1_AXP_s115
                                 + 1.19662 w_k1_DIS_s115
                                 + 1.14973 w_k1_FDX_s115
                                 + 0.950641 w_k1_IBM_s115
                                 + 1.43371 w_k1_INTC_s115
                                 + 1.07962 w_k1_KO_s115
                                 + 0.976137 w_k1_MAT_s115
                                 + 1.21832 w_k1_MSFT_s115
                                 + 1.19768 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.915109 w_k1_ADM_s115
                                 + 0.670274 w_k1_AXP_s115
                                 + 0.856818 w_k1_DIS_s115
                                 + 0.856026 w_k1_FDX_s115
                                 + 0.742747 w_k1_IBM_s115
                                 + 1.11938 w_k1_INTC_s115
                                 + 0.919479 w_k1_KO_s115
                                 + 0.563773 w_k1_MAT_s115
                                 + 0.980321 w_k1_MSFT_s115
                                 + 0.917702 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.28151 w_k1_ADM_s115
                                 + 1.24739 w_k1_AXP_s115
                                 + 1.41051 w_k1_DIS_s115
                                 + 1.42509 w_k1_FDX_s115
                                 + 0.831853 w_k1_IBM_s115
                                 + 1.24198 w_k1_INTC_s115
                                 + 1.10575 w_k1_KO_s115
                                 + 0.886419 w_k1_MAT_s115
                                 + 1.26784 w_k1_MSFT_s115 + 1.1719 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.33856 w_k1_ADM_s115
                                 + 1.67195 w_k1_AXP_s115 + 1.749 w_k1_DIS_s115
                                 + 1.50473 w_k1_FDX_s115
                                 + 1.46291 w_k1_IBM_s115
                                 + 1.18593 w_k1_INTC_s115
                                 + 1.22369 w_k1_KO_s115 + 1.95184 w_k1_MAT_s115
                                 + 1.25261 w_k1_MSFT_s115 + 1.2559 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.727677 w_k1_ADM_s115
                                 + 0.655961 w_k1_AXP_s115
                                 + 0.531586 w_k1_DIS_s115
                                 + 0.688156 w_k1_FDX_s115
                                 + 0.692737 w_k1_IBM_s115
                                 + 0.949235 w_k1_INTC_s115
                                 + 0.744764 w_k1_KO_s115
                                 + 0.64575 w_k1_MAT_s115
                                 + 0.74612 w_k1_MSFT_s115
                                 + 0.878745 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.02522 w_k1_ADM_s115
                                 + 1.03535 w_k1_AXP_s115
                                 + 0.876816 w_k1_DIS_s115
                                 + 0.945006 w_k1_FDX_s115
                                 + 0.934222 w_k1_IBM_s115
                                 + 1.12653 w_k1_INTC_s115
                                 + 0.988086 w_k1_KO_s115
                                 + 1.08651 w_k1_MAT_s115
                                 + 1.00935 w_k1_MSFT_s115
                                 + 1.01806 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.1362 w_k1_ADM_s116
                                 + 1.08193 w_k1_AXP_s116
                                 + 1.31621 w_k1_DIS_s116
                                 + 1.08805 w_k1_FDX_s116
                                 + 1.29098 w_k1_IBM_s116
                                 + 1.18912 w_k1_INTC_s116
                                 + 1.28704 w_k1_KO_s116 + 1.27456 w_k1_MAT_s116
                                 + 1.21907 w_k1_MSFT_s116
                                 + 1.14091 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.16321 w_k1_ADM_s116
                                 + 1.72274 w_k1_AXP_s116
                                 + 1.74591 w_k1_DIS_s116 + 1.4752 w_k1_FDX_s116
                                 + 1.51686 w_k1_IBM_s116
                                 + 1.43498 w_k1_INTC_s116
                                 + 1.43971 w_k1_KO_s116 + 1.80367 w_k1_MAT_s116
                                 + 1.40966 w_k1_MSFT_s116
                                 + 1.28766 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.761096 w_k1_ADM_s116
                                 + 0.924452 w_k1_AXP_s116
                                 + 0.72781 w_k1_DIS_s116
                                 + 0.758465 w_k1_FDX_s116
                                 + 0.858218 w_k1_IBM_s116
                                 + 0.877683 w_k1_INTC_s116
                                 + 0.788076 w_k1_KO_s116
                                 + 0.810168 w_k1_MAT_s116
                                 + 0.736802 w_k1_MSFT_s116
                                 + 0.882191 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.3466 w_k1_ADM_s116
                                 + 1.44533 w_k1_AXP_s116
                                 + 1.43353 w_k1_DIS_s116
                                 + 1.37523 w_k1_FDX_s116
                                 + 1.03855 w_k1_IBM_s116
                                 + 1.62531 w_k1_INTC_s116
                                 + 1.06352 w_k1_KO_s116 + 0.73203 w_k1_MAT_s116
                                 + 1.14861 w_k1_MSFT_s116
                                 + 1.22667 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.826961 w_k1_ADM_s116
                                 + 0.892425 w_k1_AXP_s116
                                 + 1.06887 w_k1_DIS_s116
                                 + 0.880213 w_k1_FDX_s116
                                 + 1.33213 w_k1_IBM_s116
                                 + 1.0055 w_k1_INTC_s116 + 1.26997 w_k1_KO_s116
                                 + 1.10781 w_k1_MAT_s116
                                 + 0.916666 w_k1_MSFT_s116
                                 + 1.07652 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.17528 w_k1_ADM_s116
                                 + 1.45674 w_k1_AXP_s116
                                 + 0.930189 w_k1_DIS_s116
                                 + 1.1877 w_k1_FDX_s116
                                 + 0.898252 w_k1_IBM_s116
                                 + 1.4813 w_k1_INTC_s116 + 1.07654 w_k1_KO_s116
                                 + 0.950263 w_k1_MAT_s116
                                 + 1.09953 w_k1_MSFT_s116
                                 + 1.03416 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.90978 w_k1_ADM_s116
                                 + 0.859096 w_k1_AXP_s116
                                 + 1.00354 w_k1_DIS_s116
                                 + 1.04412 w_k1_FDX_s116
                                 + 0.704151 w_k1_IBM_s116
                                 + 1.0841 w_k1_INTC_s116
                                 + 0.847389 w_k1_KO_s116
                                 + 0.449089 w_k1_MAT_s116
                                 + 1.02666 w_k1_MSFT_s116
                                 + 0.979481 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.07749 w_k1_ADM_s116
                                 + 1.16244 w_k1_AXP_s116
                                 + 0.90198 w_k1_DIS_s116
                                 + 0.961456 w_k1_FDX_s116
                                 + 0.98155 w_k1_IBM_s116
                                 + 1.00616 w_k1_INTC_s116
                                 + 1.08778 w_k1_KO_s116 + 1.56112 w_k1_MAT_s116
                                 + 1.07095 w_k1_MSFT_s116
                                 + 1.12223 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.10039 w_k1_ADM_s117
                                 + 1.26801 w_k1_AXP_s117
                                 + 0.793429 w_k1_DIS_s117
                                 + 1.01595 w_k1_FDX_s117
                                 + 0.750789 w_k1_IBM_s117
                                 + 0.933929 w_k1_INTC_s117
                                 + 0.733535 w_k1_KO_s117
                                 + 1.43744 w_k1_MAT_s117
                                 + 1.04784 w_k1_MSFT_s117 + 0.9755 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.950716 w_k1_ADM_s117
                                 + 0.950962 w_k1_AXP_s117
                                 + 0.859536 w_k1_DIS_s117
                                 + 0.957858 w_k1_FDX_s117
                                 + 1.01286 w_k1_IBM_s117
                                 + 0.898341 w_k1_INTC_s117
                                 + 0.944713 w_k1_KO_s117
                                 + 1.05457 w_k1_MAT_s117
                                 + 0.913628 w_k1_MSFT_s117
                                 + 0.994388 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.972656 w_k1_ADM_s117
                                 + 1.42072 w_k1_AXP_s117
                                 + 1.48276 w_k1_DIS_s117 + 1.4002 w_k1_FDX_s117
                                 + 1.0057 w_k1_IBM_s117
                                 + 1.30615 w_k1_INTC_s117
                                 + 1.06786 w_k1_KO_s117 + 1.13567 w_k1_MAT_s117
                                 + 1.26943 w_k1_MSFT_s117
                                 + 1.17919 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.556361 w_k1_ADM_s117
                                 + 0.623759 w_k1_AXP_s117
                                 + 0.620819 w_k1_DIS_s117
                                 + 0.762303 w_k1_FDX_s117
                                 + 0.784598 w_k1_IBM_s117
                                 + 1.01532 w_k1_INTC_s117
                                 + 0.860728 w_k1_KO_s117
                                 + 0.921288 w_k1_MAT_s117
                                 + 0.859157 w_k1_MSFT_s117
                                 + 0.833733 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.595772 w_k1_ADM_s117
                                 + 0.742424 w_k1_AXP_s117
                                 + 0.807396 w_k1_DIS_s117
                                 + 0.825948 w_k1_FDX_s117
                                 + 1.27093 w_k1_IBM_s117
                                 + 1.06911 w_k1_INTC_s117
                                 + 1.04815 w_k1_KO_s117 + 1.2828 w_k1_MAT_s117
                                 + 0.949775 w_k1_MSFT_s117
                                 + 0.933422 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.946862 w_k1_ADM_s117
                                 + 0.871862 w_k1_AXP_s117
                                 + 0.696952 w_k1_DIS_s117
                                 + 0.880273 w_k1_FDX_s117
                                 + 0.714712 w_k1_IBM_s117
                                 + 0.803525 w_k1_INTC_s117
                                 + 0.75004 w_k1_KO_s117
                                 + 0.818712 w_k1_MAT_s117
                                 + 0.925357 w_k1_MSFT_s117
                                 + 0.927103 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.877068 w_k1_ADM_s117
                                 + 0.987407 w_k1_AXP_s117
                                 + 1.07582 w_k1_DIS_s117
                                 + 1.14724 w_k1_FDX_s117
                                 + 0.754818 w_k1_IBM_s117
                                 + 1.09667 w_k1_INTC_s117
                                 + 0.973822 w_k1_KO_s117
                                 + 0.881889 w_k1_MAT_s117
                                 + 1.17591 w_k1_MSFT_s117
                                 + 0.962501 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.941774 w_k1_ADM_s117
                                 + 1.41291 w_k1_AXP_s117
                                 + 1.26885 w_k1_DIS_s117
                                 + 1.14761 w_k1_FDX_s117 + 1.2164 w_k1_IBM_s117
                                 + 1.22243 w_k1_INTC_s117
                                 + 1.21417 w_k1_KO_s117 + 1.9678 w_k1_MAT_s117
                                 + 1.15015 w_k1_MSFT_s117
                                 + 1.18618 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0103467148499415 w_k1_ADM_s000
                                 + 0.0039632044140595 w_k1_ADM_s001
                                 + 0.0101308570284346 w_k1_ADM_s002
                                 + 0.0055932846260629 w_k1_ADM_s003
                                 + 0.003655263583541 w_k1_ADM_s004
                                 + 0.00353743073525945 w_k1_ADM_s005
                                 + 0.00555932864460515 w_k1_ADM_s006
                                 + 0.00828401131092652 w_k1_ADM_s007
                                 + 0.0374357171277346 w_k1_ADM_s010
                                 + 0.0256477817990365 w_k1_ADM_s011
                                 + 0.0210214336388923 w_k1_ADM_s012
                                 + 0.0143288048695004 w_k1_ADM_s013
                                 + 0.0269223993995167 w_k1_ADM_s014
                                 + 0.0210004166901593 w_k1_ADM_s015
                                 + 0.0198418759192962 w_k1_ADM_s016
                                 + 0.0378405616778529 w_k1_ADM_s017
                                 + 0.0253035734884142 w_k1_ADM_s100
                                 + 0.0584929042867206 w_k1_ADM_s101
                                 + 0.11217242405815 w_k1_ADM_s102
                                 + 0.111758497719965 w_k1_ADM_s103
                                 + 0.0882798279014355 w_k1_ADM_s104
                                 + 0.106650048660564 w_k1_ADM_s105
                                 + 0.073988616403321 w_k1_ADM_s106
                                 + 0.0394024856614081 w_k1_ADM_s107
                                 + 0.0136794885276821 w_k1_ADM_s110
                                 + 0.0127465494196781 w_k1_ADM_s111
                                 + 0.00872081807363189 w_k1_ADM_s112
                                 + 0.0253897738147645 w_k1_ADM_s113
                                 + 0.0210488046942547 w_k1_ADM_s114
                                 + 0.0362232159317383 w_k1_ADM_s115
                                 + 0.00928010149714899 w_k1_ADM_s116
                                 + 0.0160359699483372 w_k1_ADM_s117
                                 + 0.0103413885972228 w_k1_AXP_s000
                                 + 0.00373332606194963 w_k1_AXP_s001
                                 + 0.0110503812596724 w_k1_AXP_s002
                                 + 0.00577274691506692 w_k1_AXP_s003
                                 + 0.00374480745762492 w_k1_AXP_s004
                                 + 0.00313402952622515 w_k1_AXP_s005
                                 + 0.00574220011220782 w_k1_AXP_s006
                                 + 0.00954211402834895 w_k1_AXP_s007
                                 + 0.0376134042870289 w_k1_AXP_s010
                                 + 0.0226292192808539 w_k1_AXP_s011
                                 + 0.0171654756570963 w_k1_AXP_s012
                                 + 0.0154210218054392 w_k1_AXP_s013
                                 + 0.0268774234105708 w_k1_AXP_s014
                                 + 0.0203133182706799 w_k1_AXP_s015
                                 + 0.0190345131027967 w_k1_AXP_s016
                                 + 0.0354983260138093 w_k1_AXP_s017
                                 + 0.0239429417187363 w_k1_AXP_s100
                                 + 0.0616692547948811 w_k1_AXP_s101
                                 + 0.111056169483819 w_k1_AXP_s102
                                 + 0.113330600586523 w_k1_AXP_s103
                                 + 0.0975387634583918 w_k1_AXP_s104
                                 + 0.111915692732627 w_k1_AXP_s105
                                 + 0.071945507003645 w_k1_AXP_s106
                                 + 0.0435999225522607 w_k1_AXP_s107
                                 + 0.0118134668563097 w_k1_AXP_s110
                                 + 0.015132570604573 w_k1_AXP_s111
                                 + 0.0085128532753352 w_k1_AXP_s112
                                 + 0.0238653645320734 w_k1_AXP_s113
                                 + 0.0211232815257233 w_k1_AXP_s114
                                 + 0.0380116593792327 w_k1_AXP_s115
                                 + 0.0101575047531974 w_k1_AXP_s116
                                 + 0.0198456942448486 w_k1_AXP_s117
                                 + 0.0109096553989818 w_k1_DIS_s000
                                 + 0.00391360448232773 w_k1_DIS_s001
                                 + 0.00979427871438012 w_k1_DIS_s002
                                 + 0.00591039020575387 w_k1_DIS_s003
                                 + 0.00371204732454882 w_k1_DIS_s004
                                 + 0.00300042689717391 w_k1_DIS_s005
                                 + 0.00620762794949979 w_k1_DIS_s006
                                 + 0.00923017701523685 w_k1_DIS_s007
                                 + 0.0393193704273878 w_k1_DIS_s010
                                 + 0.0250489824296922 w_k1_DIS_s011
                                 + 0.017229943447813 w_k1_DIS_s012
                                 + 0.0152773551378977 w_k1_DIS_s013
                                 + 0.0317588405448575 w_k1_DIS_s014
                                 + 0.0193415419900087 w_k1_DIS_s015
                                 + 0.0195385101756768 w_k1_DIS_s016
                                 + 0.0365760360901213 w_k1_DIS_s017
                                 + 0.0246076067086131 w_k1_DIS_s100
                                 + 0.0652441719172064 w_k1_DIS_s101
                                 + 0.110045861008054 w_k1_DIS_s102
                                 + 0.104808934573525 w_k1_DIS_s103
                                 + 0.0905456369355811 w_k1_DIS_s104
                                 + 0.115806667365566 w_k1_DIS_s105
                                 + 0.0699159151353864 w_k1_DIS_s106
                                 + 0.0427958645176031 w_k1_DIS_s107
                                 + 0.0118139921969424 w_k1_DIS_s110
                                 + 0.0164282406460697 w_k1_DIS_s111
                                 + 0.00966300704731185 w_k1_DIS_s112
                                 + 0.0229045360372095 w_k1_DIS_s113
                                 + 0.0245138224542634 w_k1_DIS_s114
                                 + 0.0368678160214018 w_k1_DIS_s115
                                 + 0.00978633985929209 w_k1_DIS_s116
                                 + 0.0179930803799858 w_k1_DIS_s117
                                 + 0.0103106774981262 w_k1_FDX_s000
                                 + 0.00358729524150542 w_k1_FDX_s001
                                 + 0.0103774869315993 w_k1_FDX_s002
                                 + 0.00563211608605707 w_k1_FDX_s003
                                 + 0.00350165522937713 w_k1_FDX_s004
                                 + 0.00341840228547752 w_k1_FDX_s005
                                 + 0.00621877910248065 w_k1_FDX_s006
                                 + 0.00886782700795254 w_k1_FDX_s007
                                 + 0.0367379707925426 w_k1_FDX_s010
                                 + 0.0247948052943724 w_k1_FDX_s011
                                 + 0.0165362721615178 w_k1_FDX_s012
                                 + 0.0146956809967686 w_k1_FDX_s013
                                 + 0.0293969652038701 w_k1_FDX_s014
                                 + 0.0191814704475082 w_k1_FDX_s015
                                 + 0.0197992384194034 w_k1_FDX_s016
                                 + 0.0339553414033118 w_k1_FDX_s017
                                 + 0.02406112247845 w_k1_FDX_s100
                                 + 0.0646191498846143 w_k1_FDX_s101
                                 + 0.109834139547221 w_k1_FDX_s102
                                 + 0.108319647657505 w_k1_FDX_s103
                                 + 0.0909059633821978 w_k1_FDX_s104
                                 + 0.114504731468614 w_k1_FDX_s105
                                 + 0.070102204043797 w_k1_FDX_s106
                                 + 0.0412980138368749 w_k1_FDX_s107
                                 + 0.0111746656164254 w_k1_FDX_s110
                                 + 0.0152725404519886 w_k1_FDX_s111
                                 + 0.00841326720725902 w_k1_FDX_s112
                                 + 0.0224991985707298 w_k1_FDX_s113
                                 + 0.0209923497916609 w_k1_FDX_s114
                                 + 0.0367548718736161 w_k1_FDX_s115
                                 + 0.00956454826729355 w_k1_FDX_s116
                                 + 0.0186156619719964 w_k1_FDX_s117
                                 + 0.00998861594521162 w_k1_IBM_s000
                                 + 0.0039718215491135 w_k1_IBM_s001
                                 + 0.0104059869900141 w_k1_IBM_s002
                                 + 0.00612849388384226 w_k1_IBM_s003
                                 + 0.00393478014775202 w_k1_IBM_s004
                                 + 0.00295947248948208 w_k1_IBM_s005
                                 + 0.00652765146195593 w_k1_IBM_s006
                                 + 0.0091296231583579 w_k1_IBM_s007
                                 + 0.0375331616753268 w_k1_IBM_s010
                                 + 0.023275928517012 w_k1_IBM_s011
                                 + 0.0164624967308303 w_k1_IBM_s012
                                 + 0.0154726591700782 w_k1_IBM_s013
                                 + 0.0306821998695189 w_k1_IBM_s014
                                 + 0.0192025720880955 w_k1_IBM_s015
                                 + 0.0188291224700986 w_k1_IBM_s016
                                 + 0.0349570041305538 w_k1_IBM_s017
                                 + 0.0247460979716048 w_k1_IBM_s100
                                 + 0.0632576134254511 w_k1_IBM_s101
                                 + 0.10695643878327 w_k1_IBM_s102
                                 + 0.104476232536927 w_k1_IBM_s103
                                 + 0.0962003398335739 w_k1_IBM_s104
                                 + 0.114988736905699 w_k1_IBM_s105
                                 + 0.0752028142034288 w_k1_IBM_s106
                                 + 0.0410557986338664 w_k1_IBM_s107
                                 + 0.0112759017086103 w_k1_IBM_s110
                                 + 0.015591814380107 w_k1_IBM_s111
                                 + 0.009500428022459 w_k1_IBM_s112
                                 + 0.023387618208781 w_k1_IBM_s113
                                 + 0.020952143350553 w_k1_IBM_s114
                                 + 0.0378980631737569 w_k1_IBM_s115
                                 + 0.00934264832235602 w_k1_IBM_s116
                                 + 0.0180360369024809 w_k1_IBM_s117
                                 + 0.0101629176578629 w_k1_INTC_s000
                                 + 0.00355168356163036 w_k1_INTC_s001
                                 + 0.0120306996420416 w_k1_INTC_s002
                                 + 0.00526276961105705 w_k1_INTC_s003
                                 + 0.00326866392031041 w_k1_INTC_s004
                                 + 0.00290122809875679 w_k1_INTC_s005
                                 + 0.00600552166903075 w_k1_INTC_s006
                                 + 0.00903943680386678 w_k1_INTC_s007
                                 + 0.0409776414046984 w_k1_INTC_s010
                                 + 0.0276068555838403 w_k1_INTC_s011
                                 + 0.016052703349762 w_k1_INTC_s012
                                 + 0.0126804423878466 w_k1_INTC_s013
                                 + 0.0281896234184743 w_k1_INTC_s014
                                 + 0.0174113234409425 w_k1_INTC_s015
                                 + 0.021143086705721 w_k1_INTC_s016
                                 + 0.0371822760266809 w_k1_INTC_s017
                                 + 0.0223197523327013 w_k1_INTC_s100
                                 + 0.0653796808403403 w_k1_INTC_s101
                                 + 0.103725886207229 w_k1_INTC_s102
                                 + 0.0998527652437139 w_k1_INTC_s103
                                 + 0.0918508089737564 w_k1_INTC_s104
                                 + 0.108589657500683 w_k1_INTC_s105
                                 + 0.0584646705932154 w_k1_INTC_s106
                                 + 0.0386716862307121 w_k1_INTC_s107
                                 + 0.00996275110477609 w_k1_INTC_s110
                                 + 0.0164614424262004 w_k1_INTC_s111
                                 + 0.00868512662506686 w_k1_INTC_s112
                                 + 0.0205345169247577 w_k1_INTC_s113
                                 + 0.0188693141682379 w_k1_INTC_s114
                                 + 0.0451330404537705 w_k1_INTC_s115
                                 + 0.0105528941514968 w_k1_INTC_s116
                                 + 0.0191479361747322 w_k1_INTC_s117
                                 + 0.0102954171102152 w_k1_KO_s000
                                 + 0.00360539537020926 w_k1_KO_s001
                                 + 0.011308282771931 w_k1_KO_s002
                                 + 0.00595146640141011 w_k1_KO_s003
                                 + 0.0042762079509795 w_k1_KO_s004
                                 + 0.00286322979325712 w_k1_KO_s005
                                 + 0.00669273799383476 w_k1_KO_s006
                                 + 0.00894200815330499 w_k1_KO_s007
                                 + 0.0397272774943267 w_k1_KO_s010
                                 + 0.024303459779229 w_k1_KO_s011
                                 + 0.0188525715219814 w_k1_KO_s012
                                 + 0.0135676174296196 w_k1_KO_s013
                                 + 0.0305921418208027 w_k1_KO_s014
                                 + 0.0192448391087805 w_k1_KO_s015
                                 + 0.0179526432198911 w_k1_KO_s016
                                 + 0.0353174360244747 w_k1_KO_s017
                                 + 0.0225969478698724 w_k1_KO_s100
                                 + 0.0656006678021472 w_k1_KO_s101
                                 + 0.109973600945787 w_k1_KO_s102
                                 + 0.114714035358484 w_k1_KO_s103
                                 + 0.100668168170427 w_k1_KO_s104
                                 + 0.1112132926873 w_k1_KO_s105
                                 + 0.0664771160661944 w_k1_KO_s106
                                 + 0.0425179136225092 w_k1_KO_s107
                                 + 0.0122294554985004 w_k1_KO_s110
                                 + 0.0159382392770471 w_k1_KO_s111
                                 + 0.00982881425109907 w_k1_KO_s112
                                 + 0.0257932761385956 w_k1_KO_s113
                                 + 0.0220541375441886 w_k1_KO_s114
                                 + 0.0388026056703205 w_k1_KO_s115
                                 + 0.0096642053603734 w_k1_KO_s116
                                 + 0.01799513611334 w_k1_KO_s117
                                 + 0.0111022555215105 w_k1_MAT_s000
                                 + 0.00381549664074883 w_k1_MAT_s001
                                 + 0.0136061685515608 w_k1_MAT_s002
                                 + 0.00620393581964094 w_k1_MAT_s003
                                 + 0.00336108542332871 w_k1_MAT_s004
                                 + 0.00291817121339348 w_k1_MAT_s005
                                 + 0.00698740810042499 w_k1_MAT_s006
                                 + 0.0100069058243703 w_k1_MAT_s007
                                 + 0.0381975714859004 w_k1_MAT_s010
                                 + 0.0237263931988116 w_k1_MAT_s011
                                 + 0.0170060279724837 w_k1_MAT_s012
                                 + 0.0174110406194563 w_k1_MAT_s013
                                 + 0.0386141911661577 w_k1_MAT_s014
                                 + 0.0196604710464259 w_k1_MAT_s015
                                 + 0.0206742273151537 w_k1_MAT_s016
                                 + 0.0308559275703734 w_k1_MAT_s017
                                 + 0.0242389447102105 w_k1_MAT_s100
                                 + 0.0653001557878937 w_k1_MAT_s101
                                 + 0.115844631266037 w_k1_MAT_s102
                                 + 0.110470770873101 w_k1_MAT_s103
                                 + 0.0991237043883473 w_k1_MAT_s104
                                 + 0.128786303448382 w_k1_MAT_s105
                                 + 0.0765807364232819 w_k1_MAT_s106
                                 + 0.0381017675759073 w_k1_MAT_s107
                                 + 0.0136814252596658 w_k1_MAT_s110
                                 + 0.0171347418846526 w_k1_MAT_s111
                                 + 0.00900256990471285 w_k1_MAT_s112
                                 + 0.0243097366717226 w_k1_MAT_s113
                                 + 0.0221684910108794 w_k1_MAT_s114
                                 + 0.0370736676652309 w_k1_MAT_s115
                                 + 0.00920564149588578 w_k1_MAT_s116
                                 + 0.0242142380650243 w_k1_MAT_s117
                                 + 0.00950011399438146 w_k1_MSFT_s000
                                 + 0.00323734993155932 w_k1_MSFT_s001
                                 + 0.010763872654128 w_k1_MSFT_s002
                                 + 0.00571830610295234 w_k1_MSFT_s003
                                 + 0.00352556333701463 w_k1_MSFT_s004
                                 + 0.00327146168569082 w_k1_MSFT_s005
                                 + 0.00603731949220621 w_k1_MSFT_s006
                                 + 0.00934387117858481 w_k1_MSFT_s007
                                 + 0.036034424985675 w_k1_MSFT_s010
                                 + 0.0245983831011312 w_k1_MSFT_s011
                                 + 0.0166129047005358 w_k1_MSFT_s012
                                 + 0.0137713698281855 w_k1_MSFT_s013
                                 + 0.0310095654919427 w_k1_MSFT_s014
                                 + 0.0170069431952994 w_k1_MSFT_s015
                                 + 0.0197672314157993 w_k1_MSFT_s016
                                 + 0.0328723811482855 w_k1_MSFT_s017
                                 + 0.0230246238621592 w_k1_MSFT_s100
                                 + 0.0606959329390319 w_k1_MSFT_s101
                                 + 0.107386766071908 w_k1_MSFT_s102
                                 + 0.107308076564891 w_k1_MSFT_s103
                                 + 0.0941940782071865 w_k1_MSFT_s104
                                 + 0.105728870962523 w_k1_MSFT_s105
                                 + 0.0670087936838149 w_k1_MSFT_s106
                                 + 0.0418685127851976 w_k1_MSFT_s107
                                 + 0.0107642872916148 w_k1_MSFT_s110
                                 + 0.0179888356221907 w_k1_MSFT_s111
                                 + 0.00837856163393702 w_k1_MSFT_s112
                                 + 0.0237435750653221 w_k1_MSFT_s113
                                 + 0.0205297432701234 w_k1_MSFT_s114
                                 + 0.0381143682780524 w_k1_MSFT_s115
                                 + 0.00955096991622844 w_k1_MSFT_s116
                                 + 0.0189287628020522 w_k1_MSFT_s117
                                 + 0.0107132461536521 w_k1_PG_s000
                                 + 0.00372331167480168 w_k1_PG_s001
                                 + 0.0104870216922177 w_k1_PG_s002
                                 + 0.00596405632182872 w_k1_PG_s003
                                 + 0.00355700449937295 w_k1_PG_s004
                                 + 0.00308731041366506 w_k1_PG_s005
                                 + 0.00682790528473562 w_k1_PG_s006
                                 + 0.00978585526225112 w_k1_PG_s007
                                 + 0.0362824725411614 w_k1_PG_s010
                                 + 0.025098772890612 w_k1_PG_s011
                                 + 0.0154630616585308 w_k1_PG_s012
                                 + 0.0142311619222321 w_k1_PG_s013
                                 + 0.0300552371461986 w_k1_PG_s014
                                 + 0.0193860467977664 w_k1_PG_s015
                                 + 0.0195462142029054 w_k1_PG_s016
                                 + 0.0332826887265382 w_k1_PG_s017
                                 + 0.0236545392278247 w_k1_PG_s100
                                 + 0.065142395530204 w_k1_PG_s101
                                 + 0.110545214946845 w_k1_PG_s102
                                 + 0.10786236848661 w_k1_PG_s103
                                 + 0.0865087673699732 w_k1_PG_s104
                                 + 0.107190581841442 w_k1_PG_s105
                                 + 0.0698605235754397 w_k1_PG_s106
                                 + 0.0418983642791673 w_k1_PG_s107
                                 + 0.0115382869526647 w_k1_PG_s110
                                 + 0.0158883643132346 w_k1_PG_s111
                                 + 0.00869802365606445 w_k1_PG_s112
                                 + 0.0220306053173163 w_k1_PG_s113
                                 + 0.0214731578084 w_k1_PG_s114
                                 + 0.0387511421968786 w_k1_PG_s115
                                 + 0.00983969309709325 w_k1_PG_s116
                                 + 0.018595084356602 w_k1_PG_s117 >= 1.18
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
