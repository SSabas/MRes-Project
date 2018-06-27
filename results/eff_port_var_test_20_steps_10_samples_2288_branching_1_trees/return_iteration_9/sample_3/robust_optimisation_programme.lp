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
 w_k1_ADM_s0_return_constraint:  1.09599 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.876403 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.09909 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.15867 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.980508 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.19116 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.720053 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.900744 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.786404 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.22804 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.957349 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.667417 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.851075 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.07067 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.32197 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.02101 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.747328 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.953503 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.24551 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.3093 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.17126 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.872544 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.596209 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.962321 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.836399 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.149 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.19004 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.896639 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.00701 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.828078 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.39224 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.886391 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.994371 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.20759 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.11618 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.837712 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.963788 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.761788 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20119 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.892715 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.994296 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.36981 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.80396 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.23581 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.54468 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.807162 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.04448 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.64328 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.07491 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.707114 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.919947 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.21454 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.31231 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.871731 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.611462 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.03975 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.34113 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.35437 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.753157 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.870816 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.647477 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.07158 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.829573 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.30254 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.1827 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.85255 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.33438 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.863887 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.66465 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.87797 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.786064 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.20975 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.22156 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.699238 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.985339 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.66526 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.28957 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.804207 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.951699 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.38195 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.858387 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.37983 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.525919 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.896889 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.12452 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.99818 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.15721 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.63465 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.809904 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.23193 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.07338 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.759443 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.559116 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.00198 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.70652 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.2806 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.04524 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.713089 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.530843 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.00373 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.08895 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.36731 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.11811 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.793221 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.26169 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.787847 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.7591 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.860818 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.955227 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.28911 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.35266 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.67943 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.886396 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.60404 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16393 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.878884 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.965456 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.18917 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.882576 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.19356 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.688539 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.993316 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.931261 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.55112 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.0958 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.711295 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.809461 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.23863 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.06986 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.895346 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.653277 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.941636 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.36341 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.33544 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.02189 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.701493 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.649357 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.951785 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.00152 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.35577 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.12391 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.969346 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.06511 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.901337 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.51009 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.768078 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.00123 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.27561 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.06992 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.738052 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.970197 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.829572 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.1589 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.935845 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.895783 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.56882 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.926799 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.16381 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.667297 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.674639 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.93614 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.3296 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.28276 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.981662 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.27591 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.867968 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.61957 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.778256 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.785968 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.14948 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.26992 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.84997 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.72381 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.21969 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.792771 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.04556 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.700655 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.829402 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.23547 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.530897 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.72552 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.756631 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.15925 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.50797 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.637781 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.800221 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.59278 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.797814 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.960851 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.474515 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.05057 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.970406 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.01657 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.1851 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.741279 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.99456 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.744367 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.0773 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.979164 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.31621 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.26515 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.773865 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.01154 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.15355 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.04413 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.907905 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.717968 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.21284 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.70685 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.51902 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.04424 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.920849 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.882666 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.07916 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.1189 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.38604 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.12405 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.03117 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.03695 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.07324 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.55291 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.899017 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.20358 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.994565 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.15953 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.842836 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.986701 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.996866 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.24198 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.896932 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.874632 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.20733 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.934276 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.19585 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.653755 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.856354 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.32005 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.49357 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.20987 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.856932 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.04931 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.14595 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.39044 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.826984 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.720569 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.13951 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.5334 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.24656 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.933922 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.01055 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.627016 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.993116 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.810248 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.952005 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.06582 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.671045 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.19568 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.739414 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.50759 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.05791 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.834016 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.996612 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.43158 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.749104 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.956651 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.595358 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.29464 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.908943 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.874606 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.58443 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.943334 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.21032 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.500805 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.581625 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 2.02075 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.23749 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.15465 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.895837 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.10484 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.0265 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.79725 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.79492 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.773935 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.17698 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.28212 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.860298 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.63437 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.993188 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.743581 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.13403 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.579971 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.725343 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.34441 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.6456 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.43905 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.841265 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.28282 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.47549 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.563334 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.15027 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.63073 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.58157 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.05829 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.557644 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15883 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.900881 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.954267 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.14623 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.86575 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.16458 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.689446 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.977473 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.999891 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.40421 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.09356 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.780696 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.874069 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.26539 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.0656 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.862602 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.710621 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.10856 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.51708 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.17658 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.05394 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.723873 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.758925 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.981851 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.06486 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.2213 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.06187 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.03439 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.07027 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.935232 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.34933 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.761572 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.01958 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.15238 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.0331 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.816879 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.03218 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.822617 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09014 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.930709 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.00987 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.15632 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.883654 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.16803 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.742626 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.877332 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.03887 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.28692 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.03171 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.858723 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.990387 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.13224 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.21784 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.919312 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.793057 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.05647 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.28882 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.17971 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.04313 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.868121 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.805576 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.00284 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.00005 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.14488 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.06828 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.864874 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.18268 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.918622 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.23613 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.01988 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.92348 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.04048 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.1214 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.763269 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.908956 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.765846 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.297000065693051 z_k1_s0000
                                 - 0.261000077815109 z_k1_s0001
                                 - 0.147999950164874 z_k1_s0002
                                 - 0.738998922660959 z_k1_s0003
                                 - 0.710001417926356 z_k1_s0004
                                 - 0.462001501433864 z_k1_s0005
                                 - 0.363999877432528 z_k1_s0006
                                 - 0.333000716371102 z_k1_s0007
                                 - 0.92600048324212 z_k1_s0010
                                 - 1.21900068517066 z_k1_s0011
                                 - 1.90099826307282 z_k1_s0012
                                 - 0.750997816541516 z_k1_s0013
                                 - 0.435998766626416 z_k1_s0014
                                 - 0.613998318623304 z_k1_s0015
                                 - 0.585998890660773 z_k1_s0016
                                 - 1.13299817276758 z_k1_s0017
                                 - 0.142999977202156 z_k1_s0020
                                 - 0.118999893553596 z_k1_s0021
                                 - 0.473999271635518 z_k1_s0022
                                 - 0.235999857928098 z_k1_s0023
                                 - 0.11199995472688 z_k1_s0024
                                 - 0.163999980382189 z_k1_s0025
                                 - 0.121000009432576 z_k1_s0026
                                 - 0.497999541983964 z_k1_s0027
                                 - 0.258000251636861 z_k1_s0030
                                 - 0.839999682631026 z_k1_s0031
                                 - 0.278000321661452 z_k1_s0032
                                 - 0.288000356673747 z_k1_s0033
                                 - 0.274000226196489 z_k1_s0034
                                 - 0.356000350377221 z_k1_s0035
                                 - 0.733999800260963 z_k1_s0036
                                 - 1.04500202958396 z_k1_s0037
                                 - 1.02799685068691 z_k1_s0040
                                 - 1.94699504032109 z_k1_s0041
                                 - 0.768998624800127 z_k1_s0042
                                 - 0.89900023049374 z_k1_s0043
                                 - 0.737999048685991 z_k1_s0044
                                 - 1.36499572691027 z_k1_s0045
                                 - 0.503999445773236 z_k1_s0046
                                 - 0.576999432276218 z_k1_s0047
                                 - 0.737995501392084 z_k1_s0050
                                 - 0.7659972472585 z_k1_s0051
                                 - 0.358998406024435 z_k1_s0052
                                 - 0.857996589416161 z_k1_s0053
                                 - 0.455999658509097 z_k1_s0054
                                 - 0.364998460425639 z_k1_s0055
                                 - 0.316998920412487 z_k1_s0056
                                 - 0.616996269293804 z_k1_s0057
                                 - 0.362000153223117 z_k1_s0060
                                 - 0.318999911327548 z_k1_s0061
                                 - 0.721998570395233 z_k1_s0062
                                 - 0.677998217274183 z_k1_s0063
                                 - 0.46100094774548 z_k1_s0064
                                 - 0.260000160224339 z_k1_s0065
                                 - 0.664001200917564 z_k1_s0066
                                 - 0.318000178502076 z_k1_s0067
                                 - 1.41700051230321 z_k1_s0070
                                 - 0.794998216342447 z_k1_s0071
                                 - 0.301999630724824 z_k1_s0072
                                 - 1.04099937002449 z_k1_s0073
                                 - 0.950996929459403 z_k1_s0074
                                 - 0.406999719452829 z_k1_s0075
                                 - 0.400999768948331 z_k1_s0076
                                 - 1.05199980978234 z_k1_s0077
                                 - 0.0789997815742805 z_k1_s0100
                                 - 0.0489999752549273 z_k1_s0101
                                 - 0.0769998916863182 z_k1_s0102
                                 - 0.0239999908554744 z_k1_s0103
                                 - 0.0469999881669704 z_k1_s0104
                                 - 0.0849999372381403 z_k1_s0105
                                 - 0.0880000150700702 z_k1_s0106
                                 - 0.0370000041271887 z_k1_s0107
                                 - 0.255999880997027 z_k1_s0110
                                 - 0.115000067029266 z_k1_s0111
                                 - 0.122999895010238 z_k1_s0112
                                 - 0.184000449967201 z_k1_s0113
                                 - 0.136000013279755 z_k1_s0114
                                 - 0.147000113353961 z_k1_s0115
                                 - 0.127999940498514 z_k1_s0116
                                 - 0.135000004182099 z_k1_s0117
                                 - 0.0500000939544689 z_k1_s0120
                                 - 0.101000001307763 z_k1_s0121
                                 - 0.0470000957572112 z_k1_s0122
                                 - 0.173000082042122 z_k1_s0123
                                 - 0.0789998905277 z_k1_s0124
                                 - 0.0560000903489843 z_k1_s0125
                                 - 0.0580000891471562 z_k1_s0126
                                 - 0.0560000903489843 z_k1_s0127
                                 - 0.322999721633108 z_k1_s0130
                                 - 0.139000075919097 z_k1_s0131
                                 - 0.123999726155169 z_k1_s0132
                                 - 0.203999753400008 z_k1_s0133
                                 - 0.125999736923768 z_k1_s0134
                                 - 0.262999398575119 z_k1_s0135
                                 - 0.0949998390099051 z_k1_s0136
                                 - 0.070999844286425 z_k1_s0137
                                 - 0.217000711126117 z_k1_s0140
                                 - 0.114000140786695 z_k1_s0141
                                 - 0.121000092326156 z_k1_s0142
                                 - 0.1130002035097 z_k1_s0143
                                 - 0.0940000932451403 z_k1_s0144
                                 - 0.247000131437617 z_k1_s0145
                                 - 0.247000131437617 z_k1_s0146
                                 - 0.148999768283833 z_k1_s0147
                                 - 0.0739997477868471 z_k1_s0150
                                 - 0.0849994662154997 z_k1_s0151
                                 - 0.100999877654223 z_k1_s0152
                                 - 0.222999217763146 z_k1_s0153
                                 - 0.0539997671365235 z_k1_s0154
                                 - 0.0779997274969709 z_k1_s0155
                                 - 0.113999544888085 z_k1_s0156
                                 - 0.0919992870337332 z_k1_s0157
                                 - 0.0880000311427429 z_k1_s0160
                                 - 0.0909998895713716 z_k1_s0161
                                 - 0.145999852598046 z_k1_s0162
                                 - 0.0810000604757291 z_k1_s0163
                                 - 0.126999816116069 z_k1_s0164
                                 - 0.0640000801129447 z_k1_s0165
                                 - 0.190999986529078 z_k1_s0166
                                 - 0.11499984060117 z_k1_s0167
                                 - 0.19400043048831 z_k1_s0170
                                 - 0.174000501345204 z_k1_s0171
                                 - 0.0840000750011066 z_k1_s0172
                                 - 0.283999490632193 z_k1_s0173
                                 - 0.128000540116039 z_k1_s0174
                                 - 0.182000224602407 z_k1_s0175
                                 - 0.0620000287436697 z_k1_s0176
                                 - 0.133999401058795 z_k1_s0177
                                 - 0.505002622490585 z_k1_s1000
                                 - 0.209000579400894 z_k1_s1001
                                 - 0.276000697641698 z_k1_s1002
                                 - 0.495001949766588 z_k1_s1003
                                 - 0.303000632992214 z_k1_s1004
                                 - 0.424000308433337 z_k1_s1005
                                 - 0.616000684705575 z_k1_s1006
                                 - 0.3070005885811 z_k1_s1007
                                 - 0.487999265954158 z_k1_s1010
                                 - 0.608999740138071 z_k1_s1011
                                 - 0.322999306229131 z_k1_s1012
                                 - 0.458999449661493 z_k1_s1013
                                 - 0.626996896601584 z_k1_s1014
                                 - 0.240999627167547 z_k1_s1015
                                 - 0.544998656736839 z_k1_s1016
                                 - 0.305999153375382 z_k1_s1017
                                 - 0.0860000156077238 z_k1_s1020
                                 - 0.110000118279718 z_k1_s1021
                                 - 0.0889999877916935 z_k1_s1022
                                 - 0.271000739287632 z_k1_s1023
                                 - 0.357999932622754 z_k1_s1024
                                 - 0.215999623247003 z_k1_s1025
                                 - 0.246000971087884 z_k1_s1026
                                 - 0.250000933999844 z_k1_s1027
                                 - 0.786996431622918 z_k1_s1030
                                 - 0.402999071877542 z_k1_s1031
                                 - 0.456997339266323 z_k1_s1032
                                 - 0.22699969762041 z_k1_s1033
                                 - 1.09599631582808 z_k1_s1034
                                 - 0.354999242534688 z_k1_s1035
                                 - 0.354999242534688 z_k1_s1036
                                 - 0.570000364066631 z_k1_s1037
                                 - 0.591001457774514 z_k1_s1040
                                 - 0.402999747048716 z_k1_s1041
                                 - 0.594001720707289 z_k1_s1042
                                 - 1.19299716763826 z_k1_s1043
                                 - 1.15200028622128 z_k1_s1044
                                 - 0.475000016637457 z_k1_s1045
                                 - 0.257999959226678 z_k1_s1046
                                 - 0.367999867698544 z_k1_s1047
                                 - 0.625000897296175 z_k1_s1050
                                 - 0.348999466798356 z_k1_s1051
                                 - 0.135999999634729 z_k1_s1052
                                 - 0.519999642573748 z_k1_s1053
                                 - 0.179000133030349 z_k1_s1054
                                 - 0.155000019165328 z_k1_s1055
                                 - 0.273999262282757 z_k1_s1056
                                 - 0.183000111657828 z_k1_s1057
                                 - 1.16600328029471 z_k1_s1060
                                 - 0.659003334036025 z_k1_s1061
                                 - 0.730001467731071 z_k1_s1062
                                 - 0.600998553827894 z_k1_s1063
                                 - 0.936999842592421 z_k1_s1064
                                 - 0.772998512361388 z_k1_s1065
                                 - 0.699002365554447 z_k1_s1066
                                 - 0.325000524345891 z_k1_s1067
                                 - 0.295999102986675 z_k1_s1070
                                 - 0.922996590981914 z_k1_s1071
                                 - 0.791996869404385 z_k1_s1072
                                 - 0.220999200801252 z_k1_s1073
                                 - 0.698997313413488 z_k1_s1074
                                 - 0.264999116523448 z_k1_s1075
                                 - 0.409998389550219 z_k1_s1076
                                 - 0.427998043355692 z_k1_s1077
                                 - 0.151999906490306 z_k1_s1100
                                 - 0.19199937972482 z_k1_s1101
                                 - 0.196000017947965 z_k1_s1102
                                 - 0.203999649394985 z_k1_s1103
                                 - 0.26299919319075 z_k1_s1104
                                 - 0.316999584206855 z_k1_s1105
                                 - 0.132000005373122 z_k1_s1106
                                 - 0.189000546056731 z_k1_s1107
                                 - 0.291000058050307 z_k1_s1110
                                 - 0.401998827153335 z_k1_s1111
                                 - 0.130999892818325 z_k1_s1112
                                 - 0.083999991009395 z_k1_s1113
                                 - 0.109999936690931 z_k1_s1114
                                 - 0.257999793993336 z_k1_s1115
                                 - 0.129999894907496 z_k1_s1116
                                 - 0.259000457903439 z_k1_s1117
                                 - 0.252000624727767 z_k1_s1120
                                 - 0.467002382482973 z_k1_s1121
                                 - 0.214000496022692 z_k1_s1122
                                 - 0.430000617502505 z_k1_s1123
                                 - 0.308000280554841 z_k1_s1124
                                 - 0.245000776424679 z_k1_s1125
                                 - 0.609001872009758 z_k1_s1126
                                 - 0.372999699942915 z_k1_s1127
                                 - 0.26199977500597 z_k1_s1130
                                 - 0.524998581818462 z_k1_s1131
                                 - 0.194999810046455 z_k1_s1132
                                 - 0.238999646282723 z_k1_s1133
                                 - 0.186999893403433 z_k1_s1134
                                 - 0.413999959421271 z_k1_s1135
                                 - 0.800999537098136 z_k1_s1136
                                 - 0.323998686939917 z_k1_s1137
                                 - 0.87700269589 z_k1_s1140
                                 - 0.229000137977626 z_k1_s1141
                                 - 0.319000502693246 z_k1_s1142
                                 - 0.929002332053849 z_k1_s1143
                                 - 0.205000358240532 z_k1_s1144
                                 - 0.223000363143569 z_k1_s1145
                                 - 0.217000248109079 z_k1_s1146
                                 - 0.403001432775241 z_k1_s1147
                                 - 0.383999780226468 z_k1_s1150
                                 - 0.388999777364834 z_k1_s1151
                                 - 0.303999826012621 z_k1_s1152
                                 - 0.119999931320771 z_k1_s1153
                                 - 0.12599992788681 z_k1_s1154
                                 - 0.165999904993734 z_k1_s1155
                                 - 0.221999872943427 z_k1_s1156
                                 - 0.288999834597524 z_k1_s1157
                                 - 0.42599975090007 z_k1_s1160
                                 - 1.23600094339795 z_k1_s1161
                                 - 1.10899721485307 z_k1_s1162
                                 - 0.673997122103464 z_k1_s1163
                                 - 0.683999692599987 z_k1_s1164
                                 - 0.602999244750642 z_k1_s1165
                                 - 1.37099621315425 z_k1_s1166
                                 - 0.468999631650204 z_k1_s1167
                                 - 0.117000039116586 z_k1_s1170
                                 - 0.0970000438282785 z_k1_s1171
                                 - 0.323999723630621 z_k1_s1172
                                 - 0.210000283927164 z_k1_s1173
                                 - 0.0879999125885662 z_k1_s1174
                                 - 0.141000000122562 z_k1_s1175
                                 - 0.387999175113309 z_k1_s1176
                                 - 0.302000328933365 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.825791 w_k1_ADM_s000
                                 + 0.879985 w_k1_AXP_s000
                                 + 0.913146 w_k1_DIS_s000
                                 + 0.800551 w_k1_FDX_s000
                                 + 1.46322 w_k1_IBM_s000
                                 + 0.914391 w_k1_INTC_s000
                                 + 1.02338 w_k1_KO_s000 + 1.57713 w_k1_MAT_s000
                                 + 0.894178 w_k1_MSFT_s000
                                 + 0.905805 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.04248 w_k1_ADM_s000
                                 + 1.15853 w_k1_AXP_s000
                                 + 1.45849 w_k1_DIS_s000
                                 + 1.27873 w_k1_FDX_s000
                                 + 0.941221 w_k1_IBM_s000
                                 + 0.943545 w_k1_INTC_s000
                                 + 1.05041 w_k1_KO_s000 + 1.03598 w_k1_MAT_s000
                                 + 1.19441 w_k1_MSFT_s000
                                 + 1.07044 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.02065 w_k1_ADM_s000
                                 + 1.01385 w_k1_AXP_s000
                                 + 1.18832 w_k1_DIS_s000 + 1.1985 w_k1_FDX_s000
                                 + 0.670584 w_k1_IBM_s000
                                 + 1.11109 w_k1_INTC_s000
                                 + 1.03625 w_k1_KO_s000
                                 + 0.951245 w_k1_MAT_s000
                                 + 1.26084 w_k1_MSFT_s000
                                 + 1.06101 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.02276 w_k1_ADM_s000
                                 + 1.20034 w_k1_AXP_s000
                                 + 0.95509 w_k1_DIS_s000
                                 + 0.94167 w_k1_FDX_s000
                                 + 1.09731 w_k1_IBM_s000
                                 + 0.823684 w_k1_INTC_s000
                                 + 1.02569 w_k1_KO_s000 + 1.16465 w_k1_MAT_s000
                                 + 0.973666 w_k1_MSFT_s000
                                 + 0.985554 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.977756 w_k1_ADM_s000
                                 + 1.0262 w_k1_AXP_s000
                                 + 0.864388 w_k1_DIS_s000
                                 + 1.04539 w_k1_FDX_s000
                                 + 0.675671 w_k1_IBM_s000
                                 + 1.13874 w_k1_INTC_s000
                                 + 0.839613 w_k1_KO_s000
                                 + 0.801963 w_k1_MAT_s000
                                 + 1.00128 w_k1_MSFT_s000
                                 + 0.891968 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.661434 w_k1_ADM_s000
                                 + 0.74847 w_k1_AXP_s000 + 0.6809 w_k1_DIS_s000
                                 + 0.691078 w_k1_FDX_s000
                                 + 0.72021 w_k1_IBM_s000
                                 + 0.754729 w_k1_INTC_s000
                                 + 0.740486 w_k1_KO_s000
                                 + 0.856794 w_k1_MAT_s000
                                 + 0.898212 w_k1_MSFT_s000
                                 + 0.814451 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.35763 w_k1_ADM_s000
                                 + 1.75122 w_k1_AXP_s000
                                 + 1.12743 w_k1_DIS_s000 + 1.2601 w_k1_FDX_s000
                                 + 1.07937 w_k1_IBM_s000
                                 + 1.2511 w_k1_INTC_s000 + 1.15519 w_k1_KO_s000
                                 + 1.90607 w_k1_MAT_s000
                                 + 1.13169 w_k1_MSFT_s000
                                 + 1.22025 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.940264 w_k1_ADM_s000
                                 + 1.37369 w_k1_AXP_s000
                                 + 1.58161 w_k1_DIS_s000
                                 + 1.22076 w_k1_FDX_s000
                                 + 1.29519 w_k1_IBM_s000
                                 + 1.22732 w_k1_INTC_s000
                                 + 1.22581 w_k1_KO_s000 + 1.14191 w_k1_MAT_s000
                                 + 1.23415 w_k1_MSFT_s000
                                 + 1.18674 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.13896 w_k1_ADM_s001
                                 + 1.4076 w_k1_AXP_s001 + 1.57459 w_k1_DIS_s001
                                 + 1.346 w_k1_FDX_s001 + 1.0389 w_k1_IBM_s001
                                 + 1.21346 w_k1_INTC_s001
                                 + 1.20253 w_k1_KO_s001 + 1.0884 w_k1_MAT_s001
                                 + 1.39696 w_k1_MSFT_s001 + 1.0914 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.92073 w_k1_ADM_s001
                                 + 0.7823 w_k1_AXP_s001
                                 + 0.795766 w_k1_DIS_s001
                                 + 0.839421 w_k1_FDX_s001
                                 + 1.0385 w_k1_IBM_s001
                                 + 0.74831 w_k1_INTC_s001
                                 + 0.943886 w_k1_KO_s001
                                 + 1.00167 w_k1_MAT_s001
                                 + 0.919102 w_k1_MSFT_s001
                                 + 0.987286 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.886037 w_k1_ADM_s001
                                 + 1.20617 w_k1_AXP_s001
                                 + 1.14123 w_k1_DIS_s001
                                 + 1.08791 w_k1_FDX_s001
                                 + 1.22795 w_k1_IBM_s001
                                 + 0.97065 w_k1_INTC_s001
                                 + 1.17429 w_k1_KO_s001 + 1.18038 w_k1_MAT_s001
                                 + 1.08435 w_k1_MSFT_s001
                                 + 1.17217 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.11029 w_k1_ADM_s001
                                 + 1.10895 w_k1_AXP_s001
                                 + 0.843342 w_k1_DIS_s001
                                 + 1.01416 w_k1_FDX_s001
                                 + 0.897042 w_k1_IBM_s001
                                 + 0.910678 w_k1_INTC_s001
                                 + 0.736252 w_k1_KO_s001
                                 + 0.96939 w_k1_MAT_s001
                                 + 0.864493 w_k1_MSFT_s001
                                 + 0.930913 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.28447 w_k1_ADM_s001
                                 + 1.47859 w_k1_AXP_s001
                                 + 1.43174 w_k1_DIS_s001
                                 + 1.22338 w_k1_FDX_s001
                                 + 1.89204 w_k1_IBM_s001
                                 + 1.20196 w_k1_INTC_s001
                                 + 1.40208 w_k1_KO_s001 + 2.08273 w_k1_MAT_s001
                                 + 1.27418 w_k1_MSFT_s001
                                 + 1.26367 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.07341 w_k1_ADM_s001
                                 + 0.883602 w_k1_AXP_s001
                                 + 1.07228 w_k1_DIS_s001
                                 + 1.09393 w_k1_FDX_s001
                                 + 0.645809 w_k1_IBM_s001
                                 + 0.902946 w_k1_INTC_s001
                                 + 0.790575 w_k1_KO_s001
                                 + 0.727389 w_k1_MAT_s001
                                 + 1.10857 w_k1_MSFT_s001
                                 + 0.88053 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.806628 w_k1_ADM_s001
                                 + 0.871323 w_k1_AXP_s001
                                 + 0.719449 w_k1_DIS_s001
                                 + 0.741632 w_k1_FDX_s001
                                 + 1.16606 w_k1_IBM_s001
                                 + 0.761177 w_k1_INTC_s001
                                 + 0.897279 w_k1_KO_s001
                                 + 1.38359 w_k1_MAT_s001
                                 + 0.883395 w_k1_MSFT_s001
                                 + 0.875765 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.897875 w_k1_ADM_s001
                                 + 0.847996 w_k1_AXP_s001
                                 + 0.643705 w_k1_DIS_s001
                                 + 0.851724 w_k1_FDX_s001
                                 + 0.667818 w_k1_IBM_s001
                                 + 0.635105 w_k1_INTC_s001
                                 + 0.54524 w_k1_KO_s001
                                 + 0.587292 w_k1_MAT_s001
                                 + 0.790895 w_k1_MSFT_s001
                                 + 0.859997 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.944873 w_k1_ADM_s002
                                 + 0.750439 w_k1_AXP_s002
                                 + 0.580624 w_k1_DIS_s002
                                 + 0.688594 w_k1_FDX_s002
                                 + 0.769025 w_k1_IBM_s002
                                 + 0.887672 w_k1_INTC_s002
                                 + 0.787403 w_k1_KO_s002
                                 + 0.64155 w_k1_MAT_s002
                                 + 0.859873 w_k1_MSFT_s002
                                 + 0.813206 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.27077 w_k1_ADM_s002
                                 + 1.18403 w_k1_AXP_s002
                                 + 1.52834 w_k1_DIS_s002
                                 + 1.29428 w_k1_FDX_s002
                                 + 1.16301 w_k1_IBM_s002
                                 + 1.18376 w_k1_INTC_s002
                                 + 1.36666 w_k1_KO_s002
                                 + 0.978181 w_k1_MAT_s002
                                 + 1.26668 w_k1_MSFT_s002
                                 + 1.20279 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.05563 w_k1_ADM_s002
                                 + 1.2361 w_k1_AXP_s002 + 0.97092 w_k1_DIS_s002
                                 + 0.966123 w_k1_FDX_s002
                                 + 1.02947 w_k1_IBM_s002
                                 + 1.04798 w_k1_INTC_s002
                                 + 1.01102 w_k1_KO_s002 + 0.93457 w_k1_MAT_s002
                                 + 0.909512 w_k1_MSFT_s002
                                 + 1.06618 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.15086 w_k1_ADM_s002
                                 + 1.13473 w_k1_AXP_s002
                                 + 1.19346 w_k1_DIS_s002
                                 + 1.08101 w_k1_FDX_s002
                                 + 1.29975 w_k1_IBM_s002
                                 + 1.28895 w_k1_INTC_s002
                                 + 1.27866 w_k1_KO_s002
                                 + 0.963294 w_k1_MAT_s002
                                 + 1.06406 w_k1_MSFT_s002
                                 + 1.14416 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.26887 w_k1_ADM_s002
                                 + 1.46116 w_k1_AXP_s002
                                 + 1.27693 w_k1_DIS_s002
                                 + 1.17893 w_k1_FDX_s002
                                 + 1.64838 w_k1_IBM_s002
                                 + 1.25701 w_k1_INTC_s002
                                 + 1.45264 w_k1_KO_s002 + 1.65852 w_k1_MAT_s002
                                 + 1.08304 w_k1_MSFT_s002 + 1.1721 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.835337 w_k1_ADM_s002
                                 + 0.801716 w_k1_AXP_s002
                                 + 0.611411 w_k1_DIS_s002
                                 + 0.725209 w_k1_FDX_s002
                                 + 1.03595 w_k1_IBM_s002
                                 + 0.904258 w_k1_INTC_s002
                                 + 0.790119 w_k1_KO_s002
                                 + 0.87072 w_k1_MAT_s002
                                 + 0.734913 w_k1_MSFT_s002
                                 + 0.859379 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.909221 w_k1_ADM_s002
                                 + 0.870296 w_k1_AXP_s002
                                 + 0.666848 w_k1_DIS_s002
                                 + 1.0466 w_k1_FDX_s002 + 0.53413 w_k1_IBM_s002
                                 + 1.07944 w_k1_INTC_s002
                                 + 0.72303 w_k1_KO_s002
                                 + 0.577252 w_k1_MAT_s002
                                 + 0.893614 w_k1_MSFT_s002
                                 + 0.878388 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.05591 w_k1_ADM_s002
                                 + 0.93588 w_k1_AXP_s002
                                 + 0.917151 w_k1_DIS_s002
                                 + 1.04151 w_k1_FDX_s002
                                 + 0.75425 w_k1_IBM_s002
                                 + 1.0405 w_k1_INTC_s002 + 0.91489 w_k1_KO_s002
                                 + 0.736006 w_k1_MAT_s002
                                 + 0.858827 w_k1_MSFT_s002
                                 + 1.01924 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.913843 w_k1_ADM_s003
                                 + 0.902899 w_k1_AXP_s003
                                 + 0.664526 w_k1_DIS_s003
                                 + 0.764311 w_k1_FDX_s003
                                 + 1.06759 w_k1_IBM_s003
                                 + 0.876548 w_k1_INTC_s003
                                 + 0.866359 w_k1_KO_s003
                                 + 1.35413 w_k1_MAT_s003
                                 + 0.721926 w_k1_MSFT_s003
                                 + 0.933091 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.83221 w_k1_ADM_s003
                                 + 0.719826 w_k1_AXP_s003
                                 + 0.815615 w_k1_DIS_s003
                                 + 0.817337 w_k1_FDX_s003
                                 + 0.780668 w_k1_IBM_s003
                                 + 1.08523 w_k1_INTC_s003
                                 + 0.919093 w_k1_KO_s003
                                 + 0.793237 w_k1_MAT_s003
                                 + 1.03502 w_k1_MSFT_s003
                                 + 0.883571 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.32737 w_k1_ADM_s003
                                 + 1.43711 w_k1_AXP_s003
                                 + 1.78932 w_k1_DIS_s003
                                 + 1.54775 w_k1_FDX_s003
                                 + 1.15275 w_k1_IBM_s003
                                 + 1.36161 w_k1_INTC_s003
                                 + 1.17631 w_k1_KO_s003 + 1.50732 w_k1_MAT_s003
                                 + 1.32035 w_k1_MSFT_s003
                                 + 1.13944 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.30898 w_k1_ADM_s003
                                 + 1.51412 w_k1_AXP_s003
                                 + 1.31523 w_k1_DIS_s003
                                 + 1.08051 w_k1_FDX_s003
                                 + 1.80577 w_k1_IBM_s003
                                 + 1.08909 w_k1_INTC_s003
                                 + 1.39881 w_k1_KO_s003 + 2.1171 w_k1_MAT_s003
                                 + 0.997643 w_k1_MSFT_s003
                                 + 1.07788 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.01633 w_k1_ADM_s003
                                 + 0.570484 w_k1_AXP_s003
                                 + 0.576542 w_k1_DIS_s003
                                 + 0.76982 w_k1_FDX_s003
                                 + 0.562508 w_k1_IBM_s003
                                 + 0.846929 w_k1_INTC_s003
                                 + 0.718537 w_k1_KO_s003
                                 + 0.725728 w_k1_MAT_s003
                                 + 0.898559 w_k1_MSFT_s003
                                 + 0.862046 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.722554 w_k1_ADM_s003
                                 + 1.10385 w_k1_AXP_s003
                                 + 0.973099 w_k1_DIS_s003
                                 + 1.02581 w_k1_FDX_s003
                                 + 1.13499 w_k1_IBM_s003
                                 + 1.16999 w_k1_INTC_s003
                                 + 0.999366 w_k1_KO_s003
                                 + 2.06981 w_k1_MAT_s003
                                 + 1.07942 w_k1_MSFT_s003
                                 + 1.01068 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.37518 w_k1_ADM_s003
                                 + 1.08354 w_k1_AXP_s003
                                 + 0.997725 w_k1_DIS_s003
                                 + 1.1478 w_k1_FDX_s003
                                 + 0.812263 w_k1_IBM_s003
                                 + 1.22062 w_k1_INTC_s003
                                 + 0.887627 w_k1_KO_s003
                                 + 0.921121 w_k1_MAT_s003
                                 + 1.08804 w_k1_MSFT_s003
                                 + 1.17301 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.02837 w_k1_ADM_s003
                                 + 0.954357 w_k1_AXP_s003
                                 + 1.07738 w_k1_DIS_s003
                                 + 1.00804 w_k1_FDX_s003
                                 + 1.23132 w_k1_IBM_s003
                                 + 0.989325 w_k1_INTC_s003
                                 + 0.979276 w_k1_KO_s003
                                 + 1.16965 w_k1_MAT_s003
                                 + 1.04028 w_k1_MSFT_s003
                                 + 1.11966 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.924712 w_k1_ADM_s004
                                 + 0.898463 w_k1_AXP_s004
                                 + 0.934701 w_k1_DIS_s004
                                 + 0.918454 w_k1_FDX_s004
                                 + 1.05733 w_k1_IBM_s004
                                 + 0.88361 w_k1_INTC_s004
                                 + 0.905199 w_k1_KO_s004
                                 + 1.18469 w_k1_MAT_s004
                                 + 1.01991 w_k1_MSFT_s004
                                 + 1.04192 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.849329 w_k1_ADM_s004
                                 + 0.705013 w_k1_AXP_s004
                                 + 0.689016 w_k1_DIS_s004
                                 + 0.784454 w_k1_FDX_s004
                                 + 0.768983 w_k1_IBM_s004
                                 + 0.65106 w_k1_INTC_s004
                                 + 0.692345 w_k1_KO_s004
                                 + 0.872216 w_k1_MAT_s004
                                 + 0.749894 w_k1_MSFT_s004
                                 + 0.875173 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.09118 w_k1_ADM_s004
                                 + 1.27214 w_k1_AXP_s004
                                 + 0.965311 w_k1_DIS_s004
                                 + 0.967886 w_k1_FDX_s004
                                 + 1.25084 w_k1_IBM_s004
                                 + 0.924042 w_k1_INTC_s004
                                 + 1.08828 w_k1_KO_s004 + 1.56202 w_k1_MAT_s004
                                 + 0.958981 w_k1_MSFT_s004
                                 + 1.01884 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.15976 w_k1_ADM_s004
                                 + 1.17646 w_k1_AXP_s004
                                 + 0.893247 w_k1_DIS_s004
                                 + 1.01823 w_k1_FDX_s004
                                 + 0.799658 w_k1_IBM_s004
                                 + 1.00222 w_k1_INTC_s004
                                 + 0.849156 w_k1_KO_s004
                                 + 0.89149 w_k1_MAT_s004
                                 + 1.08475 w_k1_MSFT_s004
                                 + 1.02501 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.51587 w_k1_ADM_s004
                                 + 1.44544 w_k1_AXP_s004
                                 + 1.73611 w_k1_DIS_s004
                                 + 1.26287 w_k1_FDX_s004
                                 + 1.35894 w_k1_IBM_s004
                                 + 1.09792 w_k1_INTC_s004
                                 + 1.40475 w_k1_KO_s004 + 1.58045 w_k1_MAT_s004
                                 + 1.45397 w_k1_MSFT_s004 + 1.4166 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.08199 w_k1_ADM_s004
                                 + 1.03879 w_k1_AXP_s004
                                 + 1.38355 w_k1_DIS_s004 + 1.1552 w_k1_FDX_s004
                                 + 0.920434 w_k1_IBM_s004
                                 + 0.952142 w_k1_INTC_s004
                                 + 1.11407 w_k1_KO_s004
                                 + 0.936381 w_k1_MAT_s004
                                 + 1.1669 w_k1_MSFT_s004 + 1.11647 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.887122 w_k1_ADM_s004
                                 + 1.54845 w_k1_AXP_s004
                                 + 1.28351 w_k1_DIS_s004
                                 + 1.04759 w_k1_FDX_s004
                                 + 1.71791 w_k1_IBM_s004
                                 + 0.885717 w_k1_INTC_s004
                                 + 1.17135 w_k1_KO_s004 + 2.03748 w_k1_MAT_s004
                                 + 1.13932 w_k1_MSFT_s004
                                 + 1.24477 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.05294 w_k1_ADM_s004
                                 + 1.0058 w_k1_AXP_s004 + 1.08442 w_k1_DIS_s004
                                 + 0.999992 w_k1_FDX_s004
                                 + 1.3151 w_k1_IBM_s004
                                 + 1.01903 w_k1_INTC_s004
                                 + 1.19488 w_k1_KO_s004 + 1.57922 w_k1_MAT_s004
                                 + 1.12108 w_k1_MSFT_s004
                                 + 1.12305 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.09874 w_k1_ADM_s005
                                 + 1.28031 w_k1_AXP_s005
                                 + 0.980994 w_k1_DIS_s005
                                 + 1.14468 w_k1_FDX_s005
                                 + 1.04212 w_k1_IBM_s005
                                 + 1.19128 w_k1_INTC_s005
                                 + 0.951594 w_k1_KO_s005
                                 + 0.989418 w_k1_MAT_s005
                                 + 1.08451 w_k1_MSFT_s005
                                 + 0.958935 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.11443 w_k1_ADM_s005
                                 + 1.27832 w_k1_AXP_s005
                                 + 1.50371 w_k1_DIS_s005
                                 + 1.29735 w_k1_FDX_s005
                                 + 1.30793 w_k1_IBM_s005
                                 + 1.62492 w_k1_INTC_s005
                                 + 1.17165 w_k1_KO_s005 + 1.07913 w_k1_MAT_s005
                                 + 1.21086 w_k1_MSFT_s005
                                 + 1.10529 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.996748 w_k1_ADM_s005
                                 + 0.785719 w_k1_AXP_s005
                                 + 0.99714 w_k1_DIS_s005
                                 + 0.931605 w_k1_FDX_s005
                                 + 0.995153 w_k1_IBM_s005
                                 + 1.11271 w_k1_INTC_s005
                                 + 1.09492 w_k1_KO_s005
                                 + 0.896165 w_k1_MAT_s005
                                 + 1.02859 w_k1_MSFT_s005
                                 + 1.02006 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.94767 w_k1_ADM_s005
                                 + 0.999403 w_k1_AXP_s005
                                 + 0.814074 w_k1_DIS_s005
                                 + 0.817464 w_k1_FDX_s005
                                 + 1.26183 w_k1_IBM_s005
                                 + 1.0234 w_k1_INTC_s005 + 0.95013 w_k1_KO_s005
                                 + 1.09474 w_k1_MAT_s005
                                 + 0.793783 w_k1_MSFT_s005
                                 + 0.885029 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.17893 w_k1_ADM_s005
                                 + 0.86336 w_k1_AXP_s005
                                 + 0.89899 w_k1_DIS_s005
                                 + 1.11272 w_k1_FDX_s005
                                 + 0.796022 w_k1_IBM_s005
                                 + 1.11256 w_k1_INTC_s005
                                 + 0.877566 w_k1_KO_s005
                                 + 0.92076 w_k1_MAT_s005
                                 + 1.06392 w_k1_MSFT_s005
                                 + 0.907147 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.75468 w_k1_ADM_s005
                                 + 0.873614 w_k1_AXP_s005
                                 + 0.824303 w_k1_DIS_s005
                                 + 0.899549 w_k1_FDX_s005
                                 + 0.807863 w_k1_IBM_s005
                                 + 1.1177 w_k1_INTC_s005
                                 + 0.775707 w_k1_KO_s005
                                 + 0.668416 w_k1_MAT_s005
                                 + 0.943842 w_k1_MSFT_s005
                                 + 0.816164 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.92209 w_k1_ADM_s005
                                 + 0.692872 w_k1_AXP_s005
                                 + 0.668866 w_k1_DIS_s005
                                 + 0.775259 w_k1_FDX_s005
                                 + 0.724778 w_k1_IBM_s005
                                 + 0.959661 w_k1_INTC_s005
                                 + 0.682724 w_k1_KO_s005
                                 + 0.655943 w_k1_MAT_s005
                                 + 0.839129 w_k1_MSFT_s005
                                 + 0.816302 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.21705 w_k1_ADM_s005
                                 + 1.33881 w_k1_AXP_s005
                                 + 1.34552 w_k1_DIS_s005
                                 + 1.33046 w_k1_FDX_s005
                                 + 0.917602 w_k1_IBM_s005
                                 + 1.35817 w_k1_INTC_s005
                                 + 1.09832 w_k1_KO_s005 + 1.07266 w_k1_MAT_s005
                                 + 1.34034 w_k1_MSFT_s005
                                 + 1.08216 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.763187 w_k1_ADM_s006
                                 + 0.848849 w_k1_AXP_s006
                                 + 0.87564 w_k1_DIS_s006
                                 + 0.90652 w_k1_FDX_s006
                                 + 0.954186 w_k1_IBM_s006
                                 + 0.863371 w_k1_INTC_s006
                                 + 0.878933 w_k1_KO_s006
                                 + 1.17709 w_k1_MAT_s006
                                 + 0.966338 w_k1_MSFT_s006
                                 + 0.855486 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.73509 w_k1_ADM_s006
                                 + 0.955929 w_k1_AXP_s006
                                 + 0.979703 w_k1_DIS_s006
                                 + 0.933751 w_k1_FDX_s006
                                 + 1.44998 w_k1_IBM_s006
                                 + 0.88802 w_k1_INTC_s006
                                 + 1.15351 w_k1_KO_s006 + 1.86872 w_k1_MAT_s006
                                 + 0.839916 w_k1_MSFT_s006
                                 + 0.954845 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.04328 w_k1_ADM_s006
                                 + 0.919586 w_k1_AXP_s006
                                 + 1.18311 w_k1_DIS_s006
                                 + 1.12059 w_k1_FDX_s006
                                 + 0.839123 w_k1_IBM_s006
                                 + 1.05015 w_k1_INTC_s006
                                 + 0.911989 w_k1_KO_s006
                                 + 0.842991 w_k1_MAT_s006
                                 + 1.16356 w_k1_MSFT_s006
                                 + 0.984426 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.98776 w_k1_ADM_s006
                                 + 1.33791 w_k1_AXP_s006
                                 + 1.47912 w_k1_DIS_s006
                                 + 1.19805 w_k1_FDX_s006
                                 + 1.35701 w_k1_IBM_s006
                                 + 1.08347 w_k1_INTC_s006
                                 + 1.34309 w_k1_KO_s006 + 1.81477 w_k1_MAT_s006
                                 + 1.27672 w_k1_MSFT_s006
                                 + 1.12289 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.887254 w_k1_ADM_s006
                                 + 0.669179 w_k1_AXP_s006
                                 + 0.714214 w_k1_DIS_s006
                                 + 0.794581 w_k1_FDX_s006
                                 + 0.829991 w_k1_IBM_s006
                                 + 0.841095 w_k1_INTC_s006
                                 + 0.841215 w_k1_KO_s006
                                 + 0.732114 w_k1_MAT_s006
                                 + 0.814643 w_k1_MSFT_s006
                                 + 0.856562 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.44333 w_k1_ADM_s006
                                 + 1.28454 w_k1_AXP_s006
                                 + 1.48466 w_k1_DIS_s006
                                 + 1.44816 w_k1_FDX_s006
                                 + 0.914765 w_k1_IBM_s006
                                 + 1.12117 w_k1_INTC_s006
                                 + 1.17419 w_k1_KO_s006 + 1.10377 w_k1_MAT_s006
                                 + 1.25226 w_k1_MSFT_s006
                                 + 1.12371 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.03279 w_k1_ADM_s006
                                 + 0.979845 w_k1_AXP_s006
                                 + 0.81473 w_k1_DIS_s006
                                 + 0.989492 w_k1_FDX_s006
                                 + 0.839348 w_k1_IBM_s006
                                 + 0.800564 w_k1_INTC_s006
                                 + 0.718291 w_k1_KO_s006
                                 + 0.913552 w_k1_MAT_s006
                                 + 0.937148 w_k1_MSFT_s006
                                 + 0.872358 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.708906 w_k1_ADM_s006
                                 + 0.699392 w_k1_AXP_s006
                                 + 0.815841 w_k1_DIS_s006
                                 + 0.835732 w_k1_FDX_s006
                                 + 1.05888 w_k1_IBM_s006
                                 + 0.777483 w_k1_INTC_s006
                                 + 0.968426 w_k1_KO_s006
                                 + 1.20739 w_k1_MAT_s006
                                 + 0.834969 w_k1_MSFT_s006
                                 + 0.913554 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.31026 w_k1_ADM_s007
                                 + 1.21632 w_k1_AXP_s007
                                 + 1.37027 w_k1_DIS_s007
                                 + 1.24324 w_k1_FDX_s007
                                 + 0.827294 w_k1_IBM_s007
                                 + 1.11193 w_k1_INTC_s007
                                 + 0.949005 w_k1_KO_s007
                                 + 0.817618 w_k1_MAT_s007
                                 + 1.10808 w_k1_MSFT_s007
                                 + 1.00557 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.06159 w_k1_ADM_s007
                                 + 1.05472 w_k1_AXP_s007
                                 + 1.12281 w_k1_DIS_s007
                                 + 0.971147 w_k1_FDX_s007
                                 + 1.49038 w_k1_IBM_s007
                                 + 1.1064 w_k1_INTC_s007 + 1.15676 w_k1_KO_s007
                                 + 1.22199 w_k1_MAT_s007
                                 + 0.927442 w_k1_MSFT_s007
                                 + 0.957598 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.957349 w_k1_ADM_s007
                                 + 1.04523 w_k1_AXP_s007
                                 + 0.711216 w_k1_DIS_s007
                                 + 0.910507 w_k1_FDX_s007
                                 + 0.802039 w_k1_IBM_s007
                                 + 0.897131 w_k1_INTC_s007
                                 + 0.689118 w_k1_KO_s007
                                 + 0.984175 w_k1_MAT_s007
                                 + 0.707456 w_k1_MSFT_s007
                                 + 0.907509 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.974122 w_k1_ADM_s007
                                 + 1.48974 w_k1_AXP_s007
                                 + 1.10738 w_k1_DIS_s007
                                 + 1.16452 w_k1_FDX_s007
                                 + 1.27424 w_k1_IBM_s007
                                 + 1.12882 w_k1_INTC_s007
                                 + 0.997085 w_k1_KO_s007
                                 + 1.31633 w_k1_MAT_s007
                                 + 1.01681 w_k1_MSFT_s007
                                 + 0.965283 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.18028 w_k1_ADM_s007
                                 + 0.978927 w_k1_AXP_s007
                                 + 0.924347 w_k1_DIS_s007
                                 + 0.950091 w_k1_FDX_s007
                                 + 1.04869 w_k1_IBM_s007
                                 + 0.908133 w_k1_INTC_s007
                                 + 0.95068 w_k1_KO_s007 + 1.03002 w_k1_MAT_s007
                                 + 0.933193 w_k1_MSFT_s007
                                 + 0.893093 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.841398 w_k1_ADM_s007
                                 + 0.683673 w_k1_AXP_s007
                                 + 0.724586 w_k1_DIS_s007
                                 + 0.956132 w_k1_FDX_s007
                                 + 0.632981 w_k1_IBM_s007
                                 + 1.0014 w_k1_INTC_s007
                                 + 0.678566 w_k1_KO_s007
                                 + 0.505328 w_k1_MAT_s007
                                 + 0.835513 w_k1_MSFT_s007
                                 + 0.861583 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.989559 w_k1_ADM_s007
                                 + 0.999778 w_k1_AXP_s007
                                 + 0.875327 w_k1_DIS_s007
                                 + 1.07761 w_k1_FDX_s007
                                 + 0.822363 w_k1_IBM_s007
                                 + 0.952557 w_k1_INTC_s007
                                 + 0.824286 w_k1_KO_s007
                                 + 1.06076 w_k1_MAT_s007
                                 + 0.86367 w_k1_MSFT_s007
                                 + 0.940736 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.738089 w_k1_ADM_s007
                                 + 0.805872 w_k1_AXP_s007
                                 + 0.65746 w_k1_DIS_s007
                                 + 0.760865 w_k1_FDX_s007
                                 + 0.935314 w_k1_IBM_s007
                                 + 0.807959 w_k1_INTC_s007
                                 + 0.66925 w_k1_KO_s007
                                 + 0.683762 w_k1_MAT_s007
                                 + 0.685648 w_k1_MSFT_s007
                                 + 0.768472 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.02994 w_k1_ADM_s010
                                 + 1.33323 w_k1_AXP_s010
                                 + 1.61938 w_k1_DIS_s010
                                 + 1.11695 w_k1_FDX_s010 + 1.3103 w_k1_IBM_s010
                                 + 1.02102 w_k1_INTC_s010
                                 + 1.15254 w_k1_KO_s010 + 1.36708 w_k1_MAT_s010
                                 + 1.06882 w_k1_MSFT_s010
                                 + 1.00483 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.873336 w_k1_ADM_s010
                                 + 1.02687 w_k1_AXP_s010
                                 + 1.11751 w_k1_DIS_s010 + 1.021 w_k1_FDX_s010
                                 + 0.985862 w_k1_IBM_s010
                                 + 0.953636 w_k1_INTC_s010
                                 + 0.90546 w_k1_KO_s010 + 1.08658 w_k1_MAT_s010
                                 + 0.982913 w_k1_MSFT_s010
                                 + 0.957173 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.748537 w_k1_ADM_s010
                                 + 0.799634 w_k1_AXP_s010
                                 + 0.775552 w_k1_DIS_s010
                                 + 0.874067 w_k1_FDX_s010
                                 + 0.974279 w_k1_IBM_s010
                                 + 0.93662 w_k1_INTC_s010
                                 + 0.838903 w_k1_KO_s010
                                 + 1.03376 w_k1_MAT_s010
                                 + 0.920343 w_k1_MSFT_s010
                                 + 0.820916 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.438255 w_k1_ADM_s010
                                 + 0.721665 w_k1_AXP_s010
                                 + 0.649328 w_k1_DIS_s010
                                 + 0.724998 w_k1_FDX_s010
                                 + 0.74527 w_k1_IBM_s010
                                 + 0.910907 w_k1_INTC_s010
                                 + 0.734544 w_k1_KO_s010
                                 + 0.749064 w_k1_MAT_s010
                                 + 0.888541 w_k1_MSFT_s010
                                 + 0.79863 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.763717 w_k1_ADM_s010
                                 + 0.95639 w_k1_AXP_s010
                                 + 0.683301 w_k1_DIS_s010
                                 + 0.902852 w_k1_FDX_s010
                                 + 0.608699 w_k1_IBM_s010
                                 + 1.01053 w_k1_INTC_s010
                                 + 0.567714 w_k1_KO_s010 + 0.5103 w_k1_MAT_s010
                                 + 0.881939 w_k1_MSFT_s010
                                 + 0.724075 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.978827 w_k1_ADM_s010
                                 + 1.01606 w_k1_AXP_s010
                                 + 1.19638 w_k1_DIS_s010
                                 + 0.940195 w_k1_FDX_s010
                                 + 1.49459 w_k1_IBM_s010
                                 + 1.61282 w_k1_INTC_s010
                                 + 1.25907 w_k1_KO_s010 + 1.06585 w_k1_MAT_s010
                                 + 1.09256 w_k1_MSFT_s010
                                 + 0.978849 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.915497 w_k1_ADM_s010
                                 + 1.24269 w_k1_AXP_s010
                                 + 0.988152 w_k1_DIS_s010
                                 + 1.11112 w_k1_FDX_s010
                                 + 0.72614 w_k1_IBM_s010
                                 + 1.1071 w_k1_INTC_s010
                                 + 0.817079 w_k1_KO_s010
                                 + 0.967672 w_k1_MAT_s010
                                 + 0.937043 w_k1_MSFT_s010
                                 + 1.00957 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.39812 w_k1_ADM_s010
                                 + 1.91503 w_k1_AXP_s010
                                 + 1.61298 w_k1_DIS_s010
                                 + 1.36907 w_k1_FDX_s010
                                 + 0.878419 w_k1_IBM_s010
                                 + 1.0378 w_k1_INTC_s010
                                 + 0.972594 w_k1_KO_s010
                                 + 0.885152 w_k1_MAT_s010
                                 + 1.01644 w_k1_MSFT_s010
                                 + 1.09095 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.949744 w_k1_ADM_s011
                                 + 1.07405 w_k1_AXP_s011
                                 + 0.976818 w_k1_DIS_s011
                                 + 1.04877 w_k1_FDX_s011
                                 + 0.923425 w_k1_IBM_s011
                                 + 0.982624 w_k1_INTC_s011
                                 + 1.01094 w_k1_KO_s011 + 1.36838 w_k1_MAT_s011
                                 + 1.08332 w_k1_MSFT_s011
                                 + 1.06092 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.11843 w_k1_ADM_s011
                                 + 1.74341 w_k1_AXP_s011 + 1.6075 w_k1_DIS_s011
                                 + 1.36572 w_k1_FDX_s011
                                 + 1.32039 w_k1_IBM_s011
                                 + 1.59438 w_k1_INTC_s011
                                 + 1.37034 w_k1_KO_s011 + 1.88831 w_k1_MAT_s011
                                 + 1.53722 w_k1_MSFT_s011
                                 + 1.37581 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.847881 w_k1_ADM_s011
                                 + 0.6046 w_k1_AXP_s011
                                 + 0.565405 w_k1_DIS_s011
                                 + 0.737941 w_k1_FDX_s011
                                 + 0.771281 w_k1_IBM_s011
                                 + 0.641557 w_k1_INTC_s011
                                 + 0.713151 w_k1_KO_s011
                                 + 0.666814 w_k1_MAT_s011
                                 + 0.734975 w_k1_MSFT_s011
                                 + 0.731542 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.78818 w_k1_ADM_s011
                                 + 0.885547 w_k1_AXP_s011
                                 + 0.860286 w_k1_DIS_s011
                                 + 0.832898 w_k1_FDX_s011
                                 + 0.963307 w_k1_IBM_s011
                                 + 0.978991 w_k1_INTC_s011
                                 + 1.03563 w_k1_KO_s011 + 1.08052 w_k1_MAT_s011
                                 + 0.876316 w_k1_MSFT_s011
                                 + 1.07058 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.714818 w_k1_ADM_s011
                                 + 0.97492 w_k1_AXP_s011
                                 + 0.95395 w_k1_DIS_s011
                                 + 1.00163 w_k1_FDX_s011
                                 + 0.626942 w_k1_IBM_s011
                                 + 1.19397 w_k1_INTC_s011
                                 + 0.88661 w_k1_KO_s011
                                 + 0.574804 w_k1_MAT_s011
                                 + 1.10551 w_k1_MSFT_s011
                                 + 0.926934 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.17966 w_k1_ADM_s011
                                 + 1.38925 w_k1_AXP_s011
                                 + 1.16283 w_k1_DIS_s011
                                 + 1.07533 w_k1_FDX_s011
                                 + 1.13639 w_k1_IBM_s011
                                 + 1.0715 w_k1_INTC_s011 + 1.1939 w_k1_KO_s011
                                 + 1.38055 w_k1_MAT_s011
                                 + 1.12334 w_k1_MSFT_s011
                                 + 1.17706 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.00478 w_k1_ADM_s011
                                 + 1.21239 w_k1_AXP_s011
                                 + 1.23953 w_k1_DIS_s011
                                 + 1.09908 w_k1_FDX_s011
                                 + 1.50697 w_k1_IBM_s011
                                 + 1.38112 w_k1_INTC_s011
                                 + 1.45654 w_k1_KO_s011 + 1.48842 w_k1_MAT_s011
                                 + 1.30715 w_k1_MSFT_s011
                                 + 1.25181 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.869799 w_k1_ADM_s011
                                 + 0.905697 w_k1_AXP_s011
                                 + 0.927414 w_k1_DIS_s011
                                 + 0.835639 w_k1_FDX_s011
                                 + 1.35931 w_k1_IBM_s011
                                 + 0.916517 w_k1_INTC_s011
                                 + 1.13846 w_k1_KO_s011 + 1.31284 w_k1_MAT_s011
                                 + 1.00357 w_k1_MSFT_s011
                                 + 0.926218 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.813299 w_k1_ADM_s012
                                 + 0.681914 w_k1_AXP_s012
                                 + 0.610569 w_k1_DIS_s012
                                 + 0.650764 w_k1_FDX_s012
                                 + 0.925242 w_k1_IBM_s012
                                 + 1.16652 w_k1_INTC_s012
                                 + 0.849284 w_k1_KO_s012
                                 + 0.539833 w_k1_MAT_s012
                                 + 0.908975 w_k1_MSFT_s012
                                 + 0.827268 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.847922 w_k1_ADM_s012
                                 + 1.04502 w_k1_AXP_s012
                                 + 0.979492 w_k1_DIS_s012
                                 + 1.03899 w_k1_FDX_s012
                                 + 1.08684 w_k1_IBM_s012
                                 + 1.24908 w_k1_INTC_s012
                                 + 1.09011 w_k1_KO_s012 + 1.11175 w_k1_MAT_s012
                                 + 1.07867 w_k1_MSFT_s012
                                 + 1.13727 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.03748 w_k1_ADM_s012
                                 + 1.14612 w_k1_AXP_s012
                                 + 0.861296 w_k1_DIS_s012
                                 + 0.928966 w_k1_FDX_s012
                                 + 1.15129 w_k1_IBM_s012
                                 + 1.00797 w_k1_INTC_s012
                                 + 0.977136 w_k1_KO_s012
                                 + 0.833499 w_k1_MAT_s012
                                 + 0.908225 w_k1_MSFT_s012
                                 + 0.889205 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.06159 w_k1_ADM_s012
                                 + 0.911663 w_k1_AXP_s012
                                 + 0.877309 w_k1_DIS_s012
                                 + 0.960608 w_k1_FDX_s012
                                 + 0.645292 w_k1_IBM_s012
                                 + 0.788354 w_k1_INTC_s012
                                 + 0.78924 w_k1_KO_s012
                                 + 0.652594 w_k1_MAT_s012
                                 + 0.844044 w_k1_MSFT_s012
                                 + 0.971783 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.992983 w_k1_ADM_s012
                                 + 0.951112 w_k1_AXP_s012
                                 + 0.671855 w_k1_DIS_s012
                                 + 0.916231 w_k1_FDX_s012
                                 + 0.955124 w_k1_IBM_s012
                                 + 1.19762 w_k1_INTC_s012
                                 + 1.07167 w_k1_KO_s012
                                 + 0.804229 w_k1_MAT_s012
                                 + 0.854431 w_k1_MSFT_s012
                                 + 0.932368 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.1846 w_k1_ADM_s012
                                 + 1.26908 w_k1_AXP_s012
                                 + 0.786442 w_k1_DIS_s012
                                 + 0.831419 w_k1_FDX_s012
                                 + 1.52144 w_k1_IBM_s012
                                 + 0.997294 w_k1_INTC_s012
                                 + 1.05269 w_k1_KO_s012 + 1.21014 w_k1_MAT_s012
                                 + 0.739442 w_k1_MSFT_s012
                                 + 1.0175 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.05437 w_k1_ADM_s012
                                 + 1.59953 w_k1_AXP_s012
                                 + 1.28805 w_k1_DIS_s012 + 1.2442 w_k1_FDX_s012
                                 + 0.971077 w_k1_IBM_s012
                                 + 1.30375 w_k1_INTC_s012
                                 + 1.17242 w_k1_KO_s012 + 1.31187 w_k1_MAT_s012
                                 + 1.218 w_k1_MSFT_s012 + 1.30641 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.970847 w_k1_ADM_s012
                                 + 1.41267 w_k1_AXP_s012
                                 + 1.45866 w_k1_DIS_s012
                                 + 1.19725 w_k1_FDX_s012
                                 + 1.46985 w_k1_IBM_s012
                                 + 1.07198 w_k1_INTC_s012
                                 + 1.45883 w_k1_KO_s012 + 2.04801 w_k1_MAT_s012
                                 + 1.2617 w_k1_MSFT_s012 + 1.15578 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.33725 w_k1_ADM_s013
                                 + 1.20831 w_k1_AXP_s013
                                 + 1.27029 w_k1_DIS_s013
                                 + 1.26561 w_k1_FDX_s013
                                 + 1.01246 w_k1_IBM_s013
                                 + 0.948589 w_k1_INTC_s013
                                 + 1.10052 w_k1_KO_s013
                                 + 0.849179 w_k1_MAT_s013
                                 + 1.03619 w_k1_MSFT_s013
                                 + 1.10734 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.11734 w_k1_ADM_s013
                                 + 1.09166 w_k1_AXP_s013
                                 + 1.25627 w_k1_DIS_s013
                                 + 0.979925 w_k1_FDX_s013
                                 + 1.53075 w_k1_IBM_s013
                                 + 0.915702 w_k1_INTC_s013
                                 + 1.29706 w_k1_KO_s013 + 1.62997 w_k1_MAT_s013
                                 + 0.871332 w_k1_MSFT_s013
                                 + 1.15157 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.18128 w_k1_ADM_s013
                                 + 0.830573 w_k1_AXP_s013
                                 + 0.819854 w_k1_DIS_s013
                                 + 0.939519 w_k1_FDX_s013
                                 + 0.86726 w_k1_IBM_s013
                                 + 0.805188 w_k1_INTC_s013
                                 + 0.976961 w_k1_KO_s013
                                 + 0.862887 w_k1_MAT_s013
                                 + 0.836398 w_k1_MSFT_s013
                                 + 0.939858 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.01687 w_k1_ADM_s013
                                 + 0.910677 w_k1_AXP_s013
                                 + 1.06256 w_k1_DIS_s013
                                 + 1.09783 w_k1_FDX_s013
                                 + 0.790014 w_k1_IBM_s013
                                 + 0.965314 w_k1_INTC_s013
                                 + 0.955675 w_k1_KO_s013
                                 + 0.630756 w_k1_MAT_s013
                                 + 0.865852 w_k1_MSFT_s013
                                 + 1.02255 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.704972 w_k1_ADM_s013
                                 + 0.680117 w_k1_AXP_s013
                                 + 0.676281 w_k1_DIS_s013
                                 + 0.707145 w_k1_FDX_s013
                                 + 0.775508 w_k1_IBM_s013
                                 + 0.632188 w_k1_INTC_s013
                                 + 0.629975 w_k1_KO_s013
                                 + 0.629895 w_k1_MAT_s013
                                 + 0.622295 w_k1_MSFT_s013
                                 + 0.718524 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.07742 w_k1_ADM_s013
                                 + 1.05594 w_k1_AXP_s013
                                 + 1.28348 w_k1_DIS_s013
                                 + 1.09901 w_k1_FDX_s013 + 1.2271 w_k1_IBM_s013
                                 + 0.902991 w_k1_INTC_s013
                                 + 1.21738 w_k1_KO_s013 + 1.1609 w_k1_MAT_s013
                                 + 0.939313 w_k1_MSFT_s013
                                 + 1.09512 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.848788 w_k1_ADM_s013
                                 + 0.778608 w_k1_AXP_s013
                                 + 0.768255 w_k1_DIS_s013
                                 + 0.896684 w_k1_FDX_s013
                                 + 0.69824 w_k1_IBM_s013
                                 + 0.86544 w_k1_INTC_s013
                                 + 0.809368 w_k1_KO_s013
                                 + 0.617551 w_k1_MAT_s013
                                 + 0.831877 w_k1_MSFT_s013
                                 + 0.903504 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.18339 w_k1_ADM_s013
                                 + 1.06939 w_k1_AXP_s013
                                 + 0.940003 w_k1_DIS_s013
                                 + 1.05755 w_k1_FDX_s013
                                 + 0.629716 w_k1_IBM_s013
                                 + 0.873744 w_k1_INTC_s013
                                 + 0.884757 w_k1_KO_s013
                                 + 0.66276 w_k1_MAT_s013
                                 + 0.804268 w_k1_MSFT_s013
                                 + 0.964825 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.967107 w_k1_ADM_s014
                                 + 0.949304 w_k1_AXP_s014
                                 + 1.00017 w_k1_DIS_s014
                                 + 0.909431 w_k1_FDX_s014
                                 + 1.05457 w_k1_IBM_s014
                                 + 0.78112 w_k1_INTC_s014
                                 + 1.03671 w_k1_KO_s014 + 1.01611 w_k1_MAT_s014
                                 + 0.848258 w_k1_MSFT_s014
                                 + 1.02414 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.05925 w_k1_ADM_s014
                                 + 0.788691 w_k1_AXP_s014
                                 + 0.673287 w_k1_DIS_s014
                                 + 0.804445 w_k1_FDX_s014
                                 + 0.736076 w_k1_IBM_s014
                                 + 0.567645 w_k1_INTC_s014
                                 + 0.750494 w_k1_KO_s014
                                 + 0.83716 w_k1_MAT_s014
                                 + 0.699642 w_k1_MSFT_s014
                                 + 0.994521 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.879835 w_k1_ADM_s014
                                 + 0.944644 w_k1_AXP_s014
                                 + 0.747201 w_k1_DIS_s014
                                 + 0.780249 w_k1_FDX_s014
                                 + 1.32346 w_k1_IBM_s014
                                 + 0.781839 w_k1_INTC_s014
                                 + 1.02831 w_k1_KO_s014
                                 + 0.920221 w_k1_MAT_s014
                                 + 0.720653 w_k1_MSFT_s014
                                 + 0.982292 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.03791 w_k1_ADM_s014
                                 + 1.22915 w_k1_AXP_s014
                                 + 1.26793 w_k1_DIS_s014
                                 + 0.978466 w_k1_FDX_s014
                                 + 1.83187 w_k1_IBM_s014
                                 + 0.796109 w_k1_INTC_s014
                                 + 1.33966 w_k1_KO_s014 + 1.70231 w_k1_MAT_s014
                                 + 1.00937 w_k1_MSFT_s014
                                 + 1.08469 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.44207 w_k1_ADM_s014
                                 + 1.91585 w_k1_AXP_s014
                                 + 2.12373 w_k1_DIS_s014
                                 + 1.69184 w_k1_FDX_s014
                                 + 1.01652 w_k1_IBM_s014
                                 + 1.19691 w_k1_INTC_s014
                                 + 1.27818 w_k1_KO_s014 + 1.05232 w_k1_MAT_s014
                                 + 1.43944 w_k1_MSFT_s014 + 1.249 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.19843 w_k1_ADM_s014
                                 + 1.29557 w_k1_AXP_s014
                                 + 1.31439 w_k1_DIS_s014 + 1.1677 w_k1_FDX_s014
                                 + 1.02344 w_k1_IBM_s014
                                 + 0.910857 w_k1_INTC_s014
                                 + 1.18848 w_k1_KO_s014 + 1.23797 w_k1_MAT_s014
                                 + 1.09436 w_k1_MSFT_s014
                                 + 1.16959 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.00167 w_k1_ADM_s014
                                 + 1.17458 w_k1_AXP_s014
                                 + 1.04232 w_k1_DIS_s014
                                 + 1.03737 w_k1_FDX_s014
                                 + 0.643497 w_k1_IBM_s014
                                 + 0.782623 w_k1_INTC_s014
                                 + 0.700752 w_k1_KO_s014
                                 + 0.728117 w_k1_MAT_s014
                                 + 0.780819 w_k1_MSFT_s014
                                 + 0.887811 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.861086 w_k1_ADM_s014
                                 + 0.608715 w_k1_AXP_s014
                                 + 0.629472 w_k1_DIS_s014
                                 + 0.709267 w_k1_FDX_s014
                                 + 0.818532 w_k1_IBM_s014
                                 + 0.699209 w_k1_INTC_s014
                                 + 0.784933 w_k1_KO_s014 + 0.5387 w_k1_MAT_s014
                                 + 0.75245 w_k1_MSFT_s014
                                 + 0.841663 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.18283 w_k1_ADM_s015
                                 + 1.38825 w_k1_AXP_s015
                                 + 1.29006 w_k1_DIS_s015
                                 + 1.11419 w_k1_FDX_s015
                                 + 1.42407 w_k1_IBM_s015
                                 + 0.923503 w_k1_INTC_s015
                                 + 1.17213 w_k1_KO_s015 + 1.89066 w_k1_MAT_s015
                                 + 1.21236 w_k1_MSFT_s015
                                 + 1.29325 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.2072 w_k1_ADM_s015
                                 + 1.41819 w_k1_AXP_s015
                                 + 1.70755 w_k1_DIS_s015
                                 + 1.49627 w_k1_FDX_s015
                                 + 1.01479 w_k1_IBM_s015
                                 + 0.907178 w_k1_INTC_s015
                                 + 0.989547 w_k1_KO_s015
                                 + 1.24929 w_k1_MAT_s015
                                 + 1.38387 w_k1_MSFT_s015
                                 + 0.979766 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.705771 w_k1_ADM_s015
                                 + 1.05339 w_k1_AXP_s015
                                 + 0.924823 w_k1_DIS_s015
                                 + 0.891504 w_k1_FDX_s015
                                 + 1.37267 w_k1_IBM_s015
                                 + 0.746066 w_k1_INTC_s015
                                 + 0.871743 w_k1_KO_s015
                                 + 1.59113 w_k1_MAT_s015
                                 + 0.954146 w_k1_MSFT_s015
                                 + 0.931522 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.863924 w_k1_ADM_s015
                                 + 0.777401 w_k1_AXP_s015
                                 + 0.659098 w_k1_DIS_s015
                                 + 0.779307 w_k1_FDX_s015
                                 + 0.914652 w_k1_IBM_s015
                                 + 0.695665 w_k1_INTC_s015
                                 + 0.669957 w_k1_KO_s015
                                 + 1.14516 w_k1_MAT_s015
                                 + 0.848936 w_k1_MSFT_s015
                                 + 0.805103 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.01462 w_k1_ADM_s015
                                 + 1.05838 w_k1_AXP_s015
                                 + 1.26344 w_k1_DIS_s015
                                 + 1.15374 w_k1_FDX_s015
                                 + 0.824539 w_k1_IBM_s015
                                 + 0.943302 w_k1_INTC_s015
                                 + 0.862197 w_k1_KO_s015
                                 + 1.04451 w_k1_MAT_s015
                                 + 1.11276 w_k1_MSFT_s015
                                 + 1.03558 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.07639 w_k1_ADM_s015
                                 + 0.824186 w_k1_AXP_s015
                                 + 1.05754 w_k1_DIS_s015 + 1.0613 w_k1_FDX_s015
                                 + 0.586329 w_k1_IBM_s015
                                 + 0.747152 w_k1_INTC_s015
                                 + 0.746395 w_k1_KO_s015
                                 + 0.418218 w_k1_MAT_s015
                                 + 1.0068 w_k1_MSFT_s015
                                 + 0.935505 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.04004 w_k1_ADM_s015
                                 + 0.850153 w_k1_AXP_s015
                                 + 1.0512 w_k1_DIS_s015
                                 + 0.989086 w_k1_FDX_s015
                                 + 1.03431 w_k1_IBM_s015
                                 + 0.875815 w_k1_INTC_s015
                                 + 1.10077 w_k1_KO_s015
                                 + 0.954449 w_k1_MAT_s015
                                 + 1.24432 w_k1_MSFT_s015
                                 + 1.06497 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.07139 w_k1_ADM_s015
                                 + 1.04117 w_k1_AXP_s015
                                 + 1.07352 w_k1_DIS_s015
                                 + 1.00066 w_k1_FDX_s015
                                 + 0.861079 w_k1_IBM_s015
                                 + 0.693141 w_k1_INTC_s015
                                 + 0.785352 w_k1_KO_s015
                                 + 1.15636 w_k1_MAT_s015
                                 + 0.983712 w_k1_MSFT_s015
                                 + 0.972213 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.79756 w_k1_ADM_s016
                                 + 0.737574 w_k1_AXP_s016
                                 + 0.925823 w_k1_DIS_s016
                                 + 0.802824 w_k1_FDX_s016
                                 + 0.871136 w_k1_IBM_s016
                                 + 0.882004 w_k1_INTC_s016
                                 + 0.924816 w_k1_KO_s016
                                 + 0.548297 w_k1_MAT_s016
                                 + 0.90517 w_k1_MSFT_s016
                                 + 0.817844 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.867432 w_k1_ADM_s016
                                 + 1.19345 w_k1_AXP_s016
                                 + 1.20529 w_k1_DIS_s016
                                 + 0.998217 w_k1_FDX_s016
                                 + 1.57862 w_k1_IBM_s016
                                 + 1.09234 w_k1_INTC_s016
                                 + 1.46543 w_k1_KO_s016 + 1.52867 w_k1_MAT_s016
                                 + 1.07989 w_k1_MSFT_s016
                                 + 1.21027 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.02638 w_k1_ADM_s016
                                 + 1.07667 w_k1_AXP_s016
                                 + 1.04979 w_k1_DIS_s016
                                 + 1.02585 w_k1_FDX_s016
                                 + 0.852477 w_k1_IBM_s016
                                 + 1.03848 w_k1_INTC_s016
                                 + 1.04832 w_k1_KO_s016 + 1.0446 w_k1_MAT_s016
                                 + 1.14372 w_k1_MSFT_s016
                                 + 1.00137 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.908143 w_k1_ADM_s016
                                 + 0.881371 w_k1_AXP_s016
                                 + 0.701427 w_k1_DIS_s016
                                 + 0.924305 w_k1_FDX_s016
                                 + 0.65179 w_k1_IBM_s016
                                 + 1.00103 w_k1_INTC_s016
                                 + 0.819715 w_k1_KO_s016
                                 + 0.350789 w_k1_MAT_s016
                                 + 0.884974 w_k1_MSFT_s016
                                 + 0.861667 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.964189 w_k1_ADM_s016
                                 + 1.15883 w_k1_AXP_s016
                                 + 1.06214 w_k1_DIS_s016
                                 + 1.03111 w_k1_FDX_s016
                                 + 1.18753 w_k1_IBM_s016
                                 + 0.919977 w_k1_INTC_s016
                                 + 1.17683 w_k1_KO_s016 + 1.19852 w_k1_MAT_s016
                                 + 0.906897 w_k1_MSFT_s016
                                 + 0.91187 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.600464 w_k1_ADM_s016
                                 + 0.538764 w_k1_AXP_s016
                                 + 0.587262 w_k1_DIS_s016
                                 + 0.644792 w_k1_FDX_s016
                                 + 0.796495 w_k1_IBM_s016
                                 + 0.755643 w_k1_INTC_s016
                                 + 0.753361 w_k1_KO_s016
                                 + 0.589795 w_k1_MAT_s016
                                 + 0.730158 w_k1_MSFT_s016
                                 + 0.804417 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.763982 w_k1_ADM_s016
                                 + 1.53507 w_k1_AXP_s016
                                 + 1.42552 w_k1_DIS_s016
                                 + 1.23331 w_k1_FDX_s016 + 1.1325 w_k1_IBM_s016
                                 + 1.26096 w_k1_INTC_s016
                                 + 1.06228 w_k1_KO_s016 + 1.23445 w_k1_MAT_s016
                                 + 1.21403 w_k1_MSFT_s016
                                 + 1.06709 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.751549 w_k1_ADM_s016
                                 + 0.739447 w_k1_AXP_s016
                                 + 0.703023 w_k1_DIS_s016
                                 + 0.703732 w_k1_FDX_s016
                                 + 1.14418 w_k1_IBM_s016
                                 + 0.745818 w_k1_INTC_s016
                                 + 1.01709 w_k1_KO_s016 + 1.1516 w_k1_MAT_s016
                                 + 0.819596 w_k1_MSFT_s016
                                 + 0.914947 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.14715 w_k1_ADM_s017
                                 + 0.652226 w_k1_AXP_s017
                                 + 0.795445 w_k1_DIS_s017
                                 + 0.822641 w_k1_FDX_s017
                                 + 0.992046 w_k1_IBM_s017
                                 + 0.766172 w_k1_INTC_s017
                                 + 0.916525 w_k1_KO_s017 + 0.9036 w_k1_MAT_s017
                                 + 1.02649 w_k1_MSFT_s017
                                 + 0.996768 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.07772 w_k1_ADM_s017
                                 + 1.28607 w_k1_AXP_s017
                                 + 1.42428 w_k1_DIS_s017
                                 + 1.33291 w_k1_FDX_s017
                                 + 1.42865 w_k1_IBM_s017
                                 + 1.19302 w_k1_INTC_s017
                                 + 1.56405 w_k1_KO_s017 + 1.88075 w_k1_MAT_s017
                                 + 1.3933 w_k1_MSFT_s017 + 1.42397 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.734333 w_k1_ADM_s017
                                 + 0.884871 w_k1_AXP_s017
                                 + 0.698602 w_k1_DIS_s017
                                 + 0.730382 w_k1_FDX_s017
                                 + 1.17273 w_k1_IBM_s017
                                 + 0.899503 w_k1_INTC_s017
                                 + 1.05267 w_k1_KO_s017 + 1.13189 w_k1_MAT_s017
                                 + 0.85026 w_k1_MSFT_s017
                                 + 0.883319 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.854435 w_k1_ADM_s017
                                 + 0.78314 w_k1_AXP_s017
                                 + 1.06801 w_k1_DIS_s017
                                 + 1.00303 w_k1_FDX_s017
                                 + 0.885771 w_k1_IBM_s017
                                 + 1.02532 w_k1_INTC_s017
                                 + 1.09233 w_k1_KO_s017 + 1.23751 w_k1_MAT_s017
                                 + 1.20894 w_k1_MSFT_s017
                                 + 1.00458 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.09624 w_k1_ADM_s017
                                 + 1.09928 w_k1_AXP_s017 + 1.2645 w_k1_DIS_s017
                                 + 1.24078 w_k1_FDX_s017
                                 + 0.750159 w_k1_IBM_s017
                                 + 0.975377 w_k1_INTC_s017
                                 + 0.876394 w_k1_KO_s017
                                 + 1.19617 w_k1_MAT_s017
                                 + 1.16712 w_k1_MSFT_s017
                                 + 1.08859 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.860628 w_k1_ADM_s017
                                 + 0.806662 w_k1_AXP_s017
                                 + 0.666293 w_k1_DIS_s017
                                 + 0.943951 w_k1_FDX_s017
                                 + 0.655451 w_k1_IBM_s017
                                 + 0.821672 w_k1_INTC_s017
                                 + 0.639947 w_k1_KO_s017
                                 + 0.614608 w_k1_MAT_s017
                                 + 0.958377 w_k1_MSFT_s017
                                 + 1.01081 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.08916 w_k1_ADM_s017
                                 + 0.937155 w_k1_AXP_s017
                                 + 1.07393 w_k1_DIS_s017
                                 + 1.01716 w_k1_FDX_s017
                                 + 1.61993 w_k1_IBM_s017
                                 + 0.843762 w_k1_INTC_s017
                                 + 1.23809 w_k1_KO_s017 + 1.60915 w_k1_MAT_s017
                                 + 1.01982 w_k1_MSFT_s017
                                 + 1.14181 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.28887 w_k1_ADM_s017
                                 + 1.11644 w_k1_AXP_s017
                                 + 1.19588 w_k1_DIS_s017
                                 + 1.08975 w_k1_FDX_s017
                                 + 1.02823 w_k1_IBM_s017
                                 + 0.887682 w_k1_INTC_s017
                                 + 1.16665 w_k1_KO_s017 + 1.30691 w_k1_MAT_s017
                                 + 1.21704 w_k1_MSFT_s017
                                 + 1.17401 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.2614 w_k1_ADM_s100
                                 + 1.18619 w_k1_AXP_s100
                                 + 1.21491 w_k1_DIS_s100
                                 + 0.961768 w_k1_FDX_s100
                                 + 1.61839 w_k1_IBM_s100
                                 + 0.951102 w_k1_INTC_s100
                                 + 1.38414 w_k1_KO_s100 + 1.37067 w_k1_MAT_s100
                                 + 0.952768 w_k1_MSFT_s100
                                 + 1.17067 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.860375 w_k1_ADM_s100
                                 + 0.63372 w_k1_AXP_s100
                                 + 0.517998 w_k1_DIS_s100
                                 + 0.793254 w_k1_FDX_s100
                                 + 0.480393 w_k1_IBM_s100
                                 + 0.853371 w_k1_INTC_s100
                                 + 0.680019 w_k1_KO_s100
                                 + 0.290313 w_k1_MAT_s100
                                 + 0.773815 w_k1_MSFT_s100
                                 + 0.763368 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.44373 w_k1_ADM_s100
                                 + 1.60791 w_k1_AXP_s100
                                 + 1.72608 w_k1_DIS_s100 + 1.4222 w_k1_FDX_s100
                                 + 1.34181 w_k1_IBM_s100
                                 + 1.16361 w_k1_INTC_s100
                                 + 1.55567 w_k1_KO_s100 + 1.38444 w_k1_MAT_s100
                                 + 1.27714 w_k1_MSFT_s100 + 1.3444 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.902906 w_k1_ADM_s100
                                 + 0.692382 w_k1_AXP_s100
                                 + 0.848291 w_k1_DIS_s100
                                 + 0.907203 w_k1_FDX_s100
                                 + 0.708347 w_k1_IBM_s100
                                 + 0.84098 w_k1_INTC_s100
                                 + 0.834223 w_k1_KO_s100
                                 + 0.476545 w_k1_MAT_s100
                                 + 0.871312 w_k1_MSFT_s100
                                 + 0.899723 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.02733 w_k1_ADM_s100
                                 + 0.817831 w_k1_AXP_s100
                                 + 0.778439 w_k1_DIS_s100
                                 + 0.772139 w_k1_FDX_s100
                                 + 1.05073 w_k1_IBM_s100
                                 + 0.737294 w_k1_INTC_s100
                                 + 0.996115 w_k1_KO_s100
                                 + 0.711035 w_k1_MAT_s100
                                 + 0.771115 w_k1_MSFT_s100
                                 + 0.85855 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.26721 w_k1_ADM_s100
                                 + 1.00251 w_k1_AXP_s100 + 1.3894 w_k1_DIS_s100
                                 + 1.13193 w_k1_FDX_s100
                                 + 1.08813 w_k1_IBM_s100
                                 + 1.04796 w_k1_INTC_s100
                                 + 1.32978 w_k1_KO_s100 + 1.07729 w_k1_MAT_s100
                                 + 1.201 w_k1_MSFT_s100 + 1.13007 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.14283 w_k1_ADM_s100
                                 + 1.00701 w_k1_AXP_s100
                                 + 1.04503 w_k1_DIS_s100
                                 + 1.14371 w_k1_FDX_s100
                                 + 0.716197 w_k1_IBM_s100
                                 + 1.13547 w_k1_INTC_s100
                                 + 0.909992 w_k1_KO_s100
                                 + 0.480609 w_k1_MAT_s100
                                 + 0.933779 w_k1_MSFT_s100
                                 + 0.992783 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.26042 w_k1_ADM_s100
                                 + 1.29913 w_k1_AXP_s100
                                 + 1.14986 w_k1_DIS_s100 + 1.1066 w_k1_FDX_s100
                                 + 0.954361 w_k1_IBM_s100
                                 + 0.900032 w_k1_INTC_s100
                                 + 0.9748 w_k1_KO_s100 + 1.10726 w_k1_MAT_s100
                                 + 0.94856 w_k1_MSFT_s100
                                 + 0.996703 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.24616 w_k1_ADM_s101
                                 + 1.57663 w_k1_AXP_s101
                                 + 1.71134 w_k1_DIS_s101
                                 + 1.35601 w_k1_FDX_s101
                                 + 1.21166 w_k1_IBM_s101
                                 + 1.37226 w_k1_INTC_s101
                                 + 1.31705 w_k1_KO_s101 + 1.72188 w_k1_MAT_s101
                                 + 1.51841 w_k1_MSFT_s101
                                 + 1.24094 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.92399 w_k1_ADM_s101
                                 + 0.81856 w_k1_AXP_s101
                                 + 1.02418 w_k1_DIS_s101
                                 + 1.10753 w_k1_FDX_s101
                                 + 0.688965 w_k1_IBM_s101
                                 + 1.04687 w_k1_INTC_s101
                                 + 0.914436 w_k1_KO_s101
                                 + 0.746324 w_k1_MAT_s101
                                 + 1.0305 w_k1_MSFT_s101 + 1.07303 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.619862 w_k1_ADM_s101
                                 + 0.583607 w_k1_AXP_s101
                                 + 0.689714 w_k1_DIS_s101
                                 + 0.763798 w_k1_FDX_s101
                                 + 0.684588 w_k1_IBM_s101
                                 + 1.03193 w_k1_INTC_s101
                                 + 0.770911 w_k1_KO_s101
                                 + 0.55418 w_k1_MAT_s101
                                 + 0.861362 w_k1_MSFT_s101
                                 + 0.791666 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.926241 w_k1_ADM_s101
                                 + 0.812467 w_k1_AXP_s101
                                 + 0.940342 w_k1_DIS_s101
                                 + 0.884944 w_k1_FDX_s101
                                 + 1.16927 w_k1_IBM_s101
                                 + 1.03513 w_k1_INTC_s101
                                 + 0.996559 w_k1_KO_s101
                                 + 0.897774 w_k1_MAT_s101
                                 + 0.98151 w_k1_MSFT_s101
                                 + 1.00416 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.851445 w_k1_ADM_s101
                                 + 0.712589 w_k1_AXP_s101
                                 + 0.714045 w_k1_DIS_s101
                                 + 0.809003 w_k1_FDX_s101
                                 + 0.983449 w_k1_IBM_s101
                                 + 1.01913 w_k1_INTC_s101
                                 + 0.854898 w_k1_KO_s101
                                 + 0.904618 w_k1_MAT_s101
                                 + 0.833889 w_k1_MSFT_s101
                                 + 0.946434 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.00667 w_k1_ADM_s101
                                 + 1.23123 w_k1_AXP_s101
                                 + 1.01151 w_k1_DIS_s101
                                 + 0.967578 w_k1_FDX_s101
                                 + 1.33646 w_k1_IBM_s101
                                 + 1.00946 w_k1_INTC_s101
                                 + 1.02045 w_k1_KO_s101 + 1.19461 w_k1_MAT_s101
                                 + 0.912982 w_k1_MSFT_s101
                                 + 1.10482 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.838023 w_k1_ADM_s101
                                 + 1.05121 w_k1_AXP_s101
                                 + 1.21612 w_k1_DIS_s101
                                 + 1.12045 w_k1_FDX_s101
                                 + 0.990041 w_k1_IBM_s101
                                 + 1.18815 w_k1_INTC_s101
                                 + 1.10156 w_k1_KO_s101 + 1.06012 w_k1_MAT_s101
                                 + 1.18351 w_k1_MSFT_s101
                                 + 1.10189 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.35608 w_k1_ADM_s101
                                 + 1.33277 w_k1_AXP_s101
                                 + 0.930679 w_k1_DIS_s101
                                 + 1.08117 w_k1_FDX_s101
                                 + 0.978925 w_k1_IBM_s101
                                 + 1.08745 w_k1_INTC_s101
                                 + 0.919026 w_k1_KO_s101
                                 + 1.16921 w_k1_MAT_s101
                                 + 1.02152 w_k1_MSFT_s101
                                 + 1.09698 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.0077 w_k1_ADM_s102
                                 + 0.870818 w_k1_AXP_s102
                                 + 0.833865 w_k1_DIS_s102
                                 + 1.03567 w_k1_FDX_s102
                                 + 0.585548 w_k1_IBM_s102
                                 + 1.0078 w_k1_INTC_s102
                                 + 0.817477 w_k1_KO_s102
                                 + 0.595085 w_k1_MAT_s102
                                 + 1.13319 w_k1_MSFT_s102
                                 + 0.79914 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.63877 w_k1_ADM_s102
                                 + 1.51745 w_k1_AXP_s102
                                 + 1.46553 w_k1_DIS_s102
                                 + 1.22005 w_k1_FDX_s102
                                 + 1.69351 w_k1_IBM_s102
                                 + 1.30536 w_k1_INTC_s102
                                 + 1.77036 w_k1_KO_s102 + 1.68731 w_k1_MAT_s102
                                 + 1.37477 w_k1_MSFT_s102
                                 + 1.23529 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.54827 w_k1_ADM_s102
                                 + 1.41576 w_k1_AXP_s102
                                 + 1.01106 w_k1_DIS_s102
                                 + 1.27068 w_k1_FDX_s102
                                 + 0.846248 w_k1_IBM_s102
                                 + 0.998341 w_k1_INTC_s102
                                 + 0.983437 w_k1_KO_s102
                                 + 1.26083 w_k1_MAT_s102
                                 + 1.16934 w_k1_MSFT_s102
                                 + 1.13187 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.97061 w_k1_ADM_s102
                                 + 0.937196 w_k1_AXP_s102
                                 + 0.959815 w_k1_DIS_s102
                                 + 0.978258 w_k1_FDX_s102
                                 + 0.989732 w_k1_IBM_s102
                                 + 1.1786 w_k1_INTC_s102 + 1.08429 w_k1_KO_s102
                                 + 1.07113 w_k1_MAT_s102
                                 + 1.04354 w_k1_MSFT_s102
                                 + 0.896143 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.837342 w_k1_ADM_s102
                                 + 0.996215 w_k1_AXP_s102
                                 + 0.857558 w_k1_DIS_s102
                                 + 0.846866 w_k1_FDX_s102
                                 + 1.27359 w_k1_IBM_s102
                                 + 1.06708 w_k1_INTC_s102
                                 + 1.15736 w_k1_KO_s102 + 1.49633 w_k1_MAT_s102
                                 + 1.04754 w_k1_MSFT_s102
                                 + 0.969834 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.936209 w_k1_ADM_s102
                                 + 0.626302 w_k1_AXP_s102
                                 + 0.696274 w_k1_DIS_s102
                                 + 0.758127 w_k1_FDX_s102
                                 + 0.780895 w_k1_IBM_s102
                                 + 0.843688 w_k1_INTC_s102
                                 + 0.893831 w_k1_KO_s102
                                 + 0.708385 w_k1_MAT_s102
                                 + 0.860918 w_k1_MSFT_s102
                                 + 0.792433 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.2653 w_k1_ADM_s102
                                 + 1.04408 w_k1_AXP_s102
                                 + 0.968134 w_k1_DIS_s102
                                 + 1.14393 w_k1_FDX_s102
                                 + 0.85024 w_k1_IBM_s102
                                 + 1.13394 w_k1_INTC_s102
                                 + 1.02585 w_k1_KO_s102
                                 + 0.713283 w_k1_MAT_s102
                                 + 1.09495 w_k1_MSFT_s102
                                 + 0.993643 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.31832 w_k1_ADM_s102
                                 + 1.20842 w_k1_AXP_s102
                                 + 1.49897 w_k1_DIS_s102
                                 + 1.34753 w_k1_FDX_s102
                                 + 1.09365 w_k1_IBM_s102
                                 + 1.3331 w_k1_INTC_s102 + 1.23345 w_k1_KO_s102
                                 + 1.0713 w_k1_MAT_s102
                                 + 1.26873 w_k1_MSFT_s102
                                 + 1.19806 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.939776 w_k1_ADM_s103
                                 + 0.991334 w_k1_AXP_s103
                                 + 0.733154 w_k1_DIS_s103
                                 + 0.85818 w_k1_FDX_s103
                                 + 1.18529 w_k1_IBM_s103
                                 + 0.872779 w_k1_INTC_s103
                                 + 1.02244 w_k1_KO_s103 + 1.38752 w_k1_MAT_s103
                                 + 0.8284 w_k1_MSFT_s103 + 0.98196 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.02647 w_k1_ADM_s103
                                 + 1.06657 w_k1_AXP_s103
                                 + 0.953993 w_k1_DIS_s103
                                 + 1.0678 w_k1_FDX_s103
                                 + 0.825275 w_k1_IBM_s103
                                 + 0.984363 w_k1_INTC_s103
                                 + 0.997928 w_k1_KO_s103
                                 + 1.33113 w_k1_MAT_s103
                                 + 1.11469 w_k1_MSFT_s103
                                 + 0.969023 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.874232 w_k1_ADM_s103
                                 + 0.705989 w_k1_AXP_s103
                                 + 0.637791 w_k1_DIS_s103
                                 + 0.761894 w_k1_FDX_s103
                                 + 0.689293 w_k1_IBM_s103
                                 + 0.608737 w_k1_INTC_s103
                                 + 0.790381 w_k1_KO_s103
                                 + 0.731448 w_k1_MAT_s103
                                 + 0.817906 w_k1_MSFT_s103
                                 + 0.811713 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.855212 w_k1_ADM_s103
                                 + 0.710218 w_k1_AXP_s103
                                 + 0.575813 w_k1_DIS_s103
                                 + 0.668357 w_k1_FDX_s103
                                 + 0.894266 w_k1_IBM_s103
                                 + 0.676741 w_k1_INTC_s103
                                 + 0.87111 w_k1_KO_s103 + 1.16736 w_k1_MAT_s103
                                 + 0.749524 w_k1_MSFT_s103
                                 + 0.839955 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.992894 w_k1_ADM_s103
                                 + 1.02081 w_k1_AXP_s103
                                 + 1.21294 w_k1_DIS_s103
                                 + 1.06418 w_k1_FDX_s103 + 1.1651 w_k1_IBM_s103
                                 + 1.0552 w_k1_INTC_s103 + 1.24531 w_k1_KO_s103
                                 + 1.22502 w_k1_MAT_s103
                                 + 1.15913 w_k1_MSFT_s103 + 1.0538 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.10595 w_k1_ADM_s103
                                 + 1.69147 w_k1_AXP_s103
                                 + 0.907584 w_k1_DIS_s103
                                 + 1.09613 w_k1_FDX_s103
                                 + 1.15809 w_k1_IBM_s103
                                 + 1.04487 w_k1_INTC_s103
                                 + 0.961825 w_k1_KO_s103
                                 + 1.72636 w_k1_MAT_s103
                                 + 1.08106 w_k1_MSFT_s103
                                 + 1.11181 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.968188 w_k1_ADM_s103
                                 + 0.874303 w_k1_AXP_s103
                                 + 0.788269 w_k1_DIS_s103
                                 + 0.921184 w_k1_FDX_s103
                                 + 0.725159 w_k1_IBM_s103
                                 + 0.958561 w_k1_INTC_s103
                                 + 0.916811 w_k1_KO_s103
                                 + 0.827165 w_k1_MAT_s103
                                 + 0.980024 w_k1_MSFT_s103
                                 + 0.935004 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.52887 w_k1_ADM_s103
                                 + 1.38739 w_k1_AXP_s103
                                 + 1.47914 w_k1_DIS_s103
                                 + 1.39506 w_k1_FDX_s103
                                 + 0.901165 w_k1_IBM_s103
                                 + 1.34344 w_k1_INTC_s103
                                 + 1.31492 w_k1_KO_s103 + 1.17367 w_k1_MAT_s103
                                 + 1.45069 w_k1_MSFT_s103 + 1.1249 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.786928 w_k1_ADM_s104
                                 + 1.03965 w_k1_AXP_s104
                                 + 1.26983 w_k1_DIS_s104
                                 + 1.13131 w_k1_FDX_s104
                                 + 0.804603 w_k1_IBM_s104
                                 + 1.24891 w_k1_INTC_s104
                                 + 0.956094 w_k1_KO_s104
                                 + 0.791183 w_k1_MAT_s104
                                 + 1.2225 w_k1_MSFT_s104
                                 + 0.979496 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.735011 w_k1_ADM_s104
                                 + 0.762134 w_k1_AXP_s104
                                 + 0.754487 w_k1_DIS_s104
                                 + 0.792601 w_k1_FDX_s104
                                 + 0.743633 w_k1_IBM_s104
                                 + 0.85518 w_k1_INTC_s104
                                 + 0.730868 w_k1_KO_s104
                                 + 0.702581 w_k1_MAT_s104
                                 + 0.868495 w_k1_MSFT_s104
                                 + 0.8961 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.766735 w_k1_ADM_s104
                                 + 1.10248 w_k1_AXP_s104
                                 + 0.954917 w_k1_DIS_s104
                                 + 0.895899 w_k1_FDX_s104
                                 + 1.35844 w_k1_IBM_s104
                                 + 1.08961 w_k1_INTC_s104
                                 + 1.14937 w_k1_KO_s104 + 0.92629 w_k1_MAT_s104
                                 + 0.9415 w_k1_MSFT_s104 + 1.007 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.18799 w_k1_ADM_s104
                                 + 1.51055 w_k1_AXP_s104
                                 + 1.54287 w_k1_DIS_s104
                                 + 1.25773 w_k1_FDX_s104
                                 + 1.18512 w_k1_IBM_s104
                                 + 1.43501 w_k1_INTC_s104
                                 + 1.35497 w_k1_KO_s104 + 1.38479 w_k1_MAT_s104
                                 + 1.26086 w_k1_MSFT_s104
                                 + 1.17843 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.946499 w_k1_ADM_s104
                                 + 0.876508 w_k1_AXP_s104
                                 + 0.86189 w_k1_DIS_s104
                                 + 0.827546 w_k1_FDX_s104
                                 + 0.934738 w_k1_IBM_s104
                                 + 1.01589 w_k1_INTC_s104
                                 + 0.965354 w_k1_KO_s104
                                 + 0.716561 w_k1_MAT_s104
                                 + 0.791988 w_k1_MSFT_s104
                                 + 0.887508 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.783258 w_k1_ADM_s104
                                 + 0.96069 w_k1_AXP_s104
                                 + 1.26305 w_k1_DIS_s104
                                 + 1.04265 w_k1_FDX_s104
                                 + 1.14826 w_k1_IBM_s104
                                 + 1.16446 w_k1_INTC_s104
                                 + 1.28365 w_k1_KO_s104 + 1.06396 w_k1_MAT_s104
                                 + 1.17394 w_k1_MSFT_s104
                                 + 1.07347 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.901122 w_k1_ADM_s104
                                 + 1.16881 w_k1_AXP_s104
                                 + 0.906647 w_k1_DIS_s104
                                 + 1.06722 w_k1_FDX_s104
                                 + 0.642608 w_k1_IBM_s104
                                 + 1.05244 w_k1_INTC_s104
                                 + 0.811978 w_k1_KO_s104
                                 + 0.560766 w_k1_MAT_s104
                                 + 1.06993 w_k1_MSFT_s104
                                 + 0.927649 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.689693 w_k1_ADM_s104
                                 + 0.748022 w_k1_AXP_s104
                                 + 0.466122 w_k1_DIS_s104
                                 + 0.646157 w_k1_FDX_s104
                                 + 0.775609 w_k1_IBM_s104
                                 + 0.802 w_k1_INTC_s104 + 0.675373 w_k1_KO_s104
                                 + 0.682203 w_k1_MAT_s104
                                 + 0.616737 w_k1_MSFT_s104
                                 + 0.792866 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.925782 w_k1_ADM_s105
                                 + 1.0197 w_k1_AXP_s105 + 1.00787 w_k1_DIS_s105
                                 + 0.952799 w_k1_FDX_s105
                                 + 0.961356 w_k1_IBM_s105
                                 + 1.06557 w_k1_INTC_s105
                                 + 0.955947 w_k1_KO_s105
                                 + 0.845128 w_k1_MAT_s105
                                 + 0.915793 w_k1_MSFT_s105
                                 + 1.02422 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.866599 w_k1_ADM_s105
                                 + 1.08396 w_k1_AXP_s105
                                 + 0.928496 w_k1_DIS_s105
                                 + 0.913258 w_k1_FDX_s105
                                 + 1.29623 w_k1_IBM_s105
                                 + 1.00039 w_k1_INTC_s105
                                 + 0.98327 w_k1_KO_s105 + 1.26004 w_k1_MAT_s105
                                 + 0.967216 w_k1_MSFT_s105
                                 + 1.09387 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.989698 w_k1_ADM_s105
                                 + 1.37467 w_k1_AXP_s105
                                 + 1.17224 w_k1_DIS_s105
                                 + 1.31639 w_k1_FDX_s105
                                 + 0.817231 w_k1_IBM_s105
                                 + 1.26674 w_k1_INTC_s105
                                 + 0.986852 w_k1_KO_s105
                                 + 1.01665 w_k1_MAT_s105
                                 + 1.0633 w_k1_MSFT_s105 + 1.08175 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.735535 w_k1_ADM_s105
                                 + 0.674712 w_k1_AXP_s105
                                 + 0.629941 w_k1_DIS_s105
                                 + 0.783967 w_k1_FDX_s105
                                 + 0.735656 w_k1_IBM_s105
                                 + 0.903268 w_k1_INTC_s105
                                 + 0.749708 w_k1_KO_s105
                                 + 0.633158 w_k1_MAT_s105
                                 + 0.795473 w_k1_MSFT_s105
                                 + 0.850813 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.43751 w_k1_ADM_s105
                                 + 1.49126 w_k1_AXP_s105
                                 + 1.49769 w_k1_DIS_s105
                                 + 1.44856 w_k1_FDX_s105
                                 + 1.18775 w_k1_IBM_s105
                                 + 1.49185 w_k1_INTC_s105
                                 + 1.23972 w_k1_KO_s105
                                 + 0.708495 w_k1_MAT_s105
                                 + 1.22307 w_k1_MSFT_s105
                                 + 1.32392 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.22199 w_k1_ADM_s105
                                 + 1.16715 w_k1_AXP_s105
                                 + 1.47481 w_k1_DIS_s105
                                 + 1.18638 w_k1_FDX_s105
                                 + 1.17867 w_k1_IBM_s105
                                 + 1.02458 w_k1_INTC_s105
                                 + 1.31628 w_k1_KO_s105
                                 + 0.935848 w_k1_MAT_s105
                                 + 1.03894 w_k1_MSFT_s105
                                 + 1.15905 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.07835 w_k1_ADM_s105
                                 + 0.902552 w_k1_AXP_s105
                                 + 0.99752 w_k1_DIS_s105
                                 + 1.08484 w_k1_FDX_s105
                                 + 0.721496 w_k1_IBM_s105
                                 + 1.06421 w_k1_INTC_s105
                                 + 0.892605 w_k1_KO_s105
                                 + 0.511234 w_k1_MAT_s105
                                 + 1.13413 w_k1_MSFT_s105
                                 + 1.02009 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.08923 w_k1_ADM_s105
                                 + 1.5705 w_k1_AXP_s105 + 1.53104 w_k1_DIS_s105
                                 + 1.20676 w_k1_FDX_s105
                                 + 1.58564 w_k1_IBM_s105
                                 + 1.20303 w_k1_INTC_s105
                                 + 1.41148 w_k1_KO_s105 + 1.67519 w_k1_MAT_s105
                                 + 1.15749 w_k1_MSFT_s105
                                 + 1.27001 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.01766 w_k1_ADM_s106
                                 + 0.926508 w_k1_AXP_s106
                                 + 1.18852 w_k1_DIS_s106
                                 + 1.09717 w_k1_FDX_s106
                                 + 0.987676 w_k1_IBM_s106
                                 + 1.36814 w_k1_INTC_s106
                                 + 1.17714 w_k1_KO_s106 + 1.22489 w_k1_MAT_s106
                                 + 1.21315 w_k1_MSFT_s106
                                 + 1.19312 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.894845 w_k1_ADM_s106
                                 + 0.566925 w_k1_AXP_s106
                                 + 0.745867 w_k1_DIS_s106
                                 + 0.79775 w_k1_FDX_s106
                                 + 0.677232 w_k1_IBM_s106
                                 + 0.921042 w_k1_INTC_s106
                                 + 0.816115 w_k1_KO_s106 + 0.5691 w_k1_MAT_s106
                                 + 0.847811 w_k1_MSFT_s106
                                 + 0.835402 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.976168 w_k1_ADM_s106
                                 + 1.28526 w_k1_AXP_s106
                                 + 1.35753 w_k1_DIS_s106
                                 + 1.15726 w_k1_FDX_s106
                                 + 1.66372 w_k1_IBM_s106
                                 + 1.53614 w_k1_INTC_s106 + 1.3921 w_k1_KO_s106
                                 + 1.91467 w_k1_MAT_s106
                                 + 1.32383 w_k1_MSFT_s106
                                 + 1.08061 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.12562 w_k1_ADM_s106
                                 + 1.36151 w_k1_AXP_s106
                                 + 1.78685 w_k1_DIS_s106 + 1.4992 w_k1_FDX_s106
                                 + 0.945147 w_k1_IBM_s106
                                 + 1.70074 w_k1_INTC_s106
                                 + 1.40867 w_k1_KO_s106
                                 + 0.996398 w_k1_MAT_s106
                                 + 1.28018 w_k1_MSFT_s106
                                 + 1.18509 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.780578 w_k1_ADM_s106
                                 + 0.89661 w_k1_AXP_s106
                                 + 0.860268 w_k1_DIS_s106
                                 + 0.808979 w_k1_FDX_s106
                                 + 1.22476 w_k1_IBM_s106
                                 + 1.08724 w_k1_INTC_s106
                                 + 1.03061 w_k1_KO_s106 + 1.54287 w_k1_MAT_s106
                                 + 0.8854 w_k1_MSFT_s106
                                 + 0.966187 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.985864 w_k1_ADM_s106
                                 + 1.01478 w_k1_AXP_s106
                                 + 0.900124 w_k1_DIS_s106
                                 + 1.02727 w_k1_FDX_s106
                                 + 0.71304 w_k1_IBM_s106
                                 + 1.28095 w_k1_INTC_s106
                                 + 0.828957 w_k1_KO_s106
                                 + 0.962929 w_k1_MAT_s106
                                 + 0.93945 w_k1_MSFT_s106
                                 + 0.985272 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.852115 w_k1_ADM_s106
                                 + 0.7028 w_k1_AXP_s106
                                 + 0.572159 w_k1_DIS_s106
                                 + 0.73569 w_k1_FDX_s106
                                 + 0.930072 w_k1_IBM_s106
                                 + 0.872474 w_k1_INTC_s106
                                 + 0.878332 w_k1_KO_s106
                                 + 0.808533 w_k1_MAT_s106
                                 + 0.786161 w_k1_MSFT_s106
                                 + 0.850975 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.640439 w_k1_ADM_s106
                                 + 0.563932 w_k1_AXP_s106
                                 + 0.506225 w_k1_DIS_s106
                                 + 0.630007 w_k1_FDX_s106
                                 + 0.714865 w_k1_IBM_s106
                                 + 0.892059 w_k1_INTC_s106
                                 + 0.741438 w_k1_KO_s106
                                 + 0.75822 w_k1_MAT_s106
                                 + 0.701328 w_k1_MSFT_s106
                                 + 0.824735 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.46812 w_k1_ADM_s107
                                 + 1.15799 w_k1_AXP_s107
                                 + 1.24716 w_k1_DIS_s107
                                 + 1.27251 w_k1_FDX_s107
                                 + 1.30262 w_k1_IBM_s107
                                 + 1.25723 w_k1_INTC_s107
                                 + 1.28439 w_k1_KO_s107 + 1.08355 w_k1_MAT_s107
                                 + 1.22835 w_k1_MSFT_s107
                                 + 1.12302 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.98327 w_k1_ADM_s107
                                 + 0.730188 w_k1_AXP_s107
                                 + 0.665604 w_k1_DIS_s107
                                 + 0.8082 w_k1_FDX_s107
                                 + 0.723573 w_k1_IBM_s107
                                 + 0.706594 w_k1_INTC_s107
                                 + 0.718168 w_k1_KO_s107
                                 + 0.545262 w_k1_MAT_s107
                                 + 0.790212 w_k1_MSFT_s107
                                 + 0.818918 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.03746 w_k1_ADM_s107
                                 + 1.18745 w_k1_AXP_s107
                                 + 1.08474 w_k1_DIS_s107
                                 + 1.06121 w_k1_FDX_s107
                                 + 1.06434 w_k1_IBM_s107
                                 + 1.09615 w_k1_INTC_s107
                                 + 1.10714 w_k1_KO_s107 + 0.98271 w_k1_MAT_s107
                                 + 1.06233 w_k1_MSFT_s107
                                 + 1.01704 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.785798 w_k1_ADM_s107
                                 + 0.977248 w_k1_AXP_s107
                                 + 0.72887 w_k1_DIS_s107
                                 + 0.772536 w_k1_FDX_s107
                                 + 1.24226 w_k1_IBM_s107
                                 + 0.795274 w_k1_INTC_s107
                                 + 0.885522 w_k1_KO_s107
                                 + 1.03014 w_k1_MAT_s107
                                 + 0.674823 w_k1_MSFT_s107
                                 + 0.870633 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.0168 w_k1_ADM_s107
                                 + 0.948281 w_k1_AXP_s107
                                 + 0.933927 w_k1_DIS_s107
                                 + 0.892321 w_k1_FDX_s107
                                 + 0.968858 w_k1_IBM_s107
                                 + 0.860361 w_k1_INTC_s107
                                 + 0.935719 w_k1_KO_s107
                                 + 0.876257 w_k1_MAT_s107
                                 + 0.959805 w_k1_MSFT_s107
                                 + 0.959745 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.13225 w_k1_ADM_s107
                                 + 1.27195 w_k1_AXP_s107
                                 + 1.27395 w_k1_DIS_s107
                                 + 1.08775 w_k1_FDX_s107
                                 + 1.52127 w_k1_IBM_s107
                                 + 1.02032 w_k1_INTC_s107
                                 + 1.36823 w_k1_KO_s107 + 1.57103 w_k1_MAT_s107
                                 + 1.18159 w_k1_MSFT_s107 + 1.1945 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.45385 w_k1_ADM_s107
                                 + 1.38333 w_k1_AXP_s107
                                 + 1.37368 w_k1_DIS_s107
                                 + 1.33849 w_k1_FDX_s107
                                 + 1.00175 w_k1_IBM_s107
                                 + 1.10642 w_k1_INTC_s107
                                 + 1.15245 w_k1_KO_s107 + 1.09704 w_k1_MAT_s107
                                 + 1.23093 w_k1_MSFT_s107
                                 + 1.23517 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.976639 w_k1_ADM_s107
                                 + 1.04572 w_k1_AXP_s107
                                 + 1.04285 w_k1_DIS_s107
                                 + 0.982809 w_k1_FDX_s107
                                 + 1.34787 w_k1_IBM_s107
                                 + 0.947485 w_k1_INTC_s107
                                 + 1.0909 w_k1_KO_s107 + 1.30121 w_k1_MAT_s107
                                 + 0.949799 w_k1_MSFT_s107
                                 + 0.999166 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.31967 w_k1_ADM_s110
                                 + 1.27002 w_k1_AXP_s110 + 1.603 w_k1_DIS_s110
                                 + 1.28914 w_k1_FDX_s110 + 1.6804 w_k1_IBM_s110
                                 + 1.27065 w_k1_INTC_s110
                                 + 1.45295 w_k1_KO_s110 + 1.496 w_k1_MAT_s110
                                 + 1.33419 w_k1_MSFT_s110 + 1.1232 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.876037 w_k1_ADM_s110
                                 + 0.709819 w_k1_AXP_s110
                                 + 0.882607 w_k1_DIS_s110
                                 + 0.791054 w_k1_FDX_s110
                                 + 1.22932 w_k1_IBM_s110
                                 + 0.873161 w_k1_INTC_s110
                                 + 0.833994 w_k1_KO_s110
                                 + 0.939467 w_k1_MAT_s110
                                 + 0.888182 w_k1_MSFT_s110
                                 + 0.874739 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.844899 w_k1_ADM_s110
                                 + 1.17737 w_k1_AXP_s110
                                 + 1.04473 w_k1_DIS_s110
                                 + 1.15664 w_k1_FDX_s110
                                 + 0.99751 w_k1_IBM_s110
                                 + 0.988577 w_k1_INTC_s110
                                 + 0.934082 w_k1_KO_s110
                                 + 1.24489 w_k1_MAT_s110
                                 + 1.21984 w_k1_MSFT_s110
                                 + 1.01102 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.991288 w_k1_ADM_s110
                                 + 1.17499 w_k1_AXP_s110
                                 + 1.44656 w_k1_DIS_s110
                                 + 1.32461 w_k1_FDX_s110
                                 + 1.02601 w_k1_IBM_s110
                                 + 1.20163 w_k1_INTC_s110
                                 + 1.07838 w_k1_KO_s110 + 1.11524 w_k1_MAT_s110
                                 + 1.32832 w_k1_MSFT_s110
                                 + 1.00932 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.96685 w_k1_ADM_s110
                                 + 0.658366 w_k1_AXP_s110
                                 + 1.0603 w_k1_DIS_s110
                                 + 0.909282 w_k1_FDX_s110
                                 + 0.820035 w_k1_IBM_s110
                                 + 0.902938 w_k1_INTC_s110
                                 + 0.821912 w_k1_KO_s110
                                 + 0.72132 w_k1_MAT_s110
                                 + 1.00082 w_k1_MSFT_s110
                                 + 0.841772 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.73592 w_k1_ADM_s110
                                 + 0.630712 w_k1_AXP_s110
                                 + 0.747307 w_k1_DIS_s110
                                 + 0.761175 w_k1_FDX_s110
                                 + 0.779285 w_k1_IBM_s110
                                 + 0.799912 w_k1_INTC_s110
                                 + 0.671084 w_k1_KO_s110
                                 + 0.635798 w_k1_MAT_s110
                                 + 0.929356 w_k1_MSFT_s110
                                 + 0.763519 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.03117 w_k1_ADM_s110
                                 + 1.03691 w_k1_AXP_s110
                                 + 0.830794 w_k1_DIS_s110
                                 + 0.88968 w_k1_FDX_s110
                                 + 1.19209 w_k1_IBM_s110
                                 + 0.961211 w_k1_INTC_s110
                                 + 1.02452 w_k1_KO_s110 + 1.3055 w_k1_MAT_s110
                                 + 0.980618 w_k1_MSFT_s110
                                 + 0.986878 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.2612 w_k1_ADM_s110
                                 + 0.914072 w_k1_AXP_s110
                                 + 1.26652 w_k1_DIS_s110
                                 + 0.974946 w_k1_FDX_s110
                                 + 1.4515 w_k1_IBM_s110
                                 + 0.862514 w_k1_INTC_s110
                                 + 1.30809 w_k1_KO_s110 + 1.68612 w_k1_MAT_s110
                                 + 1.22313 w_k1_MSFT_s110
                                 + 1.14923 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.955929 w_k1_ADM_s111
                                 + 1.12861 w_k1_AXP_s111
                                 + 1.36682 w_k1_DIS_s111
                                 + 1.04894 w_k1_FDX_s111 + 1.538 w_k1_IBM_s111
                                 + 1.45793 w_k1_INTC_s111
                                 + 1.40312 w_k1_KO_s111 + 1.61536 w_k1_MAT_s111
                                 + 1.36055 w_k1_MSFT_s111
                                 + 1.06279 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.813626 w_k1_ADM_s111
                                 + 0.719205 w_k1_AXP_s111
                                 + 0.804076 w_k1_DIS_s111
                                 + 0.871793 w_k1_FDX_s111
                                 + 0.77514 w_k1_IBM_s111
                                 + 0.852103 w_k1_INTC_s111
                                 + 0.849879 w_k1_KO_s111
                                 + 0.759513 w_k1_MAT_s111
                                 + 0.954513 w_k1_MSFT_s111
                                 + 0.754354 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.19945 w_k1_ADM_s111
                                 + 1.19599 w_k1_AXP_s111
                                 + 1.46403 w_k1_DIS_s111
                                 + 1.28858 w_k1_FDX_s111
                                 + 0.985436 w_k1_IBM_s111
                                 + 1.22476 w_k1_INTC_s111
                                 + 1.31224 w_k1_KO_s111 + 1.35881 w_k1_MAT_s111
                                 + 1.47509 w_k1_MSFT_s111
                                 + 1.04098 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.22426 w_k1_ADM_s111
                                 + 1.65216 w_k1_AXP_s111
                                 + 1.49444 w_k1_DIS_s111
                                 + 1.38437 w_k1_FDX_s111
                                 + 1.22851 w_k1_IBM_s111
                                 + 0.947206 w_k1_INTC_s111
                                 + 1.23831 w_k1_KO_s111 + 1.22815 w_k1_MAT_s111
                                 + 1.20838 w_k1_MSFT_s111
                                 + 1.15143 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.936242 w_k1_ADM_s111
                                 + 0.826617 w_k1_AXP_s111
                                 + 0.894295 w_k1_DIS_s111
                                 + 0.955703 w_k1_FDX_s111
                                 + 0.985675 w_k1_IBM_s111
                                 + 0.824877 w_k1_INTC_s111
                                 + 1.06045 w_k1_KO_s111 + 1.29372 w_k1_MAT_s111
                                 + 1.06354 w_k1_MSFT_s111
                                 + 0.999745 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.20623 w_k1_ADM_s111
                                 + 0.886755 w_k1_AXP_s111
                                 + 1.05494 w_k1_DIS_s111
                                 + 1.10025 w_k1_FDX_s111 + 1.0475 w_k1_IBM_s111
                                 + 1.09553 w_k1_INTC_s111 + 1.036 w_k1_KO_s111
                                 + 0.691622 w_k1_MAT_s111
                                 + 1.17456 w_k1_MSFT_s111
                                 + 0.922755 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.843268 w_k1_ADM_s111
                                 + 1.2863 w_k1_AXP_s111
                                 + 0.889503 w_k1_DIS_s111
                                 + 1.10821 w_k1_FDX_s111
                                 + 1.11214 w_k1_IBM_s111
                                 + 0.975082 w_k1_INTC_s111
                                 + 1.01129 w_k1_KO_s111 + 1.27139 w_k1_MAT_s111
                                 + 1.13314 w_k1_MSFT_s111
                                 + 0.892519 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.841839 w_k1_ADM_s111
                                 + 0.958339 w_k1_AXP_s111
                                 + 1.08062 w_k1_DIS_s111
                                 + 1.10274 w_k1_FDX_s111
                                 + 0.71276 w_k1_IBM_s111
                                 + 0.998799 w_k1_INTC_s111
                                 + 0.916301 w_k1_KO_s111
                                 + 0.869588 w_k1_MAT_s111
                                 + 1.2344 w_k1_MSFT_s111
                                 + 0.926815 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.04403 w_k1_ADM_s112
                                 + 1.25633 w_k1_AXP_s112
                                 + 1.37374 w_k1_DIS_s112
                                 + 1.08962 w_k1_FDX_s112
                                 + 1.39948 w_k1_IBM_s112
                                 + 0.936338 w_k1_INTC_s112
                                 + 1.2948 w_k1_KO_s112 + 1.69925 w_k1_MAT_s112
                                 + 1.12754 w_k1_MSFT_s112
                                 + 1.14427 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.765935 w_k1_ADM_s112
                                 + 0.743527 w_k1_AXP_s112
                                 + 0.683916 w_k1_DIS_s112
                                 + 0.78917 w_k1_FDX_s112
                                 + 0.710155 w_k1_IBM_s112
                                 + 0.692818 w_k1_INTC_s112
                                 + 0.804333 w_k1_KO_s112
                                 + 0.904596 w_k1_MAT_s112
                                 + 0.874015 w_k1_MSFT_s112
                                 + 0.900888 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.29352 w_k1_ADM_s112
                                 + 1.44859 w_k1_AXP_s112 + 1.6877 w_k1_DIS_s112
                                 + 1.44281 w_k1_FDX_s112
                                 + 1.35826 w_k1_IBM_s112
                                 + 1.13977 w_k1_INTC_s112 + 1.5492 w_k1_KO_s112
                                 + 1.87962 w_k1_MAT_s112
                                 + 1.35476 w_k1_MSFT_s112
                                 + 1.25705 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.09726 w_k1_ADM_s112
                                 + 0.983584 w_k1_AXP_s112
                                 + 1.03533 w_k1_DIS_s112
                                 + 1.06719 w_k1_FDX_s112
                                 + 0.865695 w_k1_IBM_s112
                                 + 0.873834 w_k1_INTC_s112 + 0.838 w_k1_KO_s112
                                 + 0.968655 w_k1_MAT_s112
                                 + 1.06342 w_k1_MSFT_s112
                                 + 0.970985 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.04883 w_k1_ADM_s112
                                 + 1.06894 w_k1_AXP_s112
                                 + 1.30357 w_k1_DIS_s112
                                 + 1.20173 w_k1_FDX_s112 + 1.0367 w_k1_IBM_s112
                                 + 1.0466 w_k1_INTC_s112 + 1.22476 w_k1_KO_s112
                                 + 1.02666 w_k1_MAT_s112
                                 + 1.24033 w_k1_MSFT_s112
                                 + 1.07253 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.20822 w_k1_ADM_s112
                                 + 1.25029 w_k1_AXP_s112
                                 + 1.06184 w_k1_DIS_s112
                                 + 1.36029 w_k1_FDX_s112
                                 + 0.709769 w_k1_IBM_s112
                                 + 1.00733 w_k1_INTC_s112
                                 + 0.749127 w_k1_KO_s112
                                 + 0.806836 w_k1_MAT_s112
                                 + 1.25133 w_k1_MSFT_s112
                                 + 1.13451 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.794377 w_k1_ADM_s112
                                 + 0.901162 w_k1_AXP_s112
                                 + 0.76584 w_k1_DIS_s112
                                 + 0.911279 w_k1_FDX_s112
                                 + 0.807047 w_k1_IBM_s112
                                 + 0.870179 w_k1_INTC_s112
                                 + 0.883825 w_k1_KO_s112
                                 + 0.839827 w_k1_MAT_s112
                                 + 0.887402 w_k1_MSFT_s112
                                 + 0.933818 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.732722 w_k1_ADM_s112
                                 + 0.966898 w_k1_AXP_s112
                                 + 0.918469 w_k1_DIS_s112
                                 + 0.903229 w_k1_FDX_s112
                                 + 1.27223 w_k1_IBM_s112
                                 + 0.672183 w_k1_INTC_s112
                                 + 0.927663 w_k1_KO_s112
                                 + 1.28805 w_k1_MAT_s112
                                 + 0.850071 w_k1_MSFT_s112
                                 + 0.96585 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.20584 w_k1_ADM_s113
                                 + 1.31176 w_k1_AXP_s113
                                 + 1.14075 w_k1_DIS_s113
                                 + 1.13269 w_k1_FDX_s113
                                 + 1.43061 w_k1_IBM_s113
                                 + 1.29447 w_k1_INTC_s113
                                 + 1.10341 w_k1_KO_s113 + 1.32925 w_k1_MAT_s113
                                 + 1.17001 w_k1_MSFT_s113
                                 + 1.04548 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.00865 w_k1_ADM_s113
                                 + 0.709918 w_k1_AXP_s113
                                 + 0.641427 w_k1_DIS_s113
                                 + 0.759385 w_k1_FDX_s113
                                 + 0.827175 w_k1_IBM_s113
                                 + 0.973339 w_k1_INTC_s113
                                 + 0.811928 w_k1_KO_s113
                                 + 0.531246 w_k1_MAT_s113
                                 + 0.842552 w_k1_MSFT_s113
                                 + 0.804928 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.23302 w_k1_ADM_s113
                                 + 1.29856 w_k1_AXP_s113
                                 + 1.05752 w_k1_DIS_s113
                                 + 1.23522 w_k1_FDX_s113
                                 + 0.945191 w_k1_IBM_s113
                                 + 1.34751 w_k1_INTC_s113
                                 + 0.942189 w_k1_KO_s113
                                 + 0.980762 w_k1_MAT_s113
                                 + 1.11862 w_k1_MSFT_s113
                                 + 1.09761 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.03447 w_k1_ADM_s113
                                 + 0.887651 w_k1_AXP_s113
                                 + 1.00401 w_k1_DIS_s113
                                 + 1.10366 w_k1_FDX_s113
                                 + 0.679282 w_k1_IBM_s113
                                 + 1.28615 w_k1_INTC_s113
                                 + 0.857465 w_k1_KO_s113
                                 + 0.489531 w_k1_MAT_s113
                                 + 1.06975 w_k1_MSFT_s113
                                 + 0.98598 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.33205 w_k1_ADM_s113
                                 + 1.6859 w_k1_AXP_s113 + 1.45425 w_k1_DIS_s113
                                 + 1.28516 w_k1_FDX_s113
                                 + 1.71354 w_k1_IBM_s113
                                 + 1.59514 w_k1_INTC_s113
                                 + 1.43092 w_k1_KO_s113 + 1.35314 w_k1_MAT_s113
                                 + 1.13438 w_k1_MSFT_s113
                                 + 1.24415 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.07832 w_k1_ADM_s113
                                 + 0.925388 w_k1_AXP_s113
                                 + 1.47191 w_k1_DIS_s113
                                 + 1.22896 w_k1_FDX_s113
                                 + 1.16955 w_k1_IBM_s113
                                 + 1.51027 w_k1_INTC_s113
                                 + 1.33979 w_k1_KO_s113
                                 + 0.659378 w_k1_MAT_s113
                                 + 1.19851 w_k1_MSFT_s113
                                 + 1.13548 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.14346 w_k1_ADM_s113
                                 + 0.933511 w_k1_AXP_s113
                                 + 0.855934 w_k1_DIS_s113
                                 + 0.97489 w_k1_FDX_s113
                                 + 0.886538 w_k1_IBM_s113
                                 + 1.12505 w_k1_INTC_s113
                                 + 0.946393 w_k1_KO_s113
                                 + 0.605659 w_k1_MAT_s113
                                 + 0.985265 w_k1_MSFT_s113
                                 + 0.984386 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.974225 w_k1_ADM_s113
                                 + 0.894994 w_k1_AXP_s113
                                 + 0.93355 w_k1_DIS_s113
                                 + 0.829609 w_k1_FDX_s113
                                 + 1.40079 w_k1_IBM_s113
                                 + 1.07972 w_k1_INTC_s113 + 1.1216 w_k1_KO_s113
                                 + 0.930527 w_k1_MAT_s113
                                 + 0.890545 w_k1_MSFT_s113
                                 + 1.00053 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.04398 w_k1_ADM_s114
                                 + 0.835801 w_k1_AXP_s114
                                 + 0.667226 w_k1_DIS_s114
                                 + 0.861226 w_k1_FDX_s114
                                 + 0.781197 w_k1_IBM_s114
                                 + 0.718795 w_k1_INTC_s114
                                 + 0.683131 w_k1_KO_s114
                                 + 0.68997 w_k1_MAT_s114
                                 + 0.843347 w_k1_MSFT_s114
                                 + 0.862755 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.05681 w_k1_ADM_s114
                                 + 1.20334 w_k1_AXP_s114
                                 + 1.26266 w_k1_DIS_s114
                                 + 1.30105 w_k1_FDX_s114
                                 + 1.28449 w_k1_IBM_s114
                                 + 1.24459 w_k1_INTC_s114
                                 + 1.21065 w_k1_KO_s114 + 1.10428 w_k1_MAT_s114
                                 + 1.35781 w_k1_MSFT_s114
                                 + 1.15745 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.915595 w_k1_ADM_s114
                                 + 1.18164 w_k1_AXP_s114
                                 + 0.998873 w_k1_DIS_s114
                                 + 1.06558 w_k1_FDX_s114 + 1.1999 w_k1_IBM_s114
                                 + 0.904955 w_k1_INTC_s114
                                 + 0.990305 w_k1_KO_s114
                                 + 1.47913 w_k1_MAT_s114
                                 + 1.03685 w_k1_MSFT_s114
                                 + 1.03204 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.92017 w_k1_ADM_s114
                                 + 1.02237 w_k1_AXP_s114
                                 + 1.09526 w_k1_DIS_s114
                                 + 1.10349 w_k1_FDX_s114
                                 + 0.914852 w_k1_IBM_s114
                                 + 1.02538 w_k1_INTC_s114
                                 + 0.871902 w_k1_KO_s114
                                 + 1.19156 w_k1_MAT_s114
                                 + 1.0436 w_k1_MSFT_s114 + 1.09431 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.04396 w_k1_ADM_s114
                                 + 1.09373 w_k1_AXP_s114
                                 + 1.08379 w_k1_DIS_s114 + 1.026 w_k1_FDX_s114
                                 + 1.92797 w_k1_IBM_s114
                                 + 1.00806 w_k1_INTC_s114
                                 + 1.40773 w_k1_KO_s114 + 2.16045 w_k1_MAT_s114
                                 + 1.10714 w_k1_MSFT_s114
                                 + 1.16201 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.848995 w_k1_ADM_s114
                                 + 1.02248 w_k1_AXP_s114
                                 + 0.728744 w_k1_DIS_s114
                                 + 0.842009 w_k1_FDX_s114
                                 + 1.2971 w_k1_IBM_s114
                                 + 0.947794 w_k1_INTC_s114
                                 + 0.875862 w_k1_KO_s114
                                 + 1.51454 w_k1_MAT_s114
                                 + 0.956654 w_k1_MSFT_s114
                                 + 0.945015 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.45809 w_k1_ADM_s114
                                 + 1.74889 w_k1_AXP_s114
                                 + 1.81193 w_k1_DIS_s114 + 1.5174 w_k1_FDX_s114
                                 + 1.58952 w_k1_IBM_s114
                                 + 1.71584 w_k1_INTC_s114
                                 + 1.70057 w_k1_KO_s114 + 1.56216 w_k1_MAT_s114
                                 + 1.69832 w_k1_MSFT_s114
                                 + 1.31587 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.07281 w_k1_ADM_s114
                                 + 0.911472 w_k1_AXP_s114
                                 + 0.935119 w_k1_DIS_s114
                                 + 1.13953 w_k1_FDX_s114
                                 + 0.672016 w_k1_IBM_s114
                                 + 1.09441 w_k1_INTC_s114
                                 + 0.777343 w_k1_KO_s114
                                 + 0.612243 w_k1_MAT_s114
                                 + 1.19339 w_k1_MSFT_s114
                                 + 0.908176 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.768808 w_k1_ADM_s115
                                 + 0.841715 w_k1_AXP_s115
                                 + 0.900217 w_k1_DIS_s115
                                 + 0.968932 w_k1_FDX_s115 + 0.873 w_k1_IBM_s115
                                 + 0.93089 w_k1_INTC_s115
                                 + 0.914602 w_k1_KO_s115
                                 + 1.00043 w_k1_MAT_s115
                                 + 0.931376 w_k1_MSFT_s115
                                 + 1.05277 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.987801 w_k1_ADM_s115
                                 + 1.01897 w_k1_AXP_s115
                                 + 1.13918 w_k1_DIS_s115
                                 + 1.07375 w_k1_FDX_s115
                                 + 1.10813 w_k1_IBM_s115
                                 + 1.17841 w_k1_INTC_s115
                                 + 1.33152 w_k1_KO_s115 + 1.07173 w_k1_MAT_s115
                                 + 1.01773 w_k1_MSFT_s115
                                 + 1.07283 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.904409 w_k1_ADM_s115
                                 + 0.770703 w_k1_AXP_s115
                                 + 0.861181 w_k1_DIS_s115
                                 + 0.869302 w_k1_FDX_s115
                                 + 1.30152 w_k1_IBM_s115
                                 + 0.864125 w_k1_INTC_s115
                                 + 1.16698 w_k1_KO_s115 + 1.65068 w_k1_MAT_s115
                                 + 0.85591 w_k1_MSFT_s115
                                 + 1.05027 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.92271 w_k1_ADM_s115
                                 + 0.912727 w_k1_AXP_s115
                                 + 0.703838 w_k1_DIS_s115
                                 + 0.827803 w_k1_FDX_s115
                                 + 0.923535 w_k1_IBM_s115
                                 + 0.987841 w_k1_INTC_s115
                                 + 0.86256 w_k1_KO_s115 + 1.18972 w_k1_MAT_s115
                                 + 0.918156 w_k1_MSFT_s115
                                 + 0.935429 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.1232 w_k1_ADM_s115
                                 + 0.97426 w_k1_AXP_s115 + 1.3289 w_k1_DIS_s115
                                 + 1.24591 w_k1_FDX_s115
                                 + 0.760382 w_k1_IBM_s115
                                 + 0.917055 w_k1_INTC_s115
                                 + 1.07592 w_k1_KO_s115
                                 + 0.991574 w_k1_MAT_s115
                                 + 1.01377 w_k1_MSFT_s115
                                 + 1.03887 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.991234 w_k1_ADM_s115
                                 + 1.29658 w_k1_AXP_s115
                                 + 1.06961 w_k1_DIS_s115
                                 + 1.31182 w_k1_FDX_s115
                                 + 0.89769 w_k1_IBM_s115
                                 + 1.1707 w_k1_INTC_s115 + 1.03894 w_k1_KO_s115
                                 + 1.44808 w_k1_MAT_s115
                                 + 0.972876 w_k1_MSFT_s115
                                 + 1.13889 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.706951 w_k1_ADM_s115
                                 + 0.643365 w_k1_AXP_s115
                                 + 0.576699 w_k1_DIS_s115
                                 + 0.728325 w_k1_FDX_s115
                                 + 0.720203 w_k1_IBM_s115
                                 + 1.07382 w_k1_INTC_s115
                                 + 0.72651 w_k1_KO_s115
                                 + 0.591078 w_k1_MAT_s115
                                 + 0.740967 w_k1_MSFT_s115
                                 + 0.862902 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.23661 w_k1_ADM_s115
                                 + 1.10864 w_k1_AXP_s115
                                 + 1.35159 w_k1_DIS_s115
                                 + 1.26657 w_k1_FDX_s115
                                 + 1.50586 w_k1_IBM_s115
                                 + 1.15136 w_k1_INTC_s115
                                 + 1.49166 w_k1_KO_s115 + 1.77849 w_k1_MAT_s115
                                 + 1.08425 w_k1_MSFT_s115
                                 + 1.27814 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.981799 w_k1_ADM_s116
                                 + 1.20523 w_k1_AXP_s116
                                 + 1.25427 w_k1_DIS_s116 + 1.0006 w_k1_FDX_s116
                                 + 1.63529 w_k1_IBM_s116
                                 + 1.26282 w_k1_INTC_s116
                                 + 1.21558 w_k1_KO_s116 + 1.60249 w_k1_MAT_s116
                                 + 0.978836 w_k1_MSFT_s116
                                 + 1.08938 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.901733 w_k1_ADM_s116
                                 + 0.780676 w_k1_AXP_s116
                                 + 0.76685 w_k1_DIS_s116
                                 + 0.907251 w_k1_FDX_s116
                                 + 0.743318 w_k1_IBM_s116
                                 + 0.954798 w_k1_INTC_s116
                                 + 0.909685 w_k1_KO_s116
                                 + 0.659136 w_k1_MAT_s116
                                 + 0.787543 w_k1_MSFT_s116
                                 + 0.946381 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.0327 w_k1_ADM_s116
                                 + 0.947103 w_k1_AXP_s116
                                 + 1.04275 w_k1_DIS_s116
                                 + 1.13821 w_k1_FDX_s116
                                 + 0.743689 w_k1_IBM_s116
                                 + 1.13629 w_k1_INTC_s116
                                 + 0.954173 w_k1_KO_s116
                                 + 0.628646 w_k1_MAT_s116
                                 + 0.967574 w_k1_MSFT_s116
                                 + 1.06701 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.14716 w_k1_ADM_s116
                                 + 1.0712 w_k1_AXP_s116 + 1.40048 w_k1_DIS_s116
                                 + 1.17107 w_k1_FDX_s116 + 1.1283 w_k1_IBM_s116
                                 + 1.22326 w_k1_INTC_s116
                                 + 1.14814 w_k1_KO_s116
                                 + 0.820657 w_k1_MAT_s116
                                 + 1.1424 w_k1_MSFT_s116 + 1.20148 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.00246 w_k1_ADM_s116
                                 + 0.889573 w_k1_AXP_s116
                                 + 1.02469 w_k1_DIS_s116
                                 + 0.821565 w_k1_FDX_s116
                                 + 1.29298 w_k1_IBM_s116
                                 + 1.09771 w_k1_INTC_s116
                                 + 1.01453 w_k1_KO_s116 + 1.03741 w_k1_MAT_s116
                                 + 0.820695 w_k1_MSFT_s116
                                 + 0.983626 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.24545 w_k1_ADM_s116
                                 + 1.4765 w_k1_AXP_s116 + 1.73671 w_k1_DIS_s116
                                 + 1.3318 w_k1_FDX_s116 + 1.16058 w_k1_IBM_s116
                                 + 1.60364 w_k1_INTC_s116
                                 + 1.28237 w_k1_KO_s116 + 1.50274 w_k1_MAT_s116
                                 + 1.29364 w_k1_MSFT_s116
                                 + 1.24682 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.932034 w_k1_ADM_s116
                                 + 1.1413 w_k1_AXP_s116 + 1.01801 w_k1_DIS_s116
                                 + 0.994879 w_k1_FDX_s116
                                 + 1.07912 w_k1_IBM_s116
                                 + 1.13331 w_k1_INTC_s116
                                 + 1.02398 w_k1_KO_s116 + 1.26474 w_k1_MAT_s116
                                 + 0.978432 w_k1_MSFT_s116
                                 + 1.05282 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.686983 w_k1_ADM_s116
                                 + 0.641274 w_k1_AXP_s116
                                 + 0.683142 w_k1_DIS_s116
                                 + 0.709128 w_k1_FDX_s116
                                 + 0.994154 w_k1_IBM_s116
                                 + 0.723218 w_k1_INTC_s116
                                 + 0.763159 w_k1_KO_s116
                                 + 0.836983 w_k1_MAT_s116
                                 + 0.666046 w_k1_MSFT_s116
                                 + 0.823292 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.52455 w_k1_ADM_s117
                                 + 1.48734 w_k1_AXP_s117
                                 + 1.57628 w_k1_DIS_s117
                                 + 1.50758 w_k1_FDX_s117
                                 + 0.868882 w_k1_IBM_s117
                                 + 1.21905 w_k1_INTC_s117
                                 + 1.14132 w_k1_KO_s117
                                 + 0.725908 w_k1_MAT_s117
                                 + 1.35392 w_k1_MSFT_s117
                                 + 1.18789 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.774566 w_k1_ADM_s117
                                 + 0.564757 w_k1_AXP_s117
                                 + 0.695717 w_k1_DIS_s117
                                 + 0.713988 w_k1_FDX_s117
                                 + 0.925869 w_k1_IBM_s117
                                 + 0.676908 w_k1_INTC_s117
                                 + 0.921775 w_k1_KO_s117
                                 + 0.588382 w_k1_MAT_s117
                                 + 0.834132 w_k1_MSFT_s117
                                 + 0.835684 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.08676 w_k1_ADM_s117
                                 + 1.04561 w_k1_AXP_s117
                                 + 1.16281 w_k1_DIS_s117
                                 + 1.16422 w_k1_FDX_s117
                                 + 1.07896 w_k1_IBM_s117
                                 + 1.02381 w_k1_INTC_s117
                                 + 1.11203 w_k1_KO_s117 + 1.06038 w_k1_MAT_s117
                                 + 1.09077 w_k1_MSFT_s117 + 1.0905 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.28645 w_k1_ADM_s117
                                 + 1.42757 w_k1_AXP_s117
                                 + 1.32515 w_k1_DIS_s117 + 1.1592 w_k1_FDX_s117
                                 + 1.3172 w_k1_IBM_s117
                                 + 0.927067 w_k1_INTC_s117
                                 + 1.13092 w_k1_KO_s117 + 1.43024 w_k1_MAT_s117
                                 + 1.08524 w_k1_MSFT_s117
                                 + 1.17103 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.880996 w_k1_ADM_s117
                                 + 0.589607 w_k1_AXP_s117
                                 + 0.678785 w_k1_DIS_s117
                                 + 0.754382 w_k1_FDX_s117
                                 + 0.563087 w_k1_IBM_s117
                                 + 0.732847 w_k1_INTC_s117
                                 + 0.710895 w_k1_KO_s117
                                 + 0.396679 w_k1_MAT_s117
                                 + 0.864973 w_k1_MSFT_s117
                                 + 0.765631 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.18307 w_k1_ADM_s117
                                 + 1.38113 w_k1_AXP_s117
                                 + 1.30006 w_k1_DIS_s117
                                 + 0.997531 w_k1_FDX_s117
                                 + 1.79628 w_k1_IBM_s117
                                 + 0.913973 w_k1_INTC_s117
                                 + 1.35754 w_k1_KO_s117 + 1.4637 w_k1_MAT_s117
                                 + 1.04866 w_k1_MSFT_s117
                                 + 1.11532 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.951158 w_k1_ADM_s117
                                 + 0.849976 w_k1_AXP_s117
                                 + 0.960209 w_k1_DIS_s117
                                 + 0.965736 w_k1_FDX_s117
                                 + 0.831526 w_k1_IBM_s117
                                 + 0.785759 w_k1_INTC_s117
                                 + 0.966299 w_k1_KO_s117
                                 + 0.721599 w_k1_MAT_s117
                                 + 1.0431 w_k1_MSFT_s117
                                 + 0.974689 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.981864 w_k1_ADM_s117
                                 + 0.962512 w_k1_AXP_s117
                                 + 0.855507 w_k1_DIS_s117
                                 + 0.864122 w_k1_FDX_s117
                                 + 1.26398 w_k1_IBM_s117
                                 + 0.869537 w_k1_INTC_s117
                                 + 1.12662 w_k1_KO_s117
                                 + 0.858422 w_k1_MAT_s117
                                 + 0.807111 w_k1_MSFT_s117
                                 + 0.876256 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0323129941630802 w_k1_ADM_s000
                                 + 0.0740430129350845 w_k1_ADM_s001
                                 + 0.0197327603992117 w_k1_ADM_s002
                                 + 0.0430055856322438 w_k1_ADM_s003
                                 + 0.0813625792416553 w_k1_ADM_s004
                                 + 0.0469170962014921 w_k1_ADM_s005
                                 + 0.0362921064435675 w_k1_ADM_s006
                                 + 0.0664195450435286 w_k1_ADM_s007
                                 + 0.0044370298043624 w_k1_ADM_s010
                                 + 0.0113772868464493 w_k1_ADM_s011
                                 + 0.00619026294995647 w_k1_ADM_s012
                                 + 0.0147800367596884 w_k1_ADM_s013
                                 + 0.0136164303143426 w_k1_ADM_s014
                                 + 0.00809956211361774 w_k1_ADM_s015
                                 + 0.00765762624397575 w_k1_ADM_s016
                                 + 0.0125160410674356 w_k1_ADM_s017
                                 + 0.0360175271467703 w_k1_ADM_s100
                                 + 0.0344433514889488 w_k1_ADM_s101
                                 + 0.018105945791102 w_k1_ADM_s102
                                 + 0.044429034206686 w_k1_ADM_s103
                                 + 0.0458270651070114 w_k1_ADM_s104
                                 + 0.0233965330512357 w_k1_ADM_s105
                                 + 0.0546264708163978 w_k1_ADM_s106
                                 + 0.0436230432089004 w_k1_ADM_s107
                                 + 0.0159865993900107 w_k1_ADM_s110
                                 + 0.0160707283276333 w_k1_ADM_s111
                                 + 0.0274556301980693 w_k1_ADM_s112
                                 + 0.0326022245384342 w_k1_ADM_s113
                                 + 0.0345658647757763 w_k1_ADM_s114
                                 + 0.0188553277385825 w_k1_ADM_s115
                                 + 0.0648793053019742 w_k1_ADM_s116
                                 + 0.0178568236200013 w_k1_ADM_s117
                                 + 0.03770113077545 w_k1_AXP_s000
                                 + 0.0804137440844465 w_k1_AXP_s001
                                 + 0.0196842371999319 w_k1_AXP_s002
                                 + 0.0401509280038977 w_k1_AXP_s003
                                 + 0.0817762689742129 w_k1_AXP_s004
                                 + 0.0482184893699216 w_k1_AXP_s005
                                 + 0.0369876290310944 w_k1_AXP_s006
                                 + 0.0688640328545906 w_k1_AXP_s007
                                 + 0.00546061379712028 w_k1_AXP_s010
                                 + 0.0132701970910704 w_k1_AXP_s011
                                 + 0.00669317177746051 w_k1_AXP_s012
                                 + 0.0134409363064536 w_k1_AXP_s013
                                 + 0.0143002258673659 w_k1_AXP_s014
                                 + 0.00817170128582644 w_k1_AXP_s015
                                 + 0.00961981891303282 w_k1_AXP_s016
                                 + 0.0114227656817304 w_k1_AXP_s017
                                 + 0.0321001201371528 w_k1_AXP_s100
                                 + 0.0355357526404968 w_k1_AXP_s101
                                 + 0.0167267019412258 w_k1_AXP_s102
                                 + 0.0451432168221359 w_k1_AXP_s103
                                 + 0.0542142152565186 w_k1_AXP_s104
                                 + 0.0253596426362385 w_k1_AXP_s105
                                 + 0.0550950797725389 w_k1_AXP_s106
                                 + 0.0418779342933978 w_k1_AXP_s107
                                 + 0.0148250721941683 w_k1_AXP_s110
                                 + 0.0164814065859329 w_k1_AXP_s111
                                 + 0.0294178338071552 w_k1_AXP_s112
                                 + 0.0291784647085856 w_k1_AXP_s113
                                 + 0.0353435516346135 w_k1_AXP_s114
                                 + 0.0186463080482101 w_k1_AXP_s115
                                 + 0.0661493554992102 w_k1_AXP_s116
                                 + 0.0173446137038261 w_k1_AXP_s117
                                 + 0.0339890738010239 w_k1_DIS_s000
                                 + 0.0766446254441954 w_k1_DIS_s001
                                 + 0.0178749016575103 w_k1_DIS_s002
                                 + 0.0409537292174366 w_k1_DIS_s003
                                 + 0.0829011792880238 w_k1_DIS_s004
                                 + 0.0468527034477143 w_k1_DIS_s005
                                 + 0.0400223588687516 w_k1_DIS_s006
                                 + 0.0641848526819669 w_k1_DIS_s007
                                 + 0.00538434936068242 w_k1_DIS_s010
                                 + 0.0124729968084831 w_k1_DIS_s011
                                 + 0.00575222125217394 w_k1_DIS_s012
                                 + 0.014658384553452 w_k1_DIS_s013
                                 + 0.0140300983198812 w_k1_DIS_s014
                                 + 0.00850302995258166 w_k1_DIS_s015
                                 + 0.00926836757885055 w_k1_DIS_s016
                                 + 0.0127409207610636 w_k1_DIS_s017
                                 + 0.0343982003198021 w_k1_DIS_s100
                                 + 0.0375229990604788 w_k1_DIS_s101
                                 + 0.0165332221067097 w_k1_DIS_s102
                                 + 0.0415814068257786 w_k1_DIS_s103
                                 + 0.0546068535351361 w_k1_DIS_s104
                                 + 0.0249114052026501 w_k1_DIS_s105
                                 + 0.06008569595278 w_k1_DIS_s106
                                 + 0.0400365127152071 w_k1_DIS_s107
                                 + 0.0177777251509509 w_k1_DIS_s110
                                 + 0.0180436609533122 w_k1_DIS_s111
                                 + 0.0294257312131659 w_k1_DIS_s112
                                 + 0.029511857298958 w_k1_DIS_s113
                                 + 0.0336517946958658 w_k1_DIS_s114
                                 + 0.0199871613608033 w_k1_DIS_s115
                                 + 0.0704667603103677 w_k1_DIS_s116
                                 + 0.0178090657219871 w_k1_DIS_s117
                                 + 0.0337148104083376 w_k1_FDX_s000
                                 + 0.0770405900738168 w_k1_FDX_s001
                                 + 0.0186183239450227 w_k1_FDX_s002
                                 + 0.0409722590436487 w_k1_FDX_s003
                                 + 0.0774500739253894 w_k1_FDX_s004
                                 + 0.0477674031585756 w_k1_FDX_s005
                                 + 0.0391297823487561 w_k1_FDX_s006
                                 + 0.0654619948962879 w_k1_FDX_s007
                                 + 0.00493755512164702 w_k1_FDX_s010
                                 + 0.0121735192732914 w_k1_FDX_s011
                                 + 0.00605474513169613 w_k1_FDX_s012
                                 + 0.0142386824904667 w_k1_FDX_s013
                                 + 0.0130339739137638 w_k1_FDX_s014
                                 + 0.00823356739173467 w_k1_FDX_s015
                                 + 0.00874837664719877 w_k1_FDX_s016
                                 + 0.0127744146897322 w_k1_FDX_s017
                                 + 0.0325123162317111 w_k1_FDX_s100
                                 + 0.036710215570946 w_k1_FDX_s101
                                 + 0.0168669719293607 w_k1_FDX_s102
                                 + 0.042832743835738 w_k1_FDX_s103
                                 + 0.0498237814014576 w_k1_FDX_s104
                                 + 0.0246218231417984 w_k1_FDX_s105
                                 + 0.0582193529232382 w_k1_FDX_s106
                                 + 0.0401523434386624 w_k1_FDX_s107
                                 + 0.0162688815517389 w_k1_FDX_s110
                                 + 0.0175946116517367 w_k1_FDX_s111
                                 + 0.0300606365177577 w_k1_FDX_s112
                                 + 0.0299887542671956 w_k1_FDX_s113
                                 + 0.0360491057184035 w_k1_FDX_s114
                                 + 0.0205583528938948 w_k1_FDX_s115
                                 + 0.0666076868629832 w_k1_FDX_s116
                                 + 0.0170899039443429 w_k1_FDX_s117
                                 + 0.032270477117115 w_k1_IBM_s000
                                 + 0.0789736070082112 w_k1_IBM_s001
                                 + 0.0183783762381157 w_k1_IBM_s002
                                 + 0.0421284357148986 w_k1_IBM_s003
                                 + 0.081488475087882 w_k1_IBM_s004
                                 + 0.0466462019432678 w_k1_IBM_s005
                                 + 0.0384837273002694 w_k1_IBM_s006
                                 + 0.0649446556503692 w_k1_IBM_s007
                                 + 0.00496777217621048 w_k1_IBM_s010
                                 + 0.0128907053725216 w_k1_IBM_s011
                                 + 0.00621069091806541 w_k1_IBM_s012
                                 + 0.0133998411957445 w_k1_IBM_s013
                                 + 0.0130914338377195 w_k1_IBM_s014
                                 + 0.0082163155130823 w_k1_IBM_s015
                                 + 0.00947250624370526 w_k1_IBM_s016
                                 + 0.0124464198518605 w_k1_IBM_s017
                                 + 0.03152573793589 w_k1_IBM_s100
                                 + 0.0354651150113139 w_k1_IBM_s101
                                 + 0.016873691499235 w_k1_IBM_s102
                                 + 0.042425688922088 w_k1_IBM_s103
                                 + 0.0506942476744646 w_k1_IBM_s104
                                 + 0.0243007916361329 w_k1_IBM_s105
                                 + 0.059617144382727 w_k1_IBM_s106
                                 + 0.0423888813353085 w_k1_IBM_s107
                                 + 0.0179065872190206 w_k1_IBM_s110
                                 + 0.0169931252079881 w_k1_IBM_s111
                                 + 0.0280646399851443 w_k1_IBM_s112
                                 + 0.0312434152140296 w_k1_IBM_s113
                                 + 0.0351216656385057 w_k1_IBM_s114
                                 + 0.0211268291885403 w_k1_IBM_s115
                                 + 0.0673054894383349 w_k1_IBM_s116
                                 + 0.0182484454221468 w_k1_IBM_s117
                                 + 0.0331227399043051 w_k1_INTC_s000
                                 + 0.0680905818819305 w_k1_INTC_s001
                                 + 0.020066022942889 w_k1_INTC_s002
                                 + 0.0440828584320362 w_k1_INTC_s003
                                 + 0.0693185621197939 w_k1_INTC_s004
                                 + 0.0545886143720695 w_k1_INTC_s005
                                 + 0.0354668142469251 w_k1_INTC_s006
                                 + 0.064043746106123 w_k1_INTC_s007
                                 + 0.00541777813392664 w_k1_INTC_s010
                                 + 0.0131435529606054 w_k1_INTC_s011
                                 + 0.00654352180508093 w_k1_INTC_s012
                                 + 0.0119183750014454 w_k1_INTC_s013
                                 + 0.0105375968876678 w_k1_INTC_s014
                                 + 0.00648760690348257 w_k1_INTC_s015
                                 + 0.00901531032805442 w_k1_INTC_s016
                                 + 0.0116862784909143 w_k1_INTC_s017
                                 + 0.0303960451232915 w_k1_INTC_s100
                                 + 0.0397821196927579 w_k1_INTC_s101
                                 + 0.0181499182277512 w_k1_INTC_s102
                                 + 0.0414885561470029 w_k1_INTC_s103
                                 + 0.0573202749586849 w_k1_INTC_s104
                                 + 0.0259469210674728 w_k1_INTC_s105
                                 + 0.07254449397669 w_k1_INTC_s106
                                 + 0.0379915116268381 w_k1_INTC_s107
                                 + 0.0158061747107688 w_k1_INTC_s110
                                 + 0.0176564373343035 w_k1_INTC_s111
                                 + 0.0252898034556478 w_k1_INTC_s112
                                 + 0.0359484115263654 w_k1_INTC_s113
                                 + 0.0338805551674625 w_k1_INTC_s114
                                 + 0.0207811319063923 w_k1_INTC_s115
                                 + 0.0741348810438428 w_k1_INTC_s116
                                 + 0.0149552246253296 w_k1_INTC_s117
                                 + 0.032563664631555 w_k1_KO_s000
                                 + 0.0728966523039444 w_k1_KO_s001
                                 + 0.0189159388003995 w_k1_KO_s002
                                 + 0.0395294840052337 w_k1_KO_s003
                                 + 0.0771602105112393 w_k1_KO_s004
                                 + 0.0438541937061943 w_k1_KO_s005
                                 + 0.0373320707610912 w_k1_KO_s006
                                 + 0.0572530024707417 w_k1_KO_s007
                                 + 0.00459234783856385 w_k1_KO_s010
                                 + 0.0133087514087611 w_k1_KO_s011
                                 + 0.00628337085651387 w_k1_KO_s012
                                 + 0.0139111378395515 w_k1_KO_s013
                                 + 0.012900751130834 w_k1_KO_s014
                                 + 0.007108102613773 w_k1_KO_s015
                                 + 0.00951722346545322 w_k1_KO_s016
                                 + 0.0131033867115653 w_k1_KO_s017
                                 + 0.034088971862451 w_k1_KO_s100
                                 + 0.0356955083101646 w_k1_KO_s101
                                 + 0.018145378262521 w_k1_KO_s102
                                 + 0.0454703881773068 w_k1_KO_s103
                                 + 0.0533864501806127 w_k1_KO_s104
                                 + 0.0239349578142063 w_k1_KO_s105
                                 + 0.0623460310346997 w_k1_KO_s106
                                 + 0.0407164648976636 w_k1_KO_s107
                                 + 0.0159540573714664 w_k1_KO_s110
                                 + 0.0177860688263193 w_k1_KO_s111
                                 + 0.0283881814254285 w_k1_KO_s112
                                 + 0.0304772690329231 w_k1_KO_s113
                                 + 0.0316845068288496 w_k1_KO_s114
                                 + 0.022278412129458 w_k1_KO_s115
                                 + 0.0670322689251452 w_k1_KO_s116
                                 + 0.0178987229733398 w_k1_KO_s117
                                 + 0.037795589580462 w_k1_MAT_s000
                                 + 0.0803167174762998 w_k1_MAT_s001
                                 + 0.0164339819286002 w_k1_MAT_s002
                                 + 0.048785376842514 w_k1_MAT_s003
                                 + 0.0930132325699118 w_k1_MAT_s004
                                 + 0.0435139913282828 w_k1_MAT_s005
                                 + 0.0447631288160295 w_k1_MAT_s006
                                 + 0.0612746040292904 w_k1_MAT_s007
                                 + 0.00491305406989831 w_k1_MAT_s010
                                 + 0.0149716054697195 w_k1_MAT_s011
                                 + 0.00613431074806591 w_k1_MAT_s012
                                 + 0.0122692806299609 w_k1_MAT_s013
                                 + 0.0128444925355643 w_k1_MAT_s014
                                 + 0.00966386299823964 w_k1_MAT_s015
                                 + 0.00926456827173453 w_k1_MAT_s016
                                 + 0.0148894299004314 w_k1_MAT_s017
                                 + 0.0267906493191111 w_k1_MAT_s100
                                 + 0.0367649931488486 w_k1_MAT_s101
                                 + 0.0177126373889278 w_k1_MAT_s102
                                 + 0.0514578677508281 w_k1_MAT_s103
                                 + 0.0467958638697124 w_k1_MAT_s104
                                 + 0.0215398023288519 w_k1_MAT_s105
                                 + 0.0680141023150633 w_k1_MAT_s106
                                 + 0.0386549097363761 w_k1_MAT_s107
                                 + 0.0176153386011963 w_k1_MAT_s110
                                 + 0.0176781345048934 w_k1_MAT_s111
                                 + 0.0317520790239191 w_k1_MAT_s112
                                 + 0.0224805434287329 w_k1_MAT_s113
                                 + 0.0380315096252882 w_k1_MAT_s114
                                 + 0.0245616233427141 w_k1_MAT_s115
                                 + 0.0648987785348892 w_k1_MAT_s116
                                 + 0.0156643093784158 w_k1_MAT_s117
                                 + 0.0343224777763014 w_k1_MSFT_s000
                                 + 0.0777450352919265 w_k1_MSFT_s001
                                 + 0.0173357154255652 w_k1_MSFT_s002
                                 + 0.0422624794718534 w_k1_MSFT_s003
                                 + 0.0810807089963232 w_k1_MSFT_s004
                                 + 0.0470084000545748 w_k1_MSFT_s005
                                 + 0.0391237685831323 w_k1_MSFT_s006
                                 + 0.0587476495384013 w_k1_MSFT_s007
                                 + 0.0047917728271818 w_k1_MSFT_s010
                                 + 0.0132403213870464 w_k1_MSFT_s011
                                 + 0.0059330904687502 w_k1_MSFT_s012
                                 + 0.0119772985735075 w_k1_MSFT_s013
                                 + 0.0117568180000543 w_k1_MSFT_s014
                                 + 0.00863994861598204 w_k1_MSFT_s015
                                 + 0.00904629803606165 w_k1_MSFT_s016
                                 + 0.0140646296840352 w_k1_MSFT_s017
                                 + 0.0303596047702176 w_k1_MSFT_s100
                                 + 0.0379775882799064 w_k1_MSFT_s101
                                 + 0.0178307023884698 w_k1_MSFT_s102
                                 + 0.044740687630308 w_k1_MSFT_s103
                                 + 0.051089505292721 w_k1_MSFT_s104
                                 + 0.0237072078549769 w_k1_MSFT_s105
                                 + 0.0604230480630211 w_k1_MSFT_s106
                                 + 0.0397866371198737 w_k1_MSFT_s107
                                 + 0.0180182676177771 w_k1_MSFT_s110
                                 + 0.0196141759336884 w_k1_MSFT_s111
                                 + 0.0298560295708708 w_k1_MSFT_s112
                                 + 0.0300872577101303 w_k1_MSFT_s113
                                 + 0.0364059080212046 w_k1_MSFT_s114
                                 + 0.0189099499173207 w_k1_MSFT_s115
                                 + 0.0622861993278232 w_k1_MSFT_s116
                                 + 0.0169307775575941 w_k1_MSFT_s117
                                 + 0.0328269408932267 w_k1_PG_s000
                                 + 0.0772072085127135 w_k1_PG_s001
                                 + 0.019208897578408 w_k1_PG_s002
                                 + 0.0423717089460661 w_k1_PG_s003
                                 + 0.0832480758836543 w_k1_PG_s004
                                 + 0.0431790345231537 w_k1_PG_s005
                                 + 0.0364315088481648 w_k1_PG_s006
                                 + 0.0585076834137609 w_k1_PG_s007
                                 + 0.00455101309328905 w_k1_PG_s010
                                 + 0.0130114245601122 w_k1_PG_s011
                                 + 0.00637271750835644 w_k1_PG_s012
                                 + 0.0137576580729274 w_k1_PG_s013
                                 + 0.0132803428160035 w_k1_PG_s014
                                 + 0.007923399201801 w_k1_PG_s015
                                 + 0.00874422694877301 w_k1_PG_s016
                                 + 0.0135206063019882 w_k1_PG_s017
                                 + 0.0322399038893783 w_k1_PG_s100
                                 + 0.0377154516645774 w_k1_PG_s101
                                 + 0.0161051892827801 w_k1_PG_s102
                                 + 0.0424770634465465 w_k1_PG_s103
                                 + 0.0500745074738968 w_k1_PG_s104
                                 + 0.0253998942888106 w_k1_PG_s105
                                 + 0.0597260250190664 w_k1_PG_s106
                                 + 0.0400763255323762 w_k1_PG_s107
                                 + 0.0155363006589611 w_k1_PG_s110
                                 + 0.015497251252358 w_k1_PG_s111
                                 + 0.0293286111238988 w_k1_PG_s112
                                 + 0.0296159192385757 w_k1_PG_s113
                                 + 0.0346802399153075 w_k1_PG_s114
                                 + 0.0213402695263826 w_k1_PG_s115
                                 + 0.0688107587578164 w_k1_PG_s116
                                 + 0.0168672660163286 w_k1_PG_s117 >= 1.12
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
