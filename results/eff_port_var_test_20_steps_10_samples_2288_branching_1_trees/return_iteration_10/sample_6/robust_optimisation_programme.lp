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
 w_k1_ADM_s0_return_constraint:  1.13822 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.941855 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.0223 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.09222 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.783826 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.09533 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.33684 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.00462 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.25665 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.802798 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.963005 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.2904 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.791 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.741059 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.894463 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.01362 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.727236 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.988088 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.13263 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.18238 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.855432 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.31723 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.17387 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.36346 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.23476 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.919783 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.832217 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.862284 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.987264 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.15203 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.05916 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.04174 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.9707 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.966339 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.23246 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.858187 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.11395 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.25084 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.28106 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.896745 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.98974 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.980109 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.823007 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.23222 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.50442 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.89823 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.23839 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.655454 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.981566 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.49549 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.694648 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.713808 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.859474 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.1594 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.676434 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.894096 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.28581 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.40921 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.779647 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.87535 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.0483 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.21343 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.53025 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.845465 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.713678 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.68436 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.11946 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.947489 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.08294 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.971702 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.973379 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.977505 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.30112 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.669408 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.00245 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.52267 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.30852 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.858688 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.06041 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.03671 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.827429 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.29128 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.45091 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.716097 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.24762 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.513673 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.13689 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.45191 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.828916 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.551773 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.913113 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.951676 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.526572 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.4911 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.32695 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.33602 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.807402 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 2.23569 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.06914 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.40274 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.51561 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.800419 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.596549 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.65504 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.902584 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.880576 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.939753 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.23703 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.837034 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.88372 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.47185 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.684371 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.21988 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.55896 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19626 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.906101 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.998828 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.07879 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.867082 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.14204 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.39636 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.903973 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.09116 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.76829 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.02099 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.26152 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.863786 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.653556 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.850627 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.937579 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.69189 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.16641 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.17936 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.37188 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.910548 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.56658 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.04745 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.39454 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.27441 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.839951 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.671359 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.785168 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.881501 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.03852 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.09634 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.09832 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.03304 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.897616 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.38135 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.83289 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.983974 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.20702 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15759 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.945574 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.15141 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.734755 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.868092 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.19797 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.866752 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.734279 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.65029 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.477154 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.03524 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.21579 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.729769 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.903425 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.10154 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.15169 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.739648 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.02348 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.36682 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.864145 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.762506 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.83569 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.990184 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.747946 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.53246 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.23503 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.913918 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.689384 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.70523 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.595656 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.937853 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.770217 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.700182 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.21742 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.922404 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.665397 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.39946 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 2.03472 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.20957 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.967397 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.904048 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.937585 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.87197 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.1315 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.23633 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.00293 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.00277 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.816684 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.982355 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.32699 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.93842 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.815147 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.819752 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.03305 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.711884 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.989858 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.941419 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.11062 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.978516 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.43346 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.970362 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.27166 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.33039 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.87782 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.78673 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.798997 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.987766 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.885482 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.01548 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.933313 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.880252 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.995211 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.34153 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.777494 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.971018 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.951496 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.2237 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.888281 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.985621 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.859017 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.907956 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.18002 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.21988 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.792568 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.43612 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.59699 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.16285 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.34289 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.867526 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.770618 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.965979 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.08165 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.661147 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.24193 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.22555 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.01401 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.811006 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.74284 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.00628 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.992156 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.33744 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.921355 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.84063 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.692271 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.1106 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.783816 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.849981 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.1299 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.837766 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.14 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.25761 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.760882 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.37442 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.67451 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.32009 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.894909 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.02895 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.761587 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.857579 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.3091 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.983297 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.672592 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.87389 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.369978 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.18145 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.4144 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.695974 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.884689 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.89468 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.1102 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.559163 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.883138 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.52694 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.928908 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.507841 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.92309 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.26185 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.852535 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.55689 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.12593 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.744885 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.756055 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.52053 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.643359 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.796767 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.968943 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.728049 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.0735 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.917058 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.565501 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.43988 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 2.40918 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19088 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.938888 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.915096 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.02808 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.913522 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.13271 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.30502 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.955168 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.28836 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.755796 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.16754 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.38591 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.958458 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.747444 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.908135 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.925515 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.667703 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.16309 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.10452 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.21511 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.869584 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.51893 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.04248 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.25823 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.16286 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.934684 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.704999 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.743341 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.822509 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.936562 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.959304 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.10034 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.966811 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.98665 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.26058 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.895811 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.10772 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.20325 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13526 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.970076 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.973848 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.951266 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.907668 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.05223 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.24915 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.964201 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.14711 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.790314 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.05594 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.12538 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.967606 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.877172 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.937261 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.00268 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.836918 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.00054 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.19697 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.09572 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.893837 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.44243 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.969889 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.05542 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.14998 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.873095 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.772611 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.83177 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.968396 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.944047 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.01097 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.12075 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.959613 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.03 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.28701 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.927254 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.13005 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.29207 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.464997520287551 z_k1_s0000
                                 - 0.142999387793209 z_k1_s0001
                                 - 0.315999333705345 z_k1_s0002
                                 - 0.248999095649072 z_k1_s0003
                                 - 0.185999293513887 z_k1_s0004
                                 - 0.332997690233849 z_k1_s0005
                                 - 0.669996672585389 z_k1_s0006
                                 - 0.49199910711028 z_k1_s0007
                                 - 0.637000380519663 z_k1_s0010
                                 - 0.753000662615974 z_k1_s0011
                                 - 0.196000236252265 z_k1_s0012
                                 - 0.309000103720015 z_k1_s0013
                                 - 0.858001619108682 z_k1_s0014
                                 - 0.464001128385243 z_k1_s0015
                                 - 0.217000040250609 z_k1_s0016
                                 - 0.439000900648884 z_k1_s0017
                                 - 0.305000102052304 z_k1_s0020
                                 - 0.566002716524046 z_k1_s0021
                                 - 0.326000499667474 z_k1_s0022
                                 - 0.844999998974111 z_k1_s0023
                                 - 1.06600054696793 z_k1_s0024
                                 - 0.502000632000685 z_k1_s0025
                                 - 0.927000824280179 z_k1_s0026
                                 - 0.553998125596393 z_k1_s0027
                                 - 0.0920000896492022 z_k1_s0030
                                 - 0.182000319230161 z_k1_s0031
                                 - 0.22000063753135 z_k1_s0032
                                 - 0.0730001594989591 z_k1_s0033
                                 - 0.224000128667485 z_k1_s0034
                                 - 0.190000446504188 z_k1_s0035
                                 - 0.0740000895330807 z_k1_s0036
                                 - 0.0900001150807833 z_k1_s0037
                                 - 0.402000628310258 z_k1_s0040
                                 - 0.350000454526683 z_k1_s0041
                                 - 0.379000583952194 z_k1_s0042
                                 - 0.236000528648086 z_k1_s0043
                                 - 0.52600266810464 z_k1_s0044
                                 - 0.647999044866899 z_k1_s0045
                                 - 0.811000369797516 z_k1_s0046
                                 - 0.874001685606431 z_k1_s0047
                                 - 0.526000191432123 z_k1_s0050
                                 - 0.451999117285659 z_k1_s0051
                                 - 0.254999633722437 z_k1_s0052
                                 - 0.325999769572594 z_k1_s0053
                                 - 0.387000579828591 z_k1_s0054
                                 - 0.29700025089204 z_k1_s0055
                                 - 0.1549997009922 z_k1_s0056
                                 - 0.383998806933703 z_k1_s0057
                                 - 0.783001765039284 z_k1_s0060
                                 - 0.440000355447812 z_k1_s0061
                                 - 0.489000017530524 z_k1_s0062
                                 - 0.44600262530232 z_k1_s0063
                                 - 0.39899994482529 z_k1_s0064
                                 - 0.618003519252598 z_k1_s0065
                                 - 0.197000536644928 z_k1_s0066
                                 - 0.403001458061628 z_k1_s0067
                                 - 0.894997606010986 z_k1_s0070
                                 - 0.74899741128871 z_k1_s0071
                                 - 0.484998919457985 z_k1_s0072
                                 - 0.340999248278689 z_k1_s0073
                                 - 0.468000677328821 z_k1_s0074
                                 - 0.190999558466995 z_k1_s0075
                                 - 0.235999310210849 z_k1_s0076
                                 - 0.514999012619978 z_k1_s0077
                                 - 0.0920001720372741 z_k1_s0100
                                 - 0.0549999386137448 z_k1_s0101
                                 - 0.0769997943793572 z_k1_s0102
                                 - 0.0749996442917303 z_k1_s0103
                                 - 0.162999516153751 z_k1_s0104
                                 - 0.0659999412964795 z_k1_s0105
                                 - 0.0759997193355437 z_k1_s0106
                                 - 0.143999586319865 z_k1_s0107
                                 - 0.0889999758180653 z_k1_s0110
                                 - 0.185999066472319 z_k1_s0111
                                 - 0.335999904883936 z_k1_s0112
                                 - 0.168999799271533 z_k1_s0113
                                 - 0.115999922612354 z_k1_s0114
                                 - 0.246000082236305 z_k1_s0115
                                 - 0.18200001935373 z_k1_s0116
                                 - 0.376999068090812 z_k1_s0117
                                 - 0.399997802306188 z_k1_s0120
                                 - 0.1299995144985 z_k1_s0121
                                 - 0.188999734761257 z_k1_s0122
                                 - 0.331998962810659 z_k1_s0123
                                 - 0.211999073121229 z_k1_s0124
                                 - 0.161999302083157 z_k1_s0125
                                 - 0.233999235896815 z_k1_s0126
                                 - 0.170999202310269 z_k1_s0127
                                 - 0.103000035685865 z_k1_s0130
                                 - 0.0689996896835404 z_k1_s0131
                                 - 0.07899979144893 z_k1_s0132
                                 - 0.101000015332787 z_k1_s0133
                                 - 0.159999603250938 z_k1_s0134
                                 - 0.0469998707987424 z_k1_s0135
                                 - 0.0709997100366184 z_k1_s0136
                                 - 0.0449999179455077 z_k1_s0137
                                 - 0.339000197422021 z_k1_s0140
                                 - 0.28500024556715 z_k1_s0141
                                 - 0.0899999197581179 z_k1_s0142
                                 - 0.147999951324423 z_k1_s0143
                                 - 0.259000102192657 z_k1_s0144
                                 - 0.136000011989985 z_k1_s0145
                                 - 0.0939999994695526 z_k1_s0146
                                 - 0.147999951324423 z_k1_s0147
                                 - 0.0969997147237577 z_k1_s0150
                                 - 0.170000230483018 z_k1_s0151
                                 - 0.132999694067403 z_k1_s0152
                                 - 0.145999681922065 z_k1_s0153
                                 - 0.445999453552588 z_k1_s0154
                                 - 0.352998761825128 z_k1_s0155
                                 - 0.134999627314444 z_k1_s0156
                                 - 0.206999417302177 z_k1_s0157
                                 - 0.292001179584733 z_k1_s0160
                                 - 0.272999758409899 z_k1_s0161
                                 - 0.386000933608678 z_k1_s0162
                                 - 0.366999512433844 z_k1_s0163
                                 - 0.399000216675921 z_k1_s0164
                                 - 0.294001069287386 z_k1_s0165
                                 - 0.230999852554521 z_k1_s0166
                                 - 0.227000073149216 z_k1_s0167
                                 - 0.0989999083097494 z_k1_s0170
                                 - 0.0649999053690927 z_k1_s0171
                                 - 0.0459999564903903 z_k1_s0172
                                 - 0.0719999006980614 z_k1_s0173
                                 - 0.170000014703284 z_k1_s0174
                                 - 0.0599998574484979 z_k1_s0175
                                 - 0.182999314058395 z_k1_s0176
                                 - 0.201999532036587 z_k1_s0177
                                 - 0.710000680980186 z_k1_s1000
                                 - 0.638000373166217 z_k1_s1001
                                 - 1.68200527336915 z_k1_s1002
                                 - 2.04900158742475 z_k1_s1003
                                 - 0.518000442676766 z_k1_s1004
                                 - 1.19500116461737 z_k1_s1005
                                 - 0.634000113343011 z_k1_s1006
                                 - 1.31200308474129 z_k1_s1007
                                 - 0.325001045020162 z_k1_s1010
                                 - 0.334999831943993 z_k1_s1011
                                 - 0.444999978104685 z_k1_s1012
                                 - 0.372000199961646 z_k1_s1013
                                 - 0.262000053800954 z_k1_s1014
                                 - 0.182000045611349 z_k1_s1015
                                 - 0.149000001763141 z_k1_s1016
                                 - 0.627999902408417 z_k1_s1017
                                 - 0.252000135806389 z_k1_s1020
                                 - 0.826002092343204 z_k1_s1021
                                 - 0.784001899308801 z_k1_s1022
                                 - 0.222999759092153 z_k1_s1023
                                 - 0.451999983932088 z_k1_s1024
                                 - 0.645998343909979 z_k1_s1025
                                 - 0.857998101131218 z_k1_s1026
                                 - 1.07099981037713 z_k1_s1027
                                 - 0.645001008383223 z_k1_s1030
                                 - 0.332000413750361 z_k1_s1031
                                 - 0.243000733387524 z_k1_s1032
                                 - 0.445000018820072 z_k1_s1033
                                 - 0.212000790216 z_k1_s1034
                                 - 0.888004652381335 z_k1_s1035
                                 - 0.209000920903082 z_k1_s1036
                                 - 0.260000963030921 z_k1_s1037
                                 - 0.240000816123712 z_k1_s1040
                                 - 0.26499971257441 z_k1_s1041
                                 - 0.113999912233888 z_k1_s1042
                                 - 0.296998976191126 z_k1_s1043
                                 - 0.52999942514882 z_k1_s1044
                                 - 0.12800001266609 z_k1_s1045
                                 - 0.25200053997998 z_k1_s1046
                                 - 0.287000262810626 z_k1_s1047
                                 - 0.397001560056114 z_k1_s1050
                                 - 0.401001349137774 z_k1_s1051
                                 - 0.323000941780059 z_k1_s1052
                                 - 0.226000782906902 z_k1_s1053
                                 - 0.976002779109178 z_k1_s1054
                                 - 0.580001704450306 z_k1_s1055
                                 - 0.643001288371315 z_k1_s1056
                                 - 0.759004212270147 z_k1_s1057
                                 - 0.341999643751885 z_k1_s1060
                                 - 0.150999966449535 z_k1_s1061
                                 - 0.644000080451032 z_k1_s1062
                                 - 0.324998910179309 z_k1_s1063
                                 - 0.234000021409225 z_k1_s1064
                                 - 0.235000064560553 z_k1_s1065
                                 - 0.149999923298207 z_k1_s1066
                                 - 0.438001267278991 z_k1_s1067
                                 - 0.779001598013881 z_k1_s1070
                                 - 0.332999852632431 z_k1_s1071
                                 - 0.249000137622433 z_k1_s1072
                                 - 0.224000145626605 z_k1_s1073
                                 - 0.764999890912327 z_k1_s1074
                                 - 0.534999013670771 z_k1_s1075
                                 - 0.313999913051599 z_k1_s1076
                                 - 0.196999909759126 z_k1_s1077
                                 - 0.238000010541867 z_k1_s1100
                                 - 0.375000021776052 z_k1_s1101
                                 - 0.397999052626154 z_k1_s1102
                                 - 0.376999189110718 z_k1_s1103
                                 - 0.178999984165098 z_k1_s1104
                                 - 0.137000011234185 z_k1_s1105
                                 - 0.61700008828754 z_k1_s1106
                                 - 0.1380000867016 z_k1_s1107
                                 - 0.144999921594997 z_k1_s1110
                                 - 0.306999652240455 z_k1_s1111
                                 - 0.223000160920563 z_k1_s1112
                                 - 0.211000692731976 z_k1_s1113
                                 - 0.165000320842555 z_k1_s1114
                                 - 0.104000051267439 z_k1_s1115
                                 - 0.252000080959567 z_k1_s1116
                                 - 0.200000778475798 z_k1_s1117
                                 - 0.410000381778455 z_k1_s1120
                                 - 0.256999594457072 z_k1_s1121
                                 - 0.536998068020245 z_k1_s1122
                                 - 0.477000499439742 z_k1_s1123
                                 - 0.721999069382638 z_k1_s1124
                                 - 0.290999727450766 z_k1_s1125
                                 - 0.499998623146804 z_k1_s1126
                                 - 0.582998092429558 z_k1_s1127
                                 - 1.07799910382584 z_k1_s1130
                                 - 0.794000295620324 z_k1_s1131
                                 - 0.698999197895204 z_k1_s1132
                                 - 0.748998081169947 z_k1_s1133
                                 - 0.708000928586523 z_k1_s1134
                                 - 1.09300198831137 z_k1_s1135
                                 - 0.82399962558517 z_k1_s1136
                                 - 0.49400055696565 z_k1_s1137
                                 - 0.628003820555073 z_k1_s1140
                                 - 0.828004606972453 z_k1_s1141
                                 - 0.458999665420757 z_k1_s1142
                                 - 0.742000440400223 z_k1_s1143
                                 - 0.180000707775641 z_k1_s1144
                                 - 0.46099944808418 z_k1_s1145
                                 - 0.877003786241334 z_k1_s1146
                                 - 0.329001181056214 z_k1_s1147
                                 - 0.902003116230773 z_k1_s1150
                                 - 0.962000207396324 z_k1_s1151
                                 - 0.203000690867877 z_k1_s1152
                                 - 0.924003307261059 z_k1_s1153
                                 - 0.320000677853828 z_k1_s1154
                                 - 0.238000566051428 z_k1_s1155
                                 - 0.463000976348497 z_k1_s1156
                                 - 0.704001396958202 z_k1_s1157
                                 - 0.253000239544741 z_k1_s1160
                                 - 0.756999073034631 z_k1_s1161
                                 - 0.410999073247124 z_k1_s1162
                                 - 0.4609990392164 z_k1_s1163
                                 - 0.356999345280424 z_k1_s1164
                                 - 0.277000282030023 z_k1_s1165
                                 - 0.241000071252027 z_k1_s1166
                                 - 0.184000227687111 z_k1_s1167
                                 - 0.158000035147416 z_k1_s1170
                                 - 0.238999472742116 z_k1_s1171
                                 - 0.236001093460448 z_k1_s1172
                                 - 0.537000951348929 z_k1_s1173
                                 - 0.274000584356786 z_k1_s1174
                                 - 0.210999969450049 z_k1_s1175
                                 - 0.446000833150124 z_k1_s1176
                                 - 0.208999971929194 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.960841 w_k1_ADM_s000
                                 + 1.06665 w_k1_AXP_s000
                                 + 1.11698 w_k1_DIS_s000
                                 + 1.08358 w_k1_FDX_s000
                                 + 1.34175 w_k1_IBM_s000
                                 + 1.21882 w_k1_INTC_s000
                                 + 1.18652 w_k1_KO_s000 + 1.2277 w_k1_MAT_s000
                                 + 1.2381 w_k1_MSFT_s000 + 1.00653 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.4237 w_k1_ADM_s000
                                 + 1.56061 w_k1_AXP_s000
                                 + 1.26027 w_k1_DIS_s000
                                 + 1.33015 w_k1_FDX_s000
                                 + 1.15053 w_k1_IBM_s000 + 1.387 w_k1_INTC_s000
                                 + 1.04749 w_k1_KO_s000 + 1.52511 w_k1_MAT_s000
                                 + 1.18803 w_k1_MSFT_s000
                                 + 1.14139 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.34267 w_k1_ADM_s000
                                 + 1.39056 w_k1_AXP_s000
                                 + 1.89141 w_k1_DIS_s000 + 1.4164 w_k1_FDX_s000
                                 + 1.57883 w_k1_IBM_s000
                                 + 1.36624 w_k1_INTC_s000 + 1.6793 w_k1_KO_s000
                                 + 1.95806 w_k1_MAT_s000
                                 + 1.65602 w_k1_MSFT_s000
                                 + 1.14194 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.04822 w_k1_ADM_s000
                                 + 0.89893 w_k1_AXP_s000
                                 + 1.03794 w_k1_DIS_s000
                                 + 1.12952 w_k1_FDX_s000
                                 + 0.814354 w_k1_IBM_s000
                                 + 0.886875 w_k1_INTC_s000
                                 + 1.01269 w_k1_KO_s000
                                 + 0.932857 w_k1_MAT_s000
                                 + 1.1009 w_k1_MSFT_s000 + 1.0168 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.15363 w_k1_ADM_s000
                                 + 1.01936 w_k1_AXP_s000
                                 + 1.37506 w_k1_DIS_s000
                                 + 1.19068 w_k1_FDX_s000
                                 + 0.968999 w_k1_IBM_s000
                                 + 1.28877 w_k1_INTC_s000
                                 + 1.05438 w_k1_KO_s000
                                 + 0.906551 w_k1_MAT_s000
                                 + 1.26977 w_k1_MSFT_s000
                                 + 0.951115 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.743858 w_k1_ADM_s000
                                 + 0.672955 w_k1_AXP_s000
                                 + 0.731133 w_k1_DIS_s000
                                 + 0.712731 w_k1_FDX_s000
                                 + 0.925126 w_k1_IBM_s000
                                 + 0.860956 w_k1_INTC_s000
                                 + 0.827291 w_k1_KO_s000
                                 + 0.865689 w_k1_MAT_s000
                                 + 0.893755 w_k1_MSFT_s000
                                 + 0.871167 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.985189 w_k1_ADM_s000
                                 + 0.714203 w_k1_AXP_s000
                                 + 0.717326 w_k1_DIS_s000
                                 + 0.843331 w_k1_FDX_s000
                                 + 0.743897 w_k1_IBM_s000
                                 + 0.883999 w_k1_INTC_s000
                                 + 0.679074 w_k1_KO_s000
                                 + 0.952435 w_k1_MAT_s000
                                 + 0.830755 w_k1_MSFT_s000
                                 + 0.742122 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.775387 w_k1_ADM_s000
                                 + 1.00377 w_k1_AXP_s000
                                 + 0.755124 w_k1_DIS_s000
                                 + 0.936004 w_k1_FDX_s000
                                 + 1.10032 w_k1_IBM_s000
                                 + 1.04551 w_k1_INTC_s000
                                 + 0.981249 w_k1_KO_s000
                                 + 1.37269 w_k1_MAT_s000
                                 + 1.0563 w_k1_MSFT_s000
                                 + 0.939462 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.792973 w_k1_ADM_s001
                                 + 0.67195 w_k1_AXP_s001
                                 + 0.599677 w_k1_DIS_s001
                                 + 0.789513 w_k1_FDX_s001
                                 + 0.742988 w_k1_IBM_s001
                                 + 0.718232 w_k1_INTC_s001
                                 + 0.809357 w_k1_KO_s001
                                 + 0.929768 w_k1_MAT_s001
                                 + 0.8022 w_k1_MSFT_s001
                                 + 0.837466 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.0548 w_k1_ADM_s001
                                 + 1.29998 w_k1_AXP_s001
                                 + 1.13865 w_k1_DIS_s001
                                 + 1.09207 w_k1_FDX_s001
                                 + 1.25507 w_k1_IBM_s001
                                 + 1.10955 w_k1_INTC_s001
                                 + 1.17637 w_k1_KO_s001 + 1.4531 w_k1_MAT_s001
                                 + 0.953535 w_k1_MSFT_s001
                                 + 1.1283 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.933967 w_k1_ADM_s001
                                 + 1.5222 w_k1_AXP_s001 + 1.53878 w_k1_DIS_s001
                                 + 1.18351 w_k1_FDX_s001 + 1.6845 w_k1_IBM_s001
                                 + 1.14642 w_k1_INTC_s001
                                 + 1.44221 w_k1_KO_s001 + 1.94834 w_k1_MAT_s001
                                 + 1.16161 w_k1_MSFT_s001 + 1.0856 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.25466 w_k1_ADM_s001
                                 + 1.56313 w_k1_AXP_s001 + 1.6684 w_k1_DIS_s001
                                 + 1.47856 w_k1_FDX_s001
                                 + 1.01477 w_k1_IBM_s001
                                 + 1.15726 w_k1_INTC_s001
                                 + 1.24965 w_k1_KO_s001 + 1.34423 w_k1_MAT_s001
                                 + 1.26353 w_k1_MSFT_s001
                                 + 1.12565 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.0672 w_k1_ADM_s001
                                 + 1.01665 w_k1_AXP_s001
                                 + 1.16041 w_k1_DIS_s001
                                 + 1.03457 w_k1_FDX_s001
                                 + 1.05329 w_k1_IBM_s001
                                 + 0.799806 w_k1_INTC_s001
                                 + 1.12015 w_k1_KO_s001 + 1.1784 w_k1_MAT_s001
                                 + 0.975099 w_k1_MSFT_s001
                                 + 0.999349 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.754662 w_k1_ADM_s001
                                 + 0.871112 w_k1_AXP_s001
                                 + 0.750818 w_k1_DIS_s001
                                 + 0.792618 w_k1_FDX_s001
                                 + 1.15325 w_k1_IBM_s001
                                 + 0.8502 w_k1_INTC_s001 + 1.10523 w_k1_KO_s001
                                 + 1.12319 w_k1_MAT_s001
                                 + 0.84499 w_k1_MSFT_s001
                                 + 0.892522 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.07787 w_k1_ADM_s001
                                 + 0.914415 w_k1_AXP_s001
                                 + 0.960477 w_k1_DIS_s001
                                 + 1.14543 w_k1_FDX_s001
                                 + 0.561248 w_k1_IBM_s001
                                 + 1.04918 w_k1_INTC_s001
                                 + 0.801516 w_k1_KO_s001 + 0.6111 w_k1_MAT_s001
                                 + 1.16719 w_k1_MSFT_s001
                                 + 0.90199 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.954186 w_k1_ADM_s001
                                 + 0.978005 w_k1_AXP_s001
                                 + 0.886165 w_k1_DIS_s001
                                 + 1.01469 w_k1_FDX_s001
                                 + 0.772361 w_k1_IBM_s001
                                 + 0.829379 w_k1_INTC_s001
                                 + 0.766348 w_k1_KO_s001
                                 + 1.17169 w_k1_MAT_s001
                                 + 0.882247 w_k1_MSFT_s001
                                 + 0.847139 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.998882 w_k1_ADM_s002
                                 + 1.45999 w_k1_AXP_s002
                                 + 1.81582 w_k1_DIS_s002
                                 + 1.25088 w_k1_FDX_s002
                                 + 1.46556 w_k1_IBM_s002
                                 + 1.31572 w_k1_INTC_s002
                                 + 1.32036 w_k1_KO_s002 + 1.57533 w_k1_MAT_s002
                                 + 1.21034 w_k1_MSFT_s002
                                 + 1.27775 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.79451 w_k1_ADM_s002
                                 + 0.705725 w_k1_AXP_s002
                                 + 0.608055 w_k1_DIS_s002
                                 + 0.714848 w_k1_FDX_s002
                                 + 0.638452 w_k1_IBM_s002
                                 + 0.817682 w_k1_INTC_s002
                                 + 0.724336 w_k1_KO_s002
                                 + 0.605773 w_k1_MAT_s002
                                 + 0.689055 w_k1_MSFT_s002
                                 + 0.812806 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.741836 w_k1_ADM_s002
                                 + 1.12263 w_k1_AXP_s002
                                 + 1.27397 w_k1_DIS_s002
                                 + 0.93742 w_k1_FDX_s002
                                 + 1.77536 w_k1_IBM_s002
                                 + 1.15608 w_k1_INTC_s002
                                 + 1.51924 w_k1_KO_s002 + 1.47946 w_k1_MAT_s002
                                 + 1.06507 w_k1_MSFT_s002
                                 + 1.19015 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.876309 w_k1_ADM_s002
                                 + 1.26095 w_k1_AXP_s002
                                 + 1.41054 w_k1_DIS_s002
                                 + 1.22368 w_k1_FDX_s002
                                 + 0.831514 w_k1_IBM_s002
                                 + 1.38453 w_k1_INTC_s002
                                 + 1.06143 w_k1_KO_s002
                                 + 0.902979 w_k1_MAT_s002
                                 + 1.11397 w_k1_MSFT_s002
                                 + 1.15197 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.897632 w_k1_ADM_s002
                                 + 0.756647 w_k1_AXP_s002
                                 + 0.92635 w_k1_DIS_s002
                                 + 0.828339 w_k1_FDX_s002
                                 + 0.985721 w_k1_IBM_s002
                                 + 0.95057 w_k1_INTC_s002
                                 + 0.848377 w_k1_KO_s002
                                 + 0.716883 w_k1_MAT_s002
                                 + 0.765385 w_k1_MSFT_s002
                                 + 0.900567 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.964623 w_k1_ADM_s002
                                 + 1.39698 w_k1_AXP_s002
                                 + 0.924326 w_k1_DIS_s002
                                 + 1.03387 w_k1_FDX_s002
                                 + 0.995669 w_k1_IBM_s002
                                 + 1.14964 w_k1_INTC_s002
                                 + 0.943416 w_k1_KO_s002
                                 + 1.20913 w_k1_MAT_s002
                                 + 0.976117 w_k1_MSFT_s002 + 1.034 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.10199 w_k1_ADM_s002
                                 + 1.42009 w_k1_AXP_s002
                                 + 1.45428 w_k1_DIS_s002 + 1.2281 w_k1_FDX_s002
                                 + 1.26245 w_k1_IBM_s002
                                 + 1.25886 w_k1_INTC_s002
                                 + 0.999488 w_k1_KO_s002
                                 + 1.06286 w_k1_MAT_s002
                                 + 1.02947 w_k1_MSFT_s002
                                 + 1.13472 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.709972 w_k1_ADM_s002
                                 + 0.859826 w_k1_AXP_s002
                                 + 0.626681 w_k1_DIS_s002
                                 + 0.81424 w_k1_FDX_s002
                                 + 0.788931 w_k1_IBM_s002
                                 + 1.01646 w_k1_INTC_s002
                                 + 0.724658 w_k1_KO_s002
                                 + 0.681984 w_k1_MAT_s002
                                 + 0.705447 w_k1_MSFT_s002
                                 + 0.937871 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.993067 w_k1_ADM_s003
                                 + 0.792585 w_k1_AXP_s003
                                 + 1.36388 w_k1_DIS_s003
                                 + 0.99901 w_k1_FDX_s003
                                 + 1.29112 w_k1_IBM_s003
                                 + 1.11142 w_k1_INTC_s003
                                 + 1.28674 w_k1_KO_s003 + 1.12029 w_k1_MAT_s003
                                 + 1.29492 w_k1_MSFT_s003 + 1.0522 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.914679 w_k1_ADM_s003
                                 + 1.19004 w_k1_AXP_s003
                                 + 1.03727 w_k1_DIS_s003
                                 + 1.00082 w_k1_FDX_s003
                                 + 1.25336 w_k1_IBM_s003
                                 + 1.05081 w_k1_INTC_s003
                                 + 0.987933 w_k1_KO_s003
                                 + 2.10556 w_k1_MAT_s003
                                 + 1.09748 w_k1_MSFT_s003
                                 + 1.00785 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.32574 w_k1_ADM_s003
                                 + 1.31489 w_k1_AXP_s003
                                 + 1.88629 w_k1_DIS_s003
                                 + 1.33791 w_k1_FDX_s003
                                 + 1.08197 w_k1_IBM_s003
                                 + 1.28887 w_k1_INTC_s003
                                 + 1.55499 w_k1_KO_s003 + 1.88547 w_k1_MAT_s003
                                 + 1.42717 w_k1_MSFT_s003 + 1.2489 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.872951 w_k1_ADM_s003
                                 + 0.961344 w_k1_AXP_s003
                                 + 0.704992 w_k1_DIS_s003
                                 + 0.863034 w_k1_FDX_s003
                                 + 0.95136 w_k1_IBM_s003
                                 + 0.886726 w_k1_INTC_s003
                                 + 0.914364 w_k1_KO_s003
                                 + 1.14712 w_k1_MAT_s003
                                 + 0.910564 w_k1_MSFT_s003
                                 + 0.871253 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.606048 w_k1_ADM_s003
                                 + 0.645347 w_k1_AXP_s003
                                 + 0.742595 w_k1_DIS_s003
                                 + 0.722046 w_k1_FDX_s003
                                 + 0.771745 w_k1_IBM_s003
                                 + 0.788356 w_k1_INTC_s003
                                 + 0.710773 w_k1_KO_s003
                                 + 0.842021 w_k1_MAT_s003
                                 + 0.778403 w_k1_MSFT_s003
                                 + 0.793738 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.92963 w_k1_ADM_s003
                                 + 0.960608 w_k1_AXP_s003
                                 + 1.07295 w_k1_DIS_s003
                                 + 1.05894 w_k1_FDX_s003
                                 + 0.967769 w_k1_IBM_s003
                                 + 0.98946 w_k1_INTC_s003
                                 + 0.924415 w_k1_KO_s003
                                 + 1.27973 w_k1_MAT_s003
                                 + 1.03423 w_k1_MSFT_s003
                                 + 0.949894 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.874677 w_k1_ADM_s003
                                 + 0.938034 w_k1_AXP_s003
                                 + 1.08213 w_k1_DIS_s003
                                 + 1.10552 w_k1_FDX_s003
                                 + 0.69189 w_k1_IBM_s003
                                 + 1.01609 w_k1_INTC_s003
                                 + 0.836489 w_k1_KO_s003
                                 + 0.927253 w_k1_MAT_s003
                                 + 1.13419 w_k1_MSFT_s003
                                 + 0.846347 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.746282 w_k1_ADM_s003
                                 + 0.815444 w_k1_AXP_s003
                                 + 0.833044 w_k1_DIS_s003
                                 + 0.786532 w_k1_FDX_s003
                                 + 1.21063 w_k1_IBM_s003
                                 + 0.818058 w_k1_INTC_s003
                                 + 0.917222 w_k1_KO_s003 + 1.3428 w_k1_MAT_s003
                                 + 0.885836 w_k1_MSFT_s003
                                 + 0.865685 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.15743 w_k1_ADM_s004
                                 + 1.3336 w_k1_AXP_s004 + 1.37001 w_k1_DIS_s004
                                 + 1.33288 w_k1_FDX_s004
                                 + 1.07225 w_k1_IBM_s004
                                 + 1.37205 w_k1_INTC_s004
                                 + 1.09696 w_k1_KO_s004 + 1.02202 w_k1_MAT_s004
                                 + 1.24848 w_k1_MSFT_s004
                                 + 1.18327 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.28623 w_k1_ADM_s004
                                 + 1.22334 w_k1_AXP_s004 + 1.7394 w_k1_DIS_s004
                                 + 1.33078 w_k1_FDX_s004
                                 + 1.31671 w_k1_IBM_s004
                                 + 1.38343 w_k1_INTC_s004
                                 + 1.39663 w_k1_KO_s004 + 1.33798 w_k1_MAT_s004
                                 + 1.32781 w_k1_MSFT_s004
                                 + 1.33672 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.18594 w_k1_ADM_s004
                                 + 0.915526 w_k1_AXP_s004
                                 + 1.27182 w_k1_DIS_s004 + 1.0842 w_k1_FDX_s004
                                 + 0.956059 w_k1_IBM_s004
                                 + 1.22286 w_k1_INTC_s004
                                 + 1.25672 w_k1_KO_s004
                                 + 0.864363 w_k1_MAT_s004
                                 + 1.14387 w_k1_MSFT_s004
                                 + 1.15178 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.878032 w_k1_ADM_s004
                                 + 0.746589 w_k1_AXP_s004
                                 + 0.626789 w_k1_DIS_s004
                                 + 0.863373 w_k1_FDX_s004
                                 + 0.610959 w_k1_IBM_s004
                                 + 1.01451 w_k1_INTC_s004
                                 + 0.610222 w_k1_KO_s004
                                 + 0.500184 w_k1_MAT_s004
                                 + 0.790726 w_k1_MSFT_s004
                                 + 0.90843 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.10833 w_k1_ADM_s004
                                 + 0.995498 w_k1_AXP_s004
                                 + 0.996443 w_k1_DIS_s004
                                 + 1.06687 w_k1_FDX_s004
                                 + 0.724071 w_k1_IBM_s004
                                 + 1.02992 w_k1_INTC_s004
                                 + 0.754366 w_k1_KO_s004
                                 + 0.580407 w_k1_MAT_s004
                                 + 0.971623 w_k1_MSFT_s004
                                 + 0.938077 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.20859 w_k1_ADM_s004
                                 + 1.15807 w_k1_AXP_s004
                                 + 0.82402 w_k1_DIS_s004 + 1.0418 w_k1_FDX_s004
                                 + 0.917431 w_k1_IBM_s004
                                 + 1.13432 w_k1_INTC_s004
                                 + 0.834264 w_k1_KO_s004
                                 + 0.873307 w_k1_MAT_s004
                                 + 0.974134 w_k1_MSFT_s004
                                 + 0.900866 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.10895 w_k1_ADM_s004
                                 + 1.03922 w_k1_AXP_s004
                                 + 1.05588 w_k1_DIS_s004
                                 + 1.01474 w_k1_FDX_s004
                                 + 1.39055 w_k1_IBM_s004
                                 + 0.953775 w_k1_INTC_s004
                                 + 1.2057 w_k1_KO_s004 + 1.35267 w_k1_MAT_s004
                                 + 0.923166 w_k1_MSFT_s004
                                 + 1.00746 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.764377 w_k1_ADM_s004
                                 + 0.855782 w_k1_AXP_s004
                                 + 0.687641 w_k1_DIS_s004
                                 + 0.769127 w_k1_FDX_s004
                                 + 0.955464 w_k1_IBM_s004
                                 + 0.943028 w_k1_INTC_s004
                                 + 0.818642 w_k1_KO_s004
                                 + 0.836309 w_k1_MAT_s004
                                 + 0.737775 w_k1_MSFT_s004
                                 + 0.880956 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.989798 w_k1_ADM_s005
                                 + 1.08021 w_k1_AXP_s005
                                 + 1.41343 w_k1_DIS_s005
                                 + 1.15426 w_k1_FDX_s005
                                 + 1.19607 w_k1_IBM_s005
                                 + 1.05023 w_k1_INTC_s005
                                 + 1.23227 w_k1_KO_s005
                                 + 0.991709 w_k1_MAT_s005
                                 + 0.934264 w_k1_MSFT_s005
                                 + 1.07628 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.996552 w_k1_ADM_s005
                                 + 0.836681 w_k1_AXP_s005
                                 + 0.83012 w_k1_DIS_s005
                                 + 0.838532 w_k1_FDX_s005
                                 + 1.07443 w_k1_IBM_s005
                                 + 0.833905 w_k1_INTC_s005
                                 + 0.987398 w_k1_KO_s005
                                 + 1.10173 w_k1_MAT_s005
                                 + 0.92082 w_k1_MSFT_s005
                                 + 0.977871 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.870541 w_k1_ADM_s005
                                 + 0.926091 w_k1_AXP_s005
                                 + 0.694045 w_k1_DIS_s005
                                 + 0.84714 w_k1_FDX_s005
                                 + 0.873394 w_k1_IBM_s005
                                 + 0.808857 w_k1_INTC_s005
                                 + 0.794503 w_k1_KO_s005
                                 + 0.807371 w_k1_MAT_s005
                                 + 0.799474 w_k1_MSFT_s005
                                 + 0.93142 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.687048 w_k1_ADM_s005
                                 + 0.748909 w_k1_AXP_s005
                                 + 0.603407 w_k1_DIS_s005
                                 + 0.805001 w_k1_FDX_s005
                                 + 0.633788 w_k1_IBM_s005
                                 + 0.846879 w_k1_INTC_s005
                                 + 0.611091 w_k1_KO_s005
                                 + 0.505099 w_k1_MAT_s005
                                 + 0.77571 w_k1_MSFT_s005
                                 + 0.90465 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.835272 w_k1_ADM_s005
                                 + 1.01106 w_k1_AXP_s005
                                 + 1.05239 w_k1_DIS_s005
                                 + 1.10764 w_k1_FDX_s005
                                 + 0.816017 w_k1_IBM_s005
                                 + 1.11082 w_k1_INTC_s005
                                 + 0.904321 w_k1_KO_s005
                                 + 0.717611 w_k1_MAT_s005
                                 + 0.980182 w_k1_MSFT_s005
                                 + 1.11731 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.16736 w_k1_ADM_s005
                                 + 1.38475 w_k1_AXP_s005
                                 + 1.96472 w_k1_DIS_s005
                                 + 1.44424 w_k1_FDX_s005
                                 + 1.28074 w_k1_IBM_s005
                                 + 1.37207 w_k1_INTC_s005
                                 + 1.36717 w_k1_KO_s005 + 1.40032 w_k1_MAT_s005
                                 + 1.42856 w_k1_MSFT_s005
                                 + 1.32662 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.646605 w_k1_ADM_s005
                                 + 0.728073 w_k1_AXP_s005
                                 + 0.496256 w_k1_DIS_s005
                                 + 0.619567 w_k1_FDX_s005
                                 + 0.787782 w_k1_IBM_s005
                                 + 0.792034 w_k1_INTC_s005
                                 + 0.723026 w_k1_KO_s005
                                 + 0.844391 w_k1_MAT_s005
                                 + 0.651881 w_k1_MSFT_s005
                                 + 0.789284 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.908959 w_k1_ADM_s005
                                 + 1.16006 w_k1_AXP_s005
                                 + 0.895127 w_k1_DIS_s005
                                 + 1.02103 w_k1_FDX_s005
                                 + 1.08407 w_k1_IBM_s005
                                 + 0.815931 w_k1_INTC_s005
                                 + 0.95252 w_k1_KO_s005 + 1.11162 w_k1_MAT_s005
                                 + 0.826432 w_k1_MSFT_s005
                                 + 1.15523 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.905213 w_k1_ADM_s006
                                 + 0.952522 w_k1_AXP_s006
                                 + 0.994687 w_k1_DIS_s006
                                 + 0.890192 w_k1_FDX_s006
                                 + 1.17208 w_k1_IBM_s006
                                 + 0.842868 w_k1_INTC_s006
                                 + 1.09251 w_k1_KO_s006 + 1.3479 w_k1_MAT_s006
                                 + 1.08757 w_k1_MSFT_s006
                                 + 0.961819 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.907651 w_k1_ADM_s006
                                 + 0.807677 w_k1_AXP_s006
                                 + 0.828624 w_k1_DIS_s006
                                 + 0.929448 w_k1_FDX_s006
                                 + 0.765702 w_k1_IBM_s006
                                 + 1.00775 w_k1_INTC_s006
                                 + 0.840576 w_k1_KO_s006
                                 + 0.701708 w_k1_MAT_s006
                                 + 0.853862 w_k1_MSFT_s006
                                 + 0.920011 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.08712 w_k1_ADM_s006
                                 + 1.49963 w_k1_AXP_s006
                                 + 1.29333 w_k1_DIS_s006
                                 + 1.17958 w_k1_FDX_s006
                                 + 1.43131 w_k1_IBM_s006
                                 + 1.16263 w_k1_INTC_s006
                                 + 1.26109 w_k1_KO_s006 + 1.52287 w_k1_MAT_s006
                                 + 1.30157 w_k1_MSFT_s006
                                 + 1.13938 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.38822 w_k1_ADM_s006
                                 + 1.30459 w_k1_AXP_s006
                                 + 1.19923 w_k1_DIS_s006
                                 + 1.25087 w_k1_FDX_s006
                                 + 0.932519 w_k1_IBM_s006
                                 + 1.13331 w_k1_INTC_s006
                                 + 1.05691 w_k1_KO_s006 + 1.02578 w_k1_MAT_s006
                                 + 1.1889 w_k1_MSFT_s006 + 1.01699 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.40666 w_k1_ADM_s006
                                 + 1.17925 w_k1_AXP_s006
                                 + 1.28564 w_k1_DIS_s006
                                 + 1.16274 w_k1_FDX_s006
                                 + 1.13785 w_k1_IBM_s006
                                 + 1.12362 w_k1_INTC_s006
                                 + 1.14725 w_k1_KO_s006
                                 + 0.996497 w_k1_MAT_s006
                                 + 1.25922 w_k1_MSFT_s006
                                 + 1.07352 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.85894 w_k1_ADM_s006
                                 + 0.770428 w_k1_AXP_s006
                                 + 0.608298 w_k1_DIS_s006
                                 + 0.70873 w_k1_FDX_s006
                                 + 0.96155 w_k1_IBM_s006
                                 + 0.862355 w_k1_INTC_s006
                                 + 0.778435 w_k1_KO_s006
                                 + 0.848069 w_k1_MAT_s006
                                 + 0.866583 w_k1_MSFT_s006
                                 + 0.836558 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.35395 w_k1_ADM_s006
                                 + 1.63579 w_k1_AXP_s006
                                 + 1.68272 w_k1_DIS_s006
                                 + 1.29626 w_k1_FDX_s006 + 1.7948 w_k1_IBM_s006
                                 + 1.2874 w_k1_INTC_s006 + 1.56994 w_k1_KO_s006
                                 + 2.13936 w_k1_MAT_s006
                                 + 1.30072 w_k1_MSFT_s006
                                 + 1.32462 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.23785 w_k1_ADM_s006
                                 + 1.04919 w_k1_AXP_s006
                                 + 0.972906 w_k1_DIS_s006
                                 + 1.03352 w_k1_FDX_s006
                                 + 0.82632 w_k1_IBM_s006
                                 + 0.985583 w_k1_INTC_s006
                                 + 1.0219 w_k1_KO_s006 + 1.11103 w_k1_MAT_s006
                                 + 1.08203 w_k1_MSFT_s006
                                 + 0.980791 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.07992 w_k1_ADM_s007
                                 + 1.08818 w_k1_AXP_s007
                                 + 1.07926 w_k1_DIS_s007
                                 + 1.05737 w_k1_FDX_s007
                                 + 1.39549 w_k1_IBM_s007
                                 + 1.03585 w_k1_INTC_s007
                                 + 1.12961 w_k1_KO_s007 + 1.13063 w_k1_MAT_s007
                                 + 0.95658 w_k1_MSFT_s007
                                 + 1.00657 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.24122 w_k1_ADM_s007
                                 + 1.43482 w_k1_AXP_s007
                                 + 1.52309 w_k1_DIS_s007
                                 + 1.31455 w_k1_FDX_s007 + 1.2582 w_k1_IBM_s007
                                 + 1.08885 w_k1_INTC_s007
                                 + 1.27348 w_k1_KO_s007 + 1.4449 w_k1_MAT_s007
                                 + 1.24578 w_k1_MSFT_s007
                                 + 1.16521 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.00278 w_k1_ADM_s007
                                 + 0.919609 w_k1_AXP_s007
                                 + 1.11074 w_k1_DIS_s007
                                 + 1.06866 w_k1_FDX_s007
                                 + 1.00474 w_k1_IBM_s007
                                 + 0.965808 w_k1_INTC_s007
                                 + 0.98559 w_k1_KO_s007
                                 + 0.946671 w_k1_MAT_s007
                                 + 1.08348 w_k1_MSFT_s007
                                 + 0.92569 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.70774 w_k1_ADM_s007
                                 + 0.668799 w_k1_AXP_s007
                                 + 0.546269 w_k1_DIS_s007
                                 + 0.680991 w_k1_FDX_s007
                                 + 0.628773 w_k1_IBM_s007
                                 + 0.657451 w_k1_INTC_s007
                                 + 0.598887 w_k1_KO_s007
                                 + 0.541698 w_k1_MAT_s007
                                 + 0.681145 w_k1_MSFT_s007
                                 + 0.682811 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.98167 w_k1_ADM_s007
                                 + 1.03183 w_k1_AXP_s007
                                 + 1.14031 w_k1_DIS_s007 + 1.2023 w_k1_FDX_s007
                                 + 0.818335 w_k1_IBM_s007
                                 + 1.07162 w_k1_INTC_s007
                                 + 1.04313 w_k1_KO_s007
                                 + 0.682344 w_k1_MAT_s007
                                 + 1.11345 w_k1_MSFT_s007
                                 + 0.934745 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.871578 w_k1_ADM_s007
                                 + 0.816879 w_k1_AXP_s007
                                 + 0.836489 w_k1_DIS_s007
                                 + 0.892443 w_k1_FDX_s007
                                 + 0.911577 w_k1_IBM_s007
                                 + 0.691644 w_k1_INTC_s007
                                 + 0.865111 w_k1_KO_s007
                                 + 1.14957 w_k1_MAT_s007
                                 + 0.907752 w_k1_MSFT_s007
                                 + 0.89567 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.801889 w_k1_ADM_s007
                                 + 0.64436 w_k1_AXP_s007
                                 + 0.663297 w_k1_DIS_s007
                                 + 0.775075 w_k1_FDX_s007
                                 + 0.996197 w_k1_IBM_s007
                                 + 0.711745 w_k1_INTC_s007
                                 + 0.818883 w_k1_KO_s007
                                 + 0.944956 w_k1_MAT_s007
                                 + 0.79252 w_k1_MSFT_s007
                                 + 0.815713 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.77502 w_k1_ADM_s007
                                 + 0.84745 w_k1_AXP_s007
                                 + 0.766394 w_k1_DIS_s007
                                 + 0.894432 w_k1_FDX_s007
                                 + 0.701254 w_k1_IBM_s007
                                 + 0.780085 w_k1_INTC_s007
                                 + 0.844494 w_k1_KO_s007
                                 + 0.82404 w_k1_MAT_s007
                                 + 0.864674 w_k1_MSFT_s007
                                 + 0.819823 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.02959 w_k1_ADM_s010
                                 + 1.14078 w_k1_AXP_s010
                                 + 1.35875 w_k1_DIS_s010
                                 + 1.15963 w_k1_FDX_s010
                                 + 1.15292 w_k1_IBM_s010
                                 + 1.40239 w_k1_INTC_s010
                                 + 1.24936 w_k1_KO_s010 + 1.06696 w_k1_MAT_s010
                                 + 1.33665 w_k1_MSFT_s010
                                 + 1.10968 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.48709 w_k1_ADM_s010
                                 + 1.21099 w_k1_AXP_s010
                                 + 1.86438 w_k1_DIS_s010
                                 + 1.41154 w_k1_FDX_s010
                                 + 0.988257 w_k1_IBM_s010
                                 + 1.3452 w_k1_INTC_s010 + 1.42918 w_k1_KO_s010
                                 + 1.46826 w_k1_MAT_s010
                                 + 1.30994 w_k1_MSFT_s010
                                 + 1.23126 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.06351 w_k1_ADM_s010
                                 + 0.897748 w_k1_AXP_s010
                                 + 0.991158 w_k1_DIS_s010
                                 + 0.931493 w_k1_FDX_s010
                                 + 1.35587 w_k1_IBM_s010
                                 + 1.2353 w_k1_INTC_s010 + 1.27013 w_k1_KO_s010
                                 + 1.21883 w_k1_MAT_s010
                                 + 1.05032 w_k1_MSFT_s010
                                 + 1.10661 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.08543 w_k1_ADM_s010
                                 + 0.725514 w_k1_AXP_s010
                                 + 0.690716 w_k1_DIS_s010
                                 + 0.815885 w_k1_FDX_s010
                                 + 0.707942 w_k1_IBM_s010
                                 + 0.908722 w_k1_INTC_s010
                                 + 0.82974 w_k1_KO_s010
                                 + 0.745618 w_k1_MAT_s010
                                 + 0.824519 w_k1_MSFT_s010
                                 + 0.965943 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.00297 w_k1_ADM_s010
                                 + 0.723327 w_k1_AXP_s010
                                 + 0.720386 w_k1_DIS_s010
                                 + 0.750404 w_k1_FDX_s010
                                 + 1.09731 w_k1_IBM_s010
                                 + 0.88164 w_k1_INTC_s010
                                 + 0.981022 w_k1_KO_s010
                                 + 0.758578 w_k1_MAT_s010
                                 + 0.79126 w_k1_MSFT_s010
                                 + 0.98863 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.44789 w_k1_ADM_s010
                                 + 1.14651 w_k1_AXP_s010
                                 + 1.37302 w_k1_DIS_s010
                                 + 1.19933 w_k1_FDX_s010
                                 + 1.58304 w_k1_IBM_s010
                                 + 1.03833 w_k1_INTC_s010
                                 + 1.19791 w_k1_KO_s010
                                 + 0.968782 w_k1_MAT_s010
                                 + 0.882288 w_k1_MSFT_s010
                                 + 1.08669 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.25856 w_k1_ADM_s010
                                 + 0.94562 w_k1_AXP_s010
                                 + 1.07074 w_k1_DIS_s010
                                 + 1.12229 w_k1_FDX_s010
                                 + 0.838899 w_k1_IBM_s010
                                 + 1.09957 w_k1_INTC_s010
                                 + 0.942038 w_k1_KO_s010
                                 + 0.871725 w_k1_MAT_s010
                                 + 1.02593 w_k1_MSFT_s010
                                 + 0.958375 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.13016 w_k1_ADM_s010
                                 + 0.974554 w_k1_AXP_s010
                                 + 1.00116 w_k1_DIS_s010
                                 + 0.963174 w_k1_FDX_s010
                                 + 0.909005 w_k1_IBM_s010
                                 + 1.01214 w_k1_INTC_s010
                                 + 0.937472 w_k1_KO_s010
                                 + 0.768841 w_k1_MAT_s010
                                 + 0.930649 w_k1_MSFT_s010
                                 + 0.933165 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.84914 w_k1_ADM_s011
                                 + 0.679196 w_k1_AXP_s011
                                 + 0.722533 w_k1_DIS_s011
                                 + 0.889355 w_k1_FDX_s011
                                 + 0.609308 w_k1_IBM_s011
                                 + 0.891454 w_k1_INTC_s011
                                 + 0.889829 w_k1_KO_s011
                                 + 0.871261 w_k1_MAT_s011
                                 + 0.989844 w_k1_MSFT_s011
                                 + 1.0062 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.0825 w_k1_ADM_s011
                                 + 1.12753 w_k1_AXP_s011
                                 + 0.883281 w_k1_DIS_s011
                                 + 1.03953 w_k1_FDX_s011
                                 + 1.00789 w_k1_IBM_s011
                                 + 0.937109 w_k1_INTC_s011
                                 + 0.810147 w_k1_KO_s011
                                 + 1.03484 w_k1_MAT_s011
                                 + 0.875198 w_k1_MSFT_s011
                                 + 1.03699 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.14817 w_k1_ADM_s011
                                 + 0.864524 w_k1_AXP_s011
                                 + 1.2169 w_k1_DIS_s011 + 1.18386 w_k1_FDX_s011
                                 + 0.924086 w_k1_IBM_s011
                                 + 0.980683 w_k1_INTC_s011
                                 + 1.09655 w_k1_KO_s011 + 1.04361 w_k1_MAT_s011
                                 + 1.15534 w_k1_MSFT_s011
                                 + 1.15586 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.827448 w_k1_ADM_s011
                                 + 0.808518 w_k1_AXP_s011
                                 + 1.00669 w_k1_DIS_s011
                                 + 0.937372 w_k1_FDX_s011
                                 + 0.938815 w_k1_IBM_s011
                                 + 1.07241 w_k1_INTC_s011
                                 + 0.996607 w_k1_KO_s011
                                 + 0.890793 w_k1_MAT_s011
                                 + 0.999241 w_k1_MSFT_s011
                                 + 1.02321 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.09965 w_k1_ADM_s011
                                 + 0.905594 w_k1_AXP_s011
                                 + 0.95949 w_k1_DIS_s011
                                 + 1.02808 w_k1_FDX_s011
                                 + 0.719991 w_k1_IBM_s011
                                 + 0.847846 w_k1_INTC_s011
                                 + 0.711819 w_k1_KO_s011
                                 + 0.695838 w_k1_MAT_s011
                                 + 1.00611 w_k1_MSFT_s011
                                 + 0.932081 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.32919 w_k1_ADM_s011
                                 + 1.23378 w_k1_AXP_s011
                                 + 1.46193 w_k1_DIS_s011
                                 + 1.33172 w_k1_FDX_s011 + 1.4481 w_k1_IBM_s011
                                 + 1.06247 w_k1_INTC_s011
                                 + 1.29177 w_k1_KO_s011 + 1.66505 w_k1_MAT_s011
                                 + 1.27782 w_k1_MSFT_s011
                                 + 1.16354 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.804584 w_k1_ADM_s011
                                 + 0.706079 w_k1_AXP_s011
                                 + 0.585174 w_k1_DIS_s011
                                 + 0.776119 w_k1_FDX_s011
                                 + 0.812018 w_k1_IBM_s011
                                 + 0.816486 w_k1_INTC_s011
                                 + 0.735055 w_k1_KO_s011
                                 + 0.778146 w_k1_MAT_s011
                                 + 0.748655 w_k1_MSFT_s011
                                 + 0.848143 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.905074 w_k1_ADM_s011
                                 + 0.881072 w_k1_AXP_s011
                                 + 0.942364 w_k1_DIS_s011
                                 + 0.83664 w_k1_FDX_s011
                                 + 1.39057 w_k1_IBM_s011
                                 + 0.893796 w_k1_INTC_s011
                                 + 1.16655 w_k1_KO_s011 + 1.30508 w_k1_MAT_s011
                                 + 0.969371 w_k1_MSFT_s011
                                 + 0.995323 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.899095 w_k1_ADM_s012
                                 + 0.784533 w_k1_AXP_s012
                                 + 1.07373 w_k1_DIS_s012
                                 + 0.854576 w_k1_FDX_s012
                                 + 1.03284 w_k1_IBM_s012
                                 + 0.963561 w_k1_INTC_s012
                                 + 1.04263 w_k1_KO_s012 + 1.04763 w_k1_MAT_s012
                                 + 0.923811 w_k1_MSFT_s012
                                 + 0.932766 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.22794 w_k1_ADM_s012
                                 + 0.959461 w_k1_AXP_s012
                                 + 1.50733 w_k1_DIS_s012
                                 + 1.19635 w_k1_FDX_s012
                                 + 1.23253 w_k1_IBM_s012
                                 + 1.31983 w_k1_INTC_s012
                                 + 1.44644 w_k1_KO_s012 + 1.54117 w_k1_MAT_s012
                                 + 1.37253 w_k1_MSFT_s012
                                 + 1.09344 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.48038 w_k1_ADM_s012
                                 + 1.39529 w_k1_AXP_s012
                                 + 1.79638 w_k1_DIS_s012 + 1.4304 w_k1_FDX_s012
                                 + 1.41454 w_k1_IBM_s012
                                 + 1.59201 w_k1_INTC_s012
                                 + 1.57142 w_k1_KO_s012 + 1.83623 w_k1_MAT_s012
                                 + 1.62876 w_k1_MSFT_s012
                                 + 1.11192 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.0248 w_k1_ADM_s012
                                 + 0.583077 w_k1_AXP_s012
                                 + 0.687558 w_k1_DIS_s012
                                 + 0.810824 w_k1_FDX_s012
                                 + 0.674631 w_k1_IBM_s012
                                 + 0.867077 w_k1_INTC_s012
                                 + 0.783664 w_k1_KO_s012
                                 + 0.588079 w_k1_MAT_s012
                                 + 0.905251 w_k1_MSFT_s012
                                 + 0.799082 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.00202 w_k1_ADM_s012
                                 + 0.933568 w_k1_AXP_s012
                                 + 1.29994 w_k1_DIS_s012
                                 + 0.908566 w_k1_FDX_s012
                                 + 1.63921 w_k1_IBM_s012
                                 + 1.19365 w_k1_INTC_s012
                                 + 1.42829 w_k1_KO_s012 + 1.77322 w_k1_MAT_s012
                                 + 1.11265 w_k1_MSFT_s012
                                 + 1.03178 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.40944 w_k1_ADM_s012
                                 + 1.30348 w_k1_AXP_s012
                                 + 1.43694 w_k1_DIS_s012
                                 + 1.23082 w_k1_FDX_s012
                                 + 1.02977 w_k1_IBM_s012
                                 + 1.12693 w_k1_INTC_s012
                                 + 1.26301 w_k1_KO_s012 + 1.14568 w_k1_MAT_s012
                                 + 1.2397 w_k1_MSFT_s012 + 1.13366 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.09053 w_k1_ADM_s012
                                 + 1.02411 w_k1_AXP_s012
                                 + 0.926736 w_k1_DIS_s012
                                 + 0.99139 w_k1_FDX_s012
                                 + 0.982346 w_k1_IBM_s012
                                 + 1.11721 w_k1_INTC_s012
                                 + 0.929194 w_k1_KO_s012
                                 + 0.903552 w_k1_MAT_s012
                                 + 0.910523 w_k1_MSFT_s012
                                 + 0.844542 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.12529 w_k1_ADM_s012
                                 + 0.987526 w_k1_AXP_s012
                                 + 0.956132 w_k1_DIS_s012
                                 + 1.04252 w_k1_FDX_s012
                                 + 0.590519 w_k1_IBM_s012
                                 + 1.1175 w_k1_INTC_s012
                                 + 0.798777 w_k1_KO_s012
                                 + 0.761518 w_k1_MAT_s012
                                 + 1.11375 w_k1_MSFT_s012
                                 + 0.937037 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.36447 w_k1_ADM_s013
                                 + 1.06051 w_k1_AXP_s013
                                 + 0.927587 w_k1_DIS_s013
                                 + 1.0565 w_k1_FDX_s013
                                 + 0.938361 w_k1_IBM_s013
                                 + 0.947591 w_k1_INTC_s013
                                 + 0.966854 w_k1_KO_s013
                                 + 1.13751 w_k1_MAT_s013
                                 + 0.99086 w_k1_MSFT_s013
                                 + 1.13601 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.01401 w_k1_ADM_s013
                                 + 0.874347 w_k1_AXP_s013
                                 + 0.767442 w_k1_DIS_s013
                                 + 0.836721 w_k1_FDX_s013
                                 + 1.13834 w_k1_IBM_s013
                                 + 0.824812 w_k1_INTC_s013
                                 + 0.863101 w_k1_KO_s013
                                 + 0.919746 w_k1_MAT_s013
                                 + 0.864825 w_k1_MSFT_s013
                                 + 1.06155 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.862532 w_k1_ADM_s013
                                 + 0.812098 w_k1_AXP_s013
                                 + 0.748679 w_k1_DIS_s013
                                 + 0.911815 w_k1_FDX_s013
                                 + 0.589734 w_k1_IBM_s013
                                 + 1.12617 w_k1_INTC_s013
                                 + 0.765165 w_k1_KO_s013
                                 + 0.419269 w_k1_MAT_s013
                                 + 0.823828 w_k1_MSFT_s013
                                 + 0.926678 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.35988 w_k1_ADM_s013
                                 + 1.34737 w_k1_AXP_s013
                                 + 1.59648 w_k1_DIS_s013
                                 + 1.43275 w_k1_FDX_s013
                                 + 1.25097 w_k1_IBM_s013
                                 + 1.53258 w_k1_INTC_s013
                                 + 1.17333 w_k1_KO_s013 + 1.12627 w_k1_MAT_s013
                                 + 1.42066 w_k1_MSFT_s013
                                 + 1.39036 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.20191 w_k1_ADM_s013
                                 + 1.01993 w_k1_AXP_s013 + 1.0774 w_k1_DIS_s013
                                 + 1.00964 w_k1_FDX_s013
                                 + 1.21307 w_k1_IBM_s013
                                 + 1.28897 w_k1_INTC_s013
                                 + 1.08732 w_k1_KO_s013
                                 + 0.797748 w_k1_MAT_s013
                                 + 1.13301 w_k1_MSFT_s013
                                 + 1.22319 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.08735 w_k1_ADM_s013
                                 + 1.43113 w_k1_AXP_s013
                                 + 1.57848 w_k1_DIS_s013 + 1.0671 w_k1_FDX_s013
                                 + 1.82411 w_k1_IBM_s013
                                 + 1.66268 w_k1_INTC_s013
                                 + 1.47169 w_k1_KO_s013 + 1.37306 w_k1_MAT_s013
                                 + 1.39887 w_k1_MSFT_s013
                                 + 1.31883 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.940977 w_k1_ADM_s013
                                 + 0.627465 w_k1_AXP_s013
                                 + 0.586523 w_k1_DIS_s013
                                 + 0.707677 w_k1_FDX_s013
                                 + 0.683496 w_k1_IBM_s013
                                 + 0.913991 w_k1_INTC_s013
                                 + 0.627065 w_k1_KO_s013
                                 + 0.479645 w_k1_MAT_s013
                                 + 0.809653 w_k1_MSFT_s013
                                 + 0.870974 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.09186 w_k1_ADM_s013
                                 + 1.07885 w_k1_AXP_s013
                                 + 0.837454 w_k1_DIS_s013
                                 + 0.849751 w_k1_FDX_s013
                                 + 1.00097 w_k1_IBM_s013
                                 + 0.84841 w_k1_INTC_s013
                                 + 0.878979 w_k1_KO_s013
                                 + 1.02921 w_k1_MAT_s013
                                 + 0.806402 w_k1_MSFT_s013
                                 + 1.07186 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.834475 w_k1_ADM_s014
                                 + 0.860071 w_k1_AXP_s014
                                 + 1.03707 w_k1_DIS_s014
                                 + 1.04036 w_k1_FDX_s014
                                 + 0.826822 w_k1_IBM_s014
                                 + 1.20802 w_k1_INTC_s014
                                 + 1.00193 w_k1_KO_s014
                                 + 0.618762 w_k1_MAT_s014
                                 + 1.05737 w_k1_MSFT_s014
                                 + 1.02064 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.34675 w_k1_ADM_s014
                                 + 1.47314 w_k1_AXP_s014 + 1.5388 w_k1_DIS_s014
                                 + 1.29772 w_k1_FDX_s014
                                 + 1.16427 w_k1_IBM_s014
                                 + 1.57749 w_k1_INTC_s014
                                 + 1.19173 w_k1_KO_s014 + 1.03475 w_k1_MAT_s014
                                 + 1.18724 w_k1_MSFT_s014
                                 + 1.01606 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.21103 w_k1_ADM_s014
                                 + 1.33865 w_k1_AXP_s014
                                 + 1.64104 w_k1_DIS_s014
                                 + 1.03288 w_k1_FDX_s014
                                 + 1.65028 w_k1_IBM_s014
                                 + 1.40184 w_k1_INTC_s014
                                 + 1.69396 w_k1_KO_s014 + 1.44024 w_k1_MAT_s014
                                 + 1.2191 w_k1_MSFT_s014 + 1.22101 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.909908 w_k1_ADM_s014
                                 + 0.870647 w_k1_AXP_s014
                                 + 0.695822 w_k1_DIS_s014
                                 + 0.888885 w_k1_FDX_s014
                                 + 0.615574 w_k1_IBM_s014
                                 + 1.20895 w_k1_INTC_s014
                                 + 0.812606 w_k1_KO_s014
                                 + 0.605099 w_k1_MAT_s014
                                 + 0.995192 w_k1_MSFT_s014
                                 + 0.867648 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.0041 w_k1_ADM_s014
                                 + 1.0105 w_k1_AXP_s014
                                 + 0.789319 w_k1_DIS_s014
                                 + 0.918331 w_k1_FDX_s014
                                 + 0.898494 w_k1_IBM_s014
                                 + 1.43654 w_k1_INTC_s014
                                 + 0.973979 w_k1_KO_s014
                                 + 0.912086 w_k1_MAT_s014
                                 + 1.04409 w_k1_MSFT_s014
                                 + 0.898163 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.814619 w_k1_ADM_s014
                                 + 0.857397 w_k1_AXP_s014
                                 + 0.86134 w_k1_DIS_s014
                                 + 0.793957 w_k1_FDX_s014
                                 + 1.23031 w_k1_IBM_s014
                                 + 0.954732 w_k1_INTC_s014
                                 + 1.04339 w_k1_KO_s014 + 1.12148 w_k1_MAT_s014
                                 + 0.861835 w_k1_MSFT_s014
                                 + 0.932455 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.701593 w_k1_ADM_s014
                                 + 0.590168 w_k1_AXP_s014
                                 + 0.67572 w_k1_DIS_s014
                                 + 0.730875 w_k1_FDX_s014
                                 + 0.759614 w_k1_IBM_s014
                                 + 1.20627 w_k1_INTC_s014
                                 + 0.887209 w_k1_KO_s014
                                 + 0.560987 w_k1_MAT_s014
                                 + 0.864154 w_k1_MSFT_s014
                                 + 0.831628 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.84258 w_k1_ADM_s014
                                 + 1.13158 w_k1_AXP_s014
                                 + 1.04738 w_k1_DIS_s014
                                 + 0.949684 w_k1_FDX_s014
                                 + 1.46552 w_k1_IBM_s014
                                 + 1.28505 w_k1_INTC_s014
                                 + 1.22387 w_k1_KO_s014 + 1.62298 w_k1_MAT_s014
                                 + 1.07132 w_k1_MSFT_s014 + 1.0049 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.10603 w_k1_ADM_s015
                                 + 1.56562 w_k1_AXP_s015
                                 + 2.02285 w_k1_DIS_s015 + 1.4888 w_k1_FDX_s015
                                 + 1.45128 w_k1_IBM_s015
                                 + 1.85864 w_k1_INTC_s015
                                 + 1.64114 w_k1_KO_s015 + 1.32725 w_k1_MAT_s015
                                 + 1.51629 w_k1_MSFT_s015
                                 + 1.39595 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.10583 w_k1_ADM_s015
                                 + 0.756818 w_k1_AXP_s015
                                 + 0.945813 w_k1_DIS_s015
                                 + 0.93466 w_k1_FDX_s015 + 1.0934 w_k1_IBM_s015
                                 + 1.05757 w_k1_INTC_s015
                                 + 1.03745 w_k1_KO_s015
                                 + 0.881947 w_k1_MAT_s015
                                 + 1.01736 w_k1_MSFT_s015
                                 + 1.02889 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.42136 w_k1_ADM_s015
                                 + 1.16446 w_k1_AXP_s015
                                 + 1.15922 w_k1_DIS_s015
                                 + 1.13216 w_k1_FDX_s015
                                 + 1.06907 w_k1_IBM_s015
                                 + 1.14395 w_k1_INTC_s015
                                 + 1.04536 w_k1_KO_s015
                                 + 0.958319 w_k1_MAT_s015
                                 + 1.0203 w_k1_MSFT_s015 + 1.09566 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.13308 w_k1_ADM_s015
                                 + 1.01474 w_k1_AXP_s015
                                 + 1.18088 w_k1_DIS_s015
                                 + 0.955181 w_k1_FDX_s015
                                 + 1.42844 w_k1_IBM_s015
                                 + 1.06041 w_k1_INTC_s015 + 1.3348 w_k1_KO_s015
                                 + 1.46308 w_k1_MAT_s015
                                 + 0.998043 w_k1_MSFT_s015
                                 + 0.953515 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.788805 w_k1_ADM_s015
                                 + 0.678743 w_k1_AXP_s015
                                 + 0.655121 w_k1_DIS_s015
                                 + 0.761705 w_k1_FDX_s015
                                 + 0.804881 w_k1_IBM_s015
                                 + 0.781968 w_k1_INTC_s015
                                 + 0.731408 w_k1_KO_s015
                                 + 0.812457 w_k1_MAT_s015
                                 + 0.727115 w_k1_MSFT_s015
                                 + 0.816366 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.15905 w_k1_ADM_s015
                                 + 1.18422 w_k1_AXP_s015
                                 + 1.36244 w_k1_DIS_s015
                                 + 1.17246 w_k1_FDX_s015
                                 + 1.33199 w_k1_IBM_s015
                                 + 1.2675 w_k1_INTC_s015 + 1.16838 w_k1_KO_s015
                                 + 1.15715 w_k1_MAT_s015
                                 + 1.16718 w_k1_MSFT_s015
                                 + 1.11775 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.09852 w_k1_ADM_s015
                                 + 1.11144 w_k1_AXP_s015
                                 + 0.896315 w_k1_DIS_s015
                                 + 0.980959 w_k1_FDX_s015
                                 + 0.85581 w_k1_IBM_s015
                                 + 1.11417 w_k1_INTC_s015
                                 + 0.928231 w_k1_KO_s015
                                 + 0.916442 w_k1_MAT_s015
                                 + 0.984068 w_k1_MSFT_s015
                                 + 1.0089 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.16664 w_k1_ADM_s015
                                 + 0.865361 w_k1_AXP_s015
                                 + 0.883912 w_k1_DIS_s015
                                 + 1.01564 w_k1_FDX_s015
                                 + 0.774285 w_k1_IBM_s015
                                 + 1.21897 w_k1_INTC_s015
                                 + 0.955655 w_k1_KO_s015
                                 + 0.889271 w_k1_MAT_s015
                                 + 0.962696 w_k1_MSFT_s015
                                 + 0.97723 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.11379 w_k1_ADM_s016
                                 + 0.975139 w_k1_AXP_s016
                                 + 1.18031 w_k1_DIS_s016
                                 + 0.984649 w_k1_FDX_s016
                                 + 1.40832 w_k1_IBM_s016
                                 + 1.12255 w_k1_INTC_s016
                                 + 1.35907 w_k1_KO_s016 + 1.62171 w_k1_MAT_s016
                                 + 1.0796 w_k1_MSFT_s016 + 1.1326 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.16397 w_k1_ADM_s016
                                 + 1.08418 w_k1_AXP_s016
                                 + 0.947969 w_k1_DIS_s016
                                 + 1.06156 w_k1_FDX_s016
                                 + 0.71016 w_k1_IBM_s016
                                 + 0.817011 w_k1_INTC_s016
                                 + 0.879771 w_k1_KO_s016
                                 + 0.896285 w_k1_MAT_s016
                                 + 0.967523 w_k1_MSFT_s016
                                 + 0.978634 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.99764 w_k1_ADM_s016
                                 + 0.760642 w_k1_AXP_s016
                                 + 0.66747 w_k1_DIS_s016
                                 + 0.797308 w_k1_FDX_s016
                                 + 0.640873 w_k1_IBM_s016
                                 + 0.843146 w_k1_INTC_s016
                                 + 0.770185 w_k1_KO_s016
                                 + 0.709187 w_k1_MAT_s016
                                 + 0.780531 w_k1_MSFT_s016
                                 + 0.790923 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.08064 w_k1_ADM_s016
                                 + 1.05309 w_k1_AXP_s016
                                 + 0.783685 w_k1_DIS_s016
                                 + 0.946556 w_k1_FDX_s016
                                 + 0.940783 w_k1_IBM_s016
                                 + 0.925693 w_k1_INTC_s016
                                 + 0.933833 w_k1_KO_s016
                                 + 1.06364 w_k1_MAT_s016
                                 + 0.848768 w_k1_MSFT_s016
                                 + 0.991459 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.814503 w_k1_ADM_s016
                                 + 0.682046 w_k1_AXP_s016
                                 + 0.82339 w_k1_DIS_s016
                                 + 0.828063 w_k1_FDX_s016
                                 + 0.973587 w_k1_IBM_s016
                                 + 0.888629 w_k1_INTC_s016
                                 + 0.880522 w_k1_KO_s016
                                 + 0.946913 w_k1_MAT_s016
                                 + 0.932892 w_k1_MSFT_s016
                                 + 0.922799 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.853229 w_k1_ADM_s016
                                 + 1.07969 w_k1_AXP_s016
                                 + 1.14322 w_k1_DIS_s016
                                 + 1.03662 w_k1_FDX_s016 + 1.0492 w_k1_IBM_s016
                                 + 1.09014 w_k1_INTC_s016
                                 + 0.968943 w_k1_KO_s016
                                 + 1.04098 w_k1_MAT_s016
                                 + 1.06214 w_k1_MSFT_s016
                                 + 1.04287 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.10562 w_k1_ADM_s016
                                 + 1.4726 w_k1_AXP_s016 + 1.48316 w_k1_DIS_s016
                                 + 1.27002 w_k1_FDX_s016
                                 + 1.30494 w_k1_IBM_s016
                                 + 1.14806 w_k1_INTC_s016
                                 + 1.21568 w_k1_KO_s016 + 1.95628 w_k1_MAT_s016
                                 + 1.19546 w_k1_MSFT_s016
                                 + 1.21909 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.34909 w_k1_ADM_s016
                                 + 1.35831 w_k1_AXP_s016
                                 + 1.56834 w_k1_DIS_s016
                                 + 1.41051 w_k1_FDX_s016
                                 + 0.887031 w_k1_IBM_s016
                                 + 1.12498 w_k1_INTC_s016
                                 + 1.21941 w_k1_KO_s016 + 1.17018 w_k1_MAT_s016
                                 + 1.24877 w_k1_MSFT_s016
                                 + 1.21907 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.55209 w_k1_ADM_s017
                                 + 1.30874 w_k1_AXP_s017
                                 + 1.52255 w_k1_DIS_s017
                                 + 1.44285 w_k1_FDX_s017
                                 + 0.954132 w_k1_IBM_s017
                                 + 1.20197 w_k1_INTC_s017
                                 + 1.08713 w_k1_KO_s017
                                 + 0.848868 w_k1_MAT_s017
                                 + 1.16206 w_k1_MSFT_s017
                                 + 1.02022 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.901936 w_k1_ADM_s017
                                 + 0.804843 w_k1_AXP_s017
                                 + 0.530151 w_k1_DIS_s017
                                 + 0.728068 w_k1_FDX_s017
                                 + 0.866319 w_k1_IBM_s017
                                 + 0.842325 w_k1_INTC_s017
                                 + 0.696318 w_k1_KO_s017
                                 + 0.573394 w_k1_MAT_s017
                                 + 0.657663 w_k1_MSFT_s017
                                 + 0.75563 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.924852 w_k1_ADM_s017
                                 + 0.621521 w_k1_AXP_s017
                                 + 0.710715 w_k1_DIS_s017
                                 + 0.781822 w_k1_FDX_s017
                                 + 0.645055 w_k1_IBM_s017
                                 + 0.756082 w_k1_INTC_s017
                                 + 0.727731 w_k1_KO_s017
                                 + 0.549688 w_k1_MAT_s017
                                 + 0.773799 w_k1_MSFT_s017
                                 + 0.784395 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.35769 w_k1_ADM_s017
                                 + 0.98746 w_k1_AXP_s017
                                 + 1.57352 w_k1_DIS_s017
                                 + 1.21051 w_k1_FDX_s017
                                 + 1.18041 w_k1_IBM_s017
                                 + 1.21016 w_k1_INTC_s017
                                 + 1.48676 w_k1_KO_s017 + 1.15533 w_k1_MAT_s017
                                 + 1.38352 w_k1_MSFT_s017
                                 + 1.28811 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.06055 w_k1_ADM_s017
                                 + 1.3162 w_k1_AXP_s017 + 1.00322 w_k1_DIS_s017
                                 + 1.08975 w_k1_FDX_s017
                                 + 1.25891 w_k1_IBM_s017
                                 + 1.26065 w_k1_INTC_s017
                                 + 1.21624 w_k1_KO_s017 + 1.41887 w_k1_MAT_s017
                                 + 1.07701 w_k1_MSFT_s017
                                 + 1.08344 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.21222 w_k1_ADM_s017
                                 + 0.888908 w_k1_AXP_s017
                                 + 1.28205 w_k1_DIS_s017
                                 + 1.18759 w_k1_FDX_s017
                                 + 0.873815 w_k1_IBM_s017
                                 + 1.1881 w_k1_INTC_s017 + 1.05261 w_k1_KO_s017
                                 + 0.677959 w_k1_MAT_s017
                                 + 1.04246 w_k1_MSFT_s017
                                 + 0.960679 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.19512 w_k1_ADM_s017
                                 + 0.89032 w_k1_AXP_s017
                                 + 0.861689 w_k1_DIS_s017
                                 + 1.04956 w_k1_FDX_s017
                                 + 0.631309 w_k1_IBM_s017
                                 + 0.954737 w_k1_INTC_s017
                                 + 0.731983 w_k1_KO_s017
                                 + 0.442876 w_k1_MAT_s017
                                 + 0.896268 w_k1_MSFT_s017
                                 + 1.01664 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.930072 w_k1_ADM_s017
                                 + 0.814786 w_k1_AXP_s017
                                 + 0.788974 w_k1_DIS_s017
                                 + 0.833897 w_k1_FDX_s017
                                 + 1.01539 w_k1_IBM_s017
                                 + 0.855068 w_k1_INTC_s017
                                 + 0.972257 w_k1_KO_s017
                                 + 1.08456 w_k1_MAT_s017
                                 + 0.82365 w_k1_MSFT_s017
                                 + 1.04921 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.0482 w_k1_ADM_s100
                                 + 1.1602 w_k1_AXP_s100
                                 + 0.756113 w_k1_DIS_s100
                                 + 0.986187 w_k1_FDX_s100
                                 + 0.897377 w_k1_IBM_s100
                                 + 1.00414 w_k1_INTC_s100
                                 + 0.890033 w_k1_KO_s100
                                 + 0.857862 w_k1_MAT_s100
                                 + 0.919265 w_k1_MSFT_s100
                                 + 0.995012 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.831915 w_k1_ADM_s100
                                 + 1.01354 w_k1_AXP_s100
                                 + 0.684588 w_k1_DIS_s100
                                 + 0.878564 w_k1_FDX_s100
                                 + 1.19535 w_k1_IBM_s100
                                 + 1.00255 w_k1_INTC_s100
                                 + 0.903011 w_k1_KO_s100
                                 + 0.993218 w_k1_MAT_s100
                                 + 0.756852 w_k1_MSFT_s100
                                 + 0.887708 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.1466 w_k1_ADM_s100
                                 + 1.21277 w_k1_AXP_s100
                                 + 1.01696 w_k1_DIS_s100
                                 + 1.03392 w_k1_FDX_s100
                                 + 1.24917 w_k1_IBM_s100
                                 + 1.14399 w_k1_INTC_s100
                                 + 1.02976 w_k1_KO_s100 + 1.21015 w_k1_MAT_s100
                                 + 0.983808 w_k1_MSFT_s100
                                 + 1.16408 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.874657 w_k1_ADM_s100
                                 + 0.701701 w_k1_AXP_s100
                                 + 0.665934 w_k1_DIS_s100
                                 + 0.773603 w_k1_FDX_s100
                                 + 0.88301 w_k1_IBM_s100
                                 + 0.905196 w_k1_INTC_s100
                                 + 0.826661 w_k1_KO_s100
                                 + 0.580014 w_k1_MAT_s100
                                 + 0.74782 w_k1_MSFT_s100
                                 + 0.856051 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.943064 w_k1_ADM_s100
                                 + 0.863762 w_k1_AXP_s100
                                 + 0.93868 w_k1_DIS_s100
                                 + 1.05768 w_k1_FDX_s100
                                 + 0.650568 w_k1_IBM_s100
                                 + 1.2575 w_k1_INTC_s100
                                 + 0.957473 w_k1_KO_s100
                                 + 0.457851 w_k1_MAT_s100
                                 + 1.11185 w_k1_MSFT_s100
                                 + 0.940623 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.53637 w_k1_ADM_s100
                                 + 1.71014 w_k1_AXP_s100
                                 + 1.49877 w_k1_DIS_s100
                                 + 1.38833 w_k1_FDX_s100
                                 + 1.19481 w_k1_IBM_s100
                                 + 1.35537 w_k1_INTC_s100
                                 + 1.30447 w_k1_KO_s100 + 1.44178 w_k1_MAT_s100
                                 + 1.22663 w_k1_MSFT_s100
                                 + 1.30201 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.1381 w_k1_ADM_s100
                                 + 0.874004 w_k1_AXP_s100
                                 + 1.08539 w_k1_DIS_s100
                                 + 1.05001 w_k1_FDX_s100
                                 + 0.864699 w_k1_IBM_s100
                                 + 0.938935 w_k1_INTC_s100
                                 + 0.922255 w_k1_KO_s100
                                 + 0.693722 w_k1_MAT_s100
                                 + 0.942138 w_k1_MSFT_s100
                                 + 1.09107 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.04351 w_k1_ADM_s100
                                 + 1.15115 w_k1_AXP_s100
                                 + 1.18934 w_k1_DIS_s100
                                 + 1.22029 w_k1_FDX_s100
                                 + 0.954391 w_k1_IBM_s100
                                 + 1.13907 w_k1_INTC_s100
                                 + 1.00429 w_k1_KO_s100
                                 + 0.945518 w_k1_MAT_s100
                                 + 1.05577 w_k1_MSFT_s100
                                 + 1.06644 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.07644 w_k1_ADM_s101
                                 + 0.93288 w_k1_AXP_s101
                                 + 1.06657 w_k1_DIS_s101
                                 + 1.14874 w_k1_FDX_s101
                                 + 0.713872 w_k1_IBM_s101
                                 + 1.04598 w_k1_INTC_s101
                                 + 0.904175 w_k1_KO_s101
                                 + 0.531227 w_k1_MAT_s101
                                 + 1.09128 w_k1_MSFT_s101
                                 + 0.981179 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.889697 w_k1_ADM_s101
                                 + 1.13937 w_k1_AXP_s101
                                 + 0.98434 w_k1_DIS_s101
                                 + 1.00726 w_k1_FDX_s101
                                 + 0.838622 w_k1_IBM_s101
                                 + 1.00055 w_k1_INTC_s101
                                 + 0.962032 w_k1_KO_s101
                                 + 0.812269 w_k1_MAT_s101
                                 + 0.929785 w_k1_MSFT_s101
                                 + 0.977326 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.870742 w_k1_ADM_s101
                                 + 0.690361 w_k1_AXP_s101
                                 + 0.666903 w_k1_DIS_s101
                                 + 0.759968 w_k1_FDX_s101
                                 + 0.903956 w_k1_IBM_s101
                                 + 0.782759 w_k1_INTC_s101
                                 + 0.879419 w_k1_KO_s101
                                 + 0.684492 w_k1_MAT_s101
                                 + 0.759667 w_k1_MSFT_s101
                                 + 0.860633 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.996115 w_k1_ADM_s101
                                 + 0.805183 w_k1_AXP_s101
                                 + 1.22877 w_k1_DIS_s101
                                 + 0.959932 w_k1_FDX_s101
                                 + 1.07301 w_k1_IBM_s101
                                 + 0.946116 w_k1_INTC_s101
                                 + 1.32632 w_k1_KO_s101 + 0.87662 w_k1_MAT_s101
                                 + 1.06549 w_k1_MSFT_s101
                                 + 1.10807 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.34631 w_k1_ADM_s101
                                 + 1.66243 w_k1_AXP_s101
                                 + 1.05343 w_k1_DIS_s101
                                 + 1.16982 w_k1_FDX_s101
                                 + 1.21379 w_k1_IBM_s101
                                 + 0.96335 w_k1_INTC_s101 + 1.0758 w_k1_KO_s101
                                 + 1.23398 w_k1_MAT_s101
                                 + 0.835029 w_k1_MSFT_s101
                                 + 1.16801 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.3114 w_k1_ADM_s101
                                 + 1.23065 w_k1_AXP_s101
                                 + 1.09078 w_k1_DIS_s101
                                 + 1.26833 w_k1_FDX_s101
                                 + 0.777235 w_k1_IBM_s101
                                 + 0.990876 w_k1_INTC_s101
                                 + 0.970065 w_k1_KO_s101
                                 + 0.61618 w_k1_MAT_s101
                                 + 1.02696 w_k1_MSFT_s101
                                 + 1.09233 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.874102 w_k1_ADM_s101
                                 + 0.936253 w_k1_AXP_s101
                                 + 1.05479 w_k1_DIS_s101
                                 + 0.914899 w_k1_FDX_s101
                                 + 1.61402 w_k1_IBM_s101
                                 + 0.959193 w_k1_INTC_s101
                                 + 1.39014 w_k1_KO_s101 + 1.2473 w_k1_MAT_s101
                                 + 0.941756 w_k1_MSFT_s101
                                 + 0.991731 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.21693 w_k1_ADM_s101
                                 + 1.21701 w_k1_AXP_s101 + 1.2633 w_k1_DIS_s101
                                 + 1.12665 w_k1_FDX_s101
                                 + 1.17605 w_k1_IBM_s101
                                 + 1.01433 w_k1_INTC_s101
                                 + 1.03757 w_k1_KO_s101 + 1.11077 w_k1_MAT_s101
                                 + 1.05825 w_k1_MSFT_s101
                                 + 1.01012 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.746087 w_k1_ADM_s102
                                 + 0.567014 w_k1_AXP_s102
                                 + 0.765493 w_k1_DIS_s102
                                 + 0.693933 w_k1_FDX_s102
                                 + 1.00663 w_k1_IBM_s102
                                 + 0.736889 w_k1_INTC_s102
                                 + 0.984119 w_k1_KO_s102
                                 + 0.947108 w_k1_MAT_s102
                                 + 0.812707 w_k1_MSFT_s102
                                 + 0.890724 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.38634 w_k1_ADM_s102
                                 + 1.20906 w_k1_AXP_s102
                                 + 1.34909 w_k1_DIS_s102
                                 + 1.25704 w_k1_FDX_s102
                                 + 0.825514 w_k1_IBM_s102
                                 + 1.1308 w_k1_INTC_s102 + 1.01406 w_k1_KO_s102
                                 + 0.678206 w_k1_MAT_s102
                                 + 1.07101 w_k1_MSFT_s102
                                 + 1.15936 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.902372 w_k1_ADM_s102
                                 + 0.797102 w_k1_AXP_s102
                                 + 0.696588 w_k1_DIS_s102
                                 + 0.928538 w_k1_FDX_s102
                                 + 0.637493 w_k1_IBM_s102
                                 + 0.906423 w_k1_INTC_s102
                                 + 0.816357 w_k1_KO_s102
                                 + 0.69792 w_k1_MAT_s102
                                 + 0.920082 w_k1_MSFT_s102
                                 + 0.910081 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.10522 w_k1_ADM_s102
                                 + 1.20768 w_k1_AXP_s102
                                 + 0.870077 w_k1_DIS_s102
                                 + 0.942189 w_k1_FDX_s102
                                 + 1.04842 w_k1_IBM_s102
                                 + 0.84001 w_k1_INTC_s102
                                 + 1.06024 w_k1_KO_s102 + 1.58896 w_k1_MAT_s102
                                 + 0.834056 w_k1_MSFT_s102
                                 + 1.13221 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.26652 w_k1_ADM_s102
                                 + 0.927781 w_k1_AXP_s102
                                 + 1.01924 w_k1_DIS_s102
                                 + 0.990208 w_k1_FDX_s102
                                 + 0.789487 w_k1_IBM_s102
                                 + 0.858293 w_k1_INTC_s102
                                 + 1.0733 w_k1_KO_s102 + 0.815385 w_k1_MAT_s102
                                 + 1.03122 w_k1_MSFT_s102
                                 + 1.01419 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.887916 w_k1_ADM_s102
                                 + 0.63251 w_k1_AXP_s102
                                 + 0.661989 w_k1_DIS_s102
                                 + 0.762026 w_k1_FDX_s102
                                 + 0.845878 w_k1_IBM_s102
                                 + 0.779629 w_k1_INTC_s102
                                 + 0.81954 w_k1_KO_s102 + 0.83787 w_k1_MAT_s102
                                 + 0.830107 w_k1_MSFT_s102
                                 + 0.903845 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.16921 w_k1_ADM_s102
                                 + 0.951529 w_k1_AXP_s102
                                 + 0.913195 w_k1_DIS_s102
                                 + 0.938298 w_k1_FDX_s102
                                 + 1.08405 w_k1_IBM_s102
                                 + 0.835195 w_k1_INTC_s102
                                 + 0.976021 w_k1_KO_s102 + 1.0899 w_k1_MAT_s102
                                 + 0.972859 w_k1_MSFT_s102
                                 + 1.04672 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.06096 w_k1_ADM_s102
                                 + 1.12131 w_k1_AXP_s102
                                 + 1.34237 w_k1_DIS_s102
                                 + 1.13933 w_k1_FDX_s102
                                 + 1.40874 w_k1_IBM_s102
                                 + 1.06924 w_k1_INTC_s102
                                 + 1.16755 w_k1_KO_s102 + 1.54133 w_k1_MAT_s102
                                 + 1.16341 w_k1_MSFT_s102
                                 + 1.06885 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.894977 w_k1_ADM_s103
                                 + 0.886103 w_k1_AXP_s103
                                 + 0.977375 w_k1_DIS_s103
                                 + 0.897542 w_k1_FDX_s103
                                 + 0.966308 w_k1_IBM_s103
                                 + 1.09082 w_k1_INTC_s103
                                 + 1.05326 w_k1_KO_s103 + 1.02164 w_k1_MAT_s103
                                 + 0.849234 w_k1_MSFT_s103
                                 + 1.01556 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.06164 w_k1_ADM_s103
                                 + 1.13398 w_k1_AXP_s103
                                 + 1.56244 w_k1_DIS_s103
                                 + 1.26933 w_k1_FDX_s103
                                 + 0.910534 w_k1_IBM_s103
                                 + 1.11239 w_k1_INTC_s103
                                 + 1.23813 w_k1_KO_s103 + 1.1211 w_k1_MAT_s103
                                 + 1.13888 w_k1_MSFT_s103
                                 + 1.12721 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.684183 w_k1_ADM_s103
                                 + 0.957223 w_k1_AXP_s103
                                 + 0.695114 w_k1_DIS_s103
                                 + 0.762676 w_k1_FDX_s103
                                 + 1.14908 w_k1_IBM_s103
                                 + 1.00507 w_k1_INTC_s103
                                 + 0.886988 w_k1_KO_s103
                                 + 1.10271 w_k1_MAT_s103
                                 + 0.737319 w_k1_MSFT_s103
                                 + 0.898397 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.4451 w_k1_ADM_s103
                                 + 1.22198 w_k1_AXP_s103
                                 + 1.43424 w_k1_DIS_s103 + 1.2128 w_k1_FDX_s103
                                 + 1.11978 w_k1_IBM_s103
                                 + 1.10317 w_k1_INTC_s103
                                 + 1.30034 w_k1_KO_s103 + 1.17452 w_k1_MAT_s103
                                 + 0.924295 w_k1_MSFT_s103
                                 + 1.19775 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.03456 w_k1_ADM_s103
                                 + 0.942172 w_k1_AXP_s103
                                 + 1.28664 w_k1_DIS_s103
                                 + 1.02279 w_k1_FDX_s103
                                 + 1.50513 w_k1_IBM_s103
                                 + 0.964242 w_k1_INTC_s103
                                 + 1.38087 w_k1_KO_s103 + 1.39862 w_k1_MAT_s103
                                 + 0.896613 w_k1_MSFT_s103
                                 + 1.21337 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.890788 w_k1_ADM_s103
                                 + 0.874805 w_k1_AXP_s103
                                 + 0.847512 w_k1_DIS_s103
                                 + 0.900366 w_k1_FDX_s103
                                 + 0.773526 w_k1_IBM_s103
                                 + 1.00718 w_k1_INTC_s103
                                 + 0.877358 w_k1_KO_s103
                                 + 0.829985 w_k1_MAT_s103
                                 + 0.806019 w_k1_MSFT_s103
                                 + 0.916566 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.39926 w_k1_ADM_s103
                                 + 1.72588 w_k1_AXP_s103
                                 + 1.48351 w_k1_DIS_s103
                                 + 1.25113 w_k1_FDX_s103
                                 + 1.47857 w_k1_IBM_s103
                                 + 1.14981 w_k1_INTC_s103
                                 + 1.44986 w_k1_KO_s103 + 2.06084 w_k1_MAT_s103
                                 + 1.02788 w_k1_MSFT_s103
                                 + 1.38782 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.07789 w_k1_ADM_s103
                                 + 0.756675 w_k1_AXP_s103
                                 + 0.966033 w_k1_DIS_s103
                                 + 0.855733 w_k1_FDX_s103
                                 + 1.27135 w_k1_IBM_s103
                                 + 0.945144 w_k1_INTC_s103
                                 + 1.16085 w_k1_KO_s103 + 1.08184 w_k1_MAT_s103
                                 + 0.798752 w_k1_MSFT_s103
                                 + 1.04203 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.0803 w_k1_ADM_s104
                                 + 1.05046 w_k1_AXP_s104
                                 + 1.10591 w_k1_DIS_s104 + 1.0692 w_k1_FDX_s104
                                 + 1.59687 w_k1_IBM_s104
                                 + 1.13589 w_k1_INTC_s104
                                 + 1.12628 w_k1_KO_s104 + 1.8438 w_k1_MAT_s104
                                 + 1.11774 w_k1_MSFT_s104
                                 + 1.23462 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.25835 w_k1_ADM_s104
                                 + 1.02461 w_k1_AXP_s104
                                 + 1.14618 w_k1_DIS_s104 + 1.1457 w_k1_FDX_s104
                                 + 1.1479 w_k1_IBM_s104
                                 + 1.18146 w_k1_INTC_s104
                                 + 1.14049 w_k1_KO_s104 + 1.09777 w_k1_MAT_s104
                                 + 1.13769 w_k1_MSFT_s104
                                 + 1.05795 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.21176 w_k1_ADM_s104
                                 + 1.14758 w_k1_AXP_s104
                                 + 0.868084 w_k1_DIS_s104
                                 + 1.10712 w_k1_FDX_s104
                                 + 0.789872 w_k1_IBM_s104
                                 + 1.02075 w_k1_INTC_s104
                                 + 0.727005 w_k1_KO_s104
                                 + 1.00754 w_k1_MAT_s104
                                 + 0.988222 w_k1_MSFT_s104
                                 + 1.05267 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.38446 w_k1_ADM_s104
                                 + 1.30527 w_k1_AXP_s104
                                 + 1.50056 w_k1_DIS_s104
                                 + 1.40039 w_k1_FDX_s104
                                 + 1.16203 w_k1_IBM_s104
                                 + 1.23981 w_k1_INTC_s104
                                 + 1.19315 w_k1_KO_s104 + 1.27301 w_k1_MAT_s104
                                 + 1.21277 w_k1_MSFT_s104 + 1.0963 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.89843 w_k1_ADM_s104
                                 + 0.848987 w_k1_AXP_s104
                                 + 0.730851 w_k1_DIS_s104
                                 + 0.866956 w_k1_FDX_s104
                                 + 0.70047 w_k1_IBM_s104
                                 + 0.922918 w_k1_INTC_s104
                                 + 0.642262 w_k1_KO_s104
                                 + 0.692114 w_k1_MAT_s104
                                 + 0.794865 w_k1_MSFT_s104
                                 + 0.850985 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.2035 w_k1_ADM_s104
                                 + 1.45263 w_k1_AXP_s104
                                 + 0.917138 w_k1_DIS_s104 + 1.033 w_k1_FDX_s104
                                 + 1.19922 w_k1_IBM_s104
                                 + 0.879953 w_k1_INTC_s104
                                 + 0.916521 w_k1_KO_s104
                                 + 1.55469 w_k1_MAT_s104
                                 + 0.839727 w_k1_MSFT_s104
                                 + 1.02354 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.728368 w_k1_ADM_s104
                                 + 0.621777 w_k1_AXP_s104
                                 + 0.599387 w_k1_DIS_s104
                                 + 0.73977 w_k1_FDX_s104
                                 + 1.03557 w_k1_IBM_s104
                                 + 0.80754 w_k1_INTC_s104
                                 + 0.737873 w_k1_KO_s104
                                 + 1.18426 w_k1_MAT_s104
                                 + 0.775038 w_k1_MSFT_s104
                                 + 0.88725 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.04913 w_k1_ADM_s104
                                 + 0.762488 w_k1_AXP_s104
                                 + 1.13588 w_k1_DIS_s104
                                 + 1.06908 w_k1_FDX_s104
                                 + 0.882664 w_k1_IBM_s104
                                 + 0.84758 w_k1_INTC_s104
                                 + 0.787681 w_k1_KO_s104
                                 + 0.934141 w_k1_MAT_s104
                                 + 0.962713 w_k1_MSFT_s104
                                 + 0.996766 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.951452 w_k1_ADM_s105
                                 + 1.46694 w_k1_AXP_s105
                                 + 1.57576 w_k1_DIS_s105
                                 + 1.28593 w_k1_FDX_s105
                                 + 1.57016 w_k1_IBM_s105
                                 + 1.24979 w_k1_INTC_s105
                                 + 1.57229 w_k1_KO_s105 + 1.64406 w_k1_MAT_s105
                                 + 1.16045 w_k1_MSFT_s105
                                 + 1.21363 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.681829 w_k1_ADM_s105
                                 + 0.760438 w_k1_AXP_s105
                                 + 0.486354 w_k1_DIS_s105
                                 + 0.702641 w_k1_FDX_s105
                                 + 0.695011 w_k1_IBM_s105
                                 + 0.835321 w_k1_INTC_s105
                                 + 0.649938 w_k1_KO_s105
                                 + 0.893027 w_k1_MAT_s105
                                 + 0.673789 w_k1_MSFT_s105
                                 + 0.739307 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.45206 w_k1_ADM_s105
                                 + 1.38793 w_k1_AXP_s105
                                 + 1.64101 w_k1_DIS_s105
                                 + 1.33786 w_k1_FDX_s105
                                 + 1.29411 w_k1_IBM_s105
                                 + 1.11848 w_k1_INTC_s105
                                 + 1.36658 w_k1_KO_s105 + 1.38108 w_k1_MAT_s105
                                 + 1.29842 w_k1_MSFT_s105
                                 + 1.11055 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.933859 w_k1_ADM_s105
                                 + 0.930561 w_k1_AXP_s105
                                 + 0.851027 w_k1_DIS_s105
                                 + 0.819888 w_k1_FDX_s105
                                 + 1.3259 w_k1_IBM_s105
                                 + 0.989959 w_k1_INTC_s105
                                 + 0.99728 w_k1_KO_s105
                                 + 0.798697 w_k1_MAT_s105
                                 + 1.00051 w_k1_MSFT_s105
                                 + 0.902784 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.12914 w_k1_ADM_s105
                                 + 1.17539 w_k1_AXP_s105
                                 + 1.06776 w_k1_DIS_s105 + 1.0848 w_k1_FDX_s105
                                 + 1.09147 w_k1_IBM_s105
                                 + 1.10057 w_k1_INTC_s105
                                 + 1.09494 w_k1_KO_s105 + 0.90495 w_k1_MAT_s105
                                 + 1.0963 w_k1_MSFT_s105 + 1.04255 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.12094 w_k1_ADM_s105
                                 + 0.843 w_k1_AXP_s105 + 0.885619 w_k1_DIS_s105
                                 + 0.932221 w_k1_FDX_s105
                                 + 0.987635 w_k1_IBM_s105
                                 + 1.03096 w_k1_INTC_s105
                                 + 1.02176 w_k1_KO_s105
                                 + 0.731306 w_k1_MAT_s105
                                 + 0.926895 w_k1_MSFT_s105
                                 + 0.928575 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.11676 w_k1_ADM_s105
                                 + 1.45389 w_k1_AXP_s105 + 1.2114 w_k1_DIS_s105
                                 + 1.25209 w_k1_FDX_s105
                                 + 0.911569 w_k1_IBM_s105
                                 + 1.42152 w_k1_INTC_s105
                                 + 1.01846 w_k1_KO_s105
                                 + 0.774629 w_k1_MAT_s105
                                 + 1.24953 w_k1_MSFT_s105
                                 + 1.00752 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.860235 w_k1_ADM_s105
                                 + 0.797114 w_k1_AXP_s105
                                 + 0.822703 w_k1_DIS_s105
                                 + 0.934972 w_k1_FDX_s105
                                 + 0.724027 w_k1_IBM_s105
                                 + 1.07088 w_k1_INTC_s105
                                 + 0.866525 w_k1_KO_s105
                                 + 0.426518 w_k1_MAT_s105
                                 + 0.973811 w_k1_MSFT_s105
                                 + 0.900893 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.09028 w_k1_ADM_s106
                                 + 0.775566 w_k1_AXP_s106
                                 + 0.96806 w_k1_DIS_s106
                                 + 0.959548 w_k1_FDX_s106
                                 + 0.711218 w_k1_IBM_s106
                                 + 0.860766 w_k1_INTC_s106
                                 + 0.862898 w_k1_KO_s106
                                 + 0.640415 w_k1_MAT_s106
                                 + 0.971226 w_k1_MSFT_s106
                                 + 0.919045 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.19169 w_k1_ADM_s106
                                 + 1.18386 w_k1_AXP_s106 + 1.0401 w_k1_DIS_s106
                                 + 0.991431 w_k1_FDX_s106
                                 + 1.44669 w_k1_IBM_s106
                                 + 0.93835 w_k1_INTC_s106 + 1.3139 w_k1_KO_s106
                                 + 1.59276 w_k1_MAT_s106
                                 + 1.02707 w_k1_MSFT_s106
                                 + 1.32516 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.907204 w_k1_ADM_s106
                                 + 0.803922 w_k1_AXP_s106
                                 + 1.08518 w_k1_DIS_s106
                                 + 0.924743 w_k1_FDX_s106
                                 + 1.13967 w_k1_IBM_s106
                                 + 1.02134 w_k1_INTC_s106
                                 + 1.10323 w_k1_KO_s106 + 1.13542 w_k1_MAT_s106
                                 + 1.13711 w_k1_MSFT_s106
                                 + 1.06445 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.23359 w_k1_ADM_s106
                                 + 0.992586 w_k1_AXP_s106
                                 + 1.49922 w_k1_DIS_s106
                                 + 1.17928 w_k1_FDX_s106
                                 + 1.62405 w_k1_IBM_s106
                                 + 1.24222 w_k1_INTC_s106
                                 + 1.50322 w_k1_KO_s106 + 1.44356 w_k1_MAT_s106
                                 + 1.3179 w_k1_MSFT_s106 + 1.22709 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.886108 w_k1_ADM_s106
                                 + 0.791314 w_k1_AXP_s106
                                 + 0.848009 w_k1_DIS_s106
                                 + 0.817804 w_k1_FDX_s106
                                 + 0.936027 w_k1_IBM_s106
                                 + 0.855711 w_k1_INTC_s106
                                 + 1.01337 w_k1_KO_s106
                                 + 0.948463 w_k1_MAT_s106
                                 + 0.948747 w_k1_MSFT_s106
                                 + 0.92551 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.688136 w_k1_ADM_s106
                                 + 0.638511 w_k1_AXP_s106
                                 + 0.690971 w_k1_DIS_s106
                                 + 0.728283 w_k1_FDX_s106
                                 + 0.753472 w_k1_IBM_s106
                                 + 0.752166 w_k1_INTC_s106
                                 + 0.743895 w_k1_KO_s106
                                 + 0.762267 w_k1_MAT_s106
                                 + 0.810025 w_k1_MSFT_s106
                                 + 0.845604 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.980162 w_k1_ADM_s106
                                 + 0.636612 w_k1_AXP_s106
                                 + 0.965431 w_k1_DIS_s106
                                 + 0.910966 w_k1_FDX_s106
                                 + 1.01746 w_k1_IBM_s106
                                 + 0.906654 w_k1_INTC_s106
                                 + 1.14562 w_k1_KO_s106
                                 + 0.826176 w_k1_MAT_s106
                                 + 1.1051 w_k1_MSFT_s106
                                 + 0.946398 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.06035 w_k1_ADM_s106
                                 + 1.20134 w_k1_AXP_s106
                                 + 1.24653 w_k1_DIS_s106
                                 + 1.28827 w_k1_FDX_s106
                                 + 0.953719 w_k1_IBM_s106
                                 + 1.07419 w_k1_INTC_s106
                                 + 1.03006 w_k1_KO_s106 + 1.1159 w_k1_MAT_s106
                                 + 1.32509 w_k1_MSFT_s106
                                 + 1.15663 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.655827 w_k1_ADM_s107
                                 + 0.81515 w_k1_AXP_s107
                                 + 0.820937 w_k1_DIS_s107
                                 + 0.856853 w_k1_FDX_s107
                                 + 1.13097 w_k1_IBM_s107
                                 + 1.0047 w_k1_INTC_s107
                                 + 0.888201 w_k1_KO_s107
                                 + 1.05328 w_k1_MAT_s107
                                 + 0.861773 w_k1_MSFT_s107
                                 + 0.88545 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.11254 w_k1_ADM_s107
                                 + 1.07961 w_k1_AXP_s107
                                 + 1.37216 w_k1_DIS_s107
                                 + 1.07211 w_k1_FDX_s107
                                 + 1.36927 w_k1_IBM_s107
                                 + 1.3773 w_k1_INTC_s107 + 1.41511 w_k1_KO_s107
                                 + 1.52641 w_k1_MAT_s107
                                 + 1.1626 w_k1_MSFT_s107 + 1.06675 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.03543 w_k1_ADM_s107
                                 + 1.3467 w_k1_AXP_s107 + 1.15032 w_k1_DIS_s107
                                 + 1.26995 w_k1_FDX_s107
                                 + 0.819112 w_k1_IBM_s107
                                 + 1.26322 w_k1_INTC_s107
                                 + 0.854031 w_k1_KO_s107
                                 + 0.966092 w_k1_MAT_s107
                                 + 1.2454 w_k1_MSFT_s107 + 1.03491 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.36573 w_k1_ADM_s107
                                 + 1.32537 w_k1_AXP_s107
                                 + 1.85936 w_k1_DIS_s107
                                 + 1.33741 w_k1_FDX_s107
                                 + 1.64127 w_k1_IBM_s107
                                 + 1.38885 w_k1_INTC_s107
                                 + 1.68746 w_k1_KO_s107 + 2.06749 w_k1_MAT_s107
                                 + 1.46999 w_k1_MSFT_s107
                                 + 1.31848 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.06009 w_k1_ADM_s107
                                 + 0.8025 w_k1_AXP_s107
                                 + 0.867884 w_k1_DIS_s107
                                 + 0.89901 w_k1_FDX_s107
                                 + 0.81127 w_k1_IBM_s107
                                 + 1.02079 w_k1_INTC_s107
                                 + 0.881816 w_k1_KO_s107
                                 + 0.771405 w_k1_MAT_s107
                                 + 0.999809 w_k1_MSFT_s107
                                 + 0.887775 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.07931 w_k1_ADM_s107
                                 + 1.05566 w_k1_AXP_s107
                                 + 1.21596 w_k1_DIS_s107
                                 + 1.15745 w_k1_FDX_s107
                                 + 1.02721 w_k1_IBM_s107
                                 + 1.32465 w_k1_INTC_s107
                                 + 1.03428 w_k1_KO_s107 + 1.01998 w_k1_MAT_s107
                                 + 1.23181 w_k1_MSFT_s107
                                 + 0.993536 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.00754 w_k1_ADM_s107
                                 + 1.5059 w_k1_AXP_s107 + 1.10157 w_k1_DIS_s107
                                 + 1.28599 w_k1_FDX_s107
                                 + 1.33999 w_k1_IBM_s107
                                 + 1.28563 w_k1_INTC_s107
                                 + 1.18047 w_k1_KO_s107 + 1.89867 w_k1_MAT_s107
                                 + 1.19969 w_k1_MSFT_s107
                                 + 1.09677 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.82006 w_k1_ADM_s107
                                 + 0.629932 w_k1_AXP_s107
                                 + 0.534951 w_k1_DIS_s107
                                 + 0.779859 w_k1_FDX_s107
                                 + 0.580174 w_k1_IBM_s107
                                 + 0.655702 w_k1_INTC_s107
                                 + 0.602727 w_k1_KO_s107
                                 + 0.533827 w_k1_MAT_s107
                                 + 0.732177 w_k1_MSFT_s107
                                 + 0.737002 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.914771 w_k1_ADM_s110
                                 + 1.08124 w_k1_AXP_s110
                                 + 1.27813 w_k1_DIS_s110
                                 + 1.28376 w_k1_FDX_s110
                                 + 0.861562 w_k1_IBM_s110
                                 + 1.0918 w_k1_INTC_s110
                                 + 0.975296 w_k1_KO_s110
                                 + 0.977098 w_k1_MAT_s110
                                 + 1.24681 w_k1_MSFT_s110
                                 + 1.15547 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.865171 w_k1_ADM_s110
                                 + 0.771809 w_k1_AXP_s110
                                 + 1.00177 w_k1_DIS_s110
                                 + 0.971911 w_k1_FDX_s110
                                 + 0.964059 w_k1_IBM_s110
                                 + 0.829477 w_k1_INTC_s110
                                 + 1.23754 w_k1_KO_s110 + 1.18258 w_k1_MAT_s110
                                 + 1.00706 w_k1_MSFT_s110
                                 + 0.971934 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.14046 w_k1_ADM_s110
                                 + 1.12471 w_k1_AXP_s110
                                 + 1.25345 w_k1_DIS_s110 + 1.1261 w_k1_FDX_s110
                                 + 1.21507 w_k1_IBM_s110
                                 + 1.05322 w_k1_INTC_s110
                                 + 1.36927 w_k1_KO_s110 + 1.37896 w_k1_MAT_s110
                                 + 1.30537 w_k1_MSFT_s110
                                 + 1.04818 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.06746 w_k1_ADM_s110
                                 + 0.95766 w_k1_AXP_s110
                                 + 0.910887 w_k1_DIS_s110
                                 + 0.969554 w_k1_FDX_s110
                                 + 0.680067 w_k1_IBM_s110
                                 + 0.893687 w_k1_INTC_s110
                                 + 0.975568 w_k1_KO_s110
                                 + 0.690707 w_k1_MAT_s110
                                 + 0.932096 w_k1_MSFT_s110
                                 + 0.896857 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.806251 w_k1_ADM_s110
                                 + 1.43746 w_k1_AXP_s110
                                 + 1.07757 w_k1_DIS_s110
                                 + 1.06035 w_k1_FDX_s110
                                 + 0.942372 w_k1_IBM_s110
                                 + 1.03836 w_k1_INTC_s110
                                 + 1.02637 w_k1_KO_s110 + 1.52311 w_k1_MAT_s110
                                 + 0.94424 w_k1_MSFT_s110
                                 + 1.00008 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.61106 w_k1_ADM_s110
                                 + 1.77613 w_k1_AXP_s110
                                 + 1.62417 w_k1_DIS_s110 + 1.3751 w_k1_FDX_s110
                                 + 1.35551 w_k1_IBM_s110
                                 + 1.2746 w_k1_INTC_s110 + 1.73032 w_k1_KO_s110
                                 + 2.54252 w_k1_MAT_s110
                                 + 1.25991 w_k1_MSFT_s110
                                 + 1.41656 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.667839 w_k1_ADM_s110
                                 + 0.806033 w_k1_AXP_s110
                                 + 0.701401 w_k1_DIS_s110
                                 + 0.767512 w_k1_FDX_s110
                                 + 0.961114 w_k1_IBM_s110
                                 + 0.79344 w_k1_INTC_s110
                                 + 1.00022 w_k1_KO_s110 + 1.14779 w_k1_MAT_s110
                                 + 0.860743 w_k1_MSFT_s110
                                 + 0.927163 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.05214 w_k1_ADM_s110
                                 + 1.15076 w_k1_AXP_s110
                                 + 1.37372 w_k1_DIS_s110
                                 + 1.09787 w_k1_FDX_s110 + 1.5977 w_k1_IBM_s110
                                 + 0.906452 w_k1_INTC_s110
                                 + 1.42493 w_k1_KO_s110 + 2.06247 w_k1_MAT_s110
                                 + 1.05836 w_k1_MSFT_s110
                                 + 1.01243 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.19199 w_k1_ADM_s111
                                 + 0.901782 w_k1_AXP_s111
                                 + 1.03061 w_k1_DIS_s111
                                 + 1.11005 w_k1_FDX_s111
                                 + 0.937633 w_k1_IBM_s111
                                 + 1.16276 w_k1_INTC_s111
                                 + 1.04648 w_k1_KO_s111
                                 + 0.684686 w_k1_MAT_s111
                                 + 1.01214 w_k1_MSFT_s111
                                 + 0.99348 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.980219 w_k1_ADM_s111
                                 + 0.790676 w_k1_AXP_s111
                                 + 0.753168 w_k1_DIS_s111
                                 + 0.86958 w_k1_FDX_s111
                                 + 0.645642 w_k1_IBM_s111
                                 + 0.846929 w_k1_INTC_s111
                                 + 0.758667 w_k1_KO_s111
                                 + 0.44051 w_k1_MAT_s111
                                 + 0.919343 w_k1_MSFT_s111
                                 + 0.768869 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.50145 w_k1_ADM_s111
                                 + 1.32383 w_k1_AXP_s111
                                 + 1.46016 w_k1_DIS_s111
                                 + 1.20481 w_k1_FDX_s111 + 1.0952 w_k1_IBM_s111
                                 + 1.27909 w_k1_INTC_s111
                                 + 1.28497 w_k1_KO_s111
                                 + 0.836792 w_k1_MAT_s111
                                 + 1.26834 w_k1_MSFT_s111
                                 + 1.12302 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.10919 w_k1_ADM_s111
                                 + 1.31758 w_k1_AXP_s111
                                 + 1.21889 w_k1_DIS_s111
                                 + 1.20296 w_k1_FDX_s111
                                 + 0.981337 w_k1_IBM_s111
                                 + 1.16535 w_k1_INTC_s111
                                 + 0.906576 w_k1_KO_s111
                                 + 0.608668 w_k1_MAT_s111
                                 + 1.20445 w_k1_MSFT_s111
                                 + 0.988223 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.16167 w_k1_ADM_s111
                                 + 0.9066 w_k1_AXP_s111
                                 + 0.923604 w_k1_DIS_s111
                                 + 1.09751 w_k1_FDX_s111
                                 + 0.708351 w_k1_IBM_s111
                                 + 1.06079 w_k1_INTC_s111
                                 + 0.852889 w_k1_KO_s111
                                 + 0.675446 w_k1_MAT_s111
                                 + 1.02916 w_k1_MSFT_s111
                                 + 0.967402 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.47794 w_k1_ADM_s111
                                 + 1.57727 w_k1_AXP_s111
                                 + 1.51612 w_k1_DIS_s111
                                 + 1.24716 w_k1_FDX_s111
                                 + 1.65908 w_k1_IBM_s111
                                 + 1.25338 w_k1_INTC_s111
                                 + 1.31702 w_k1_KO_s111 + 1.58136 w_k1_MAT_s111
                                 + 1.14923 w_k1_MSFT_s111 + 1.0596 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.984348 w_k1_ADM_s111
                                 + 0.764452 w_k1_AXP_s111
                                 + 0.935006 w_k1_DIS_s111
                                 + 0.902351 w_k1_FDX_s111
                                 + 1.05324 w_k1_IBM_s111
                                 + 1.21405 w_k1_INTC_s111
                                 + 1.03719 w_k1_KO_s111
                                 + 0.540126 w_k1_MAT_s111
                                 + 1.02563 w_k1_MSFT_s111
                                 + 0.972151 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.04495 w_k1_ADM_s111
                                 + 0.813858 w_k1_AXP_s111
                                 + 0.889126 w_k1_DIS_s111
                                 + 0.841258 w_k1_FDX_s111
                                 + 1.33587 w_k1_IBM_s111
                                 + 0.960517 w_k1_INTC_s111
                                 + 0.983031 w_k1_KO_s111
                                 + 1.00598 w_k1_MAT_s111
                                 + 0.862743 w_k1_MSFT_s111
                                 + 1.0023 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.22566 w_k1_ADM_s112
                                 + 1.36701 w_k1_AXP_s112
                                 + 1.78016 w_k1_DIS_s112
                                 + 1.32758 w_k1_FDX_s112
                                 + 1.15651 w_k1_IBM_s112
                                 + 1.25528 w_k1_INTC_s112
                                 + 1.43848 w_k1_KO_s112 + 1.48054 w_k1_MAT_s112
                                 + 1.31597 w_k1_MSFT_s112
                                 + 1.26488 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.06566 w_k1_ADM_s112
                                 + 1.16165 w_k1_AXP_s112
                                 + 0.774792 w_k1_DIS_s112
                                 + 0.98241 w_k1_FDX_s112
                                 + 0.818883 w_k1_IBM_s112
                                 + 1.04318 w_k1_INTC_s112
                                 + 0.798661 w_k1_KO_s112
                                 + 0.853137 w_k1_MAT_s112
                                 + 0.845303 w_k1_MSFT_s112
                                 + 0.985425 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.01702 w_k1_ADM_s112
                                 + 0.926333 w_k1_AXP_s112
                                 + 0.823135 w_k1_DIS_s112
                                 + 0.817236 w_k1_FDX_s112
                                 + 1.29224 w_k1_IBM_s112
                                 + 0.822487 w_k1_INTC_s112
                                 + 1.04723 w_k1_KO_s112 + 1.33064 w_k1_MAT_s112
                                 + 0.843532 w_k1_MSFT_s112
                                 + 1.00015 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.28525 w_k1_ADM_s112
                                 + 1.05538 w_k1_AXP_s112
                                 + 1.23621 w_k1_DIS_s112
                                 + 0.957629 w_k1_FDX_s112
                                 + 1.56551 w_k1_IBM_s112
                                 + 1.12856 w_k1_INTC_s112
                                 + 1.42005 w_k1_KO_s112 + 1.7757 w_k1_MAT_s112
                                 + 1.08713 w_k1_MSFT_s112
                                 + 1.14663 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.959544 w_k1_ADM_s112
                                 + 0.815778 w_k1_AXP_s112
                                 + 0.631087 w_k1_DIS_s112
                                 + 0.761251 w_k1_FDX_s112
                                 + 0.883122 w_k1_IBM_s112
                                 + 0.93792 w_k1_INTC_s112
                                 + 0.83399 w_k1_KO_s112
                                 + 0.881248 w_k1_MAT_s112
                                 + 0.831643 w_k1_MSFT_s112
                                 + 0.797133 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.4579 w_k1_ADM_s112
                                 + 1.29345 w_k1_AXP_s112 + 1.1595 w_k1_DIS_s112
                                 + 1.2824 w_k1_FDX_s112
                                 + 0.639592 w_k1_IBM_s112
                                 + 1.08641 w_k1_INTC_s112
                                 + 0.860095 w_k1_KO_s112
                                 + 0.709603 w_k1_MAT_s112
                                 + 1.09526 w_k1_MSFT_s112
                                 + 1.00003 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.05064 w_k1_ADM_s112
                                 + 1.25512 w_k1_AXP_s112
                                 + 1.13104 w_k1_DIS_s112
                                 + 1.04457 w_k1_FDX_s112
                                 + 1.18137 w_k1_IBM_s112
                                 + 1.07777 w_k1_INTC_s112
                                 + 1.03127 w_k1_KO_s112 + 1.25185 w_k1_MAT_s112
                                 + 0.972008 w_k1_MSFT_s112
                                 + 1.10997 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.11487 w_k1_ADM_s112
                                 + 0.826428 w_k1_AXP_s112
                                 + 1.01586 w_k1_DIS_s112
                                 + 1.04631 w_k1_FDX_s112
                                 + 0.871581 w_k1_IBM_s112
                                 + 1.04169 w_k1_INTC_s112
                                 + 0.902909 w_k1_KO_s112
                                 + 0.652855 w_k1_MAT_s112
                                 + 1.02122 w_k1_MSFT_s112
                                 + 0.92974 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.20359 w_k1_ADM_s113
                                 + 1.45583 w_k1_AXP_s113
                                 + 1.44287 w_k1_DIS_s113
                                 + 1.37614 w_k1_FDX_s113
                                 + 1.26472 w_k1_IBM_s113
                                 + 1.22952 w_k1_INTC_s113
                                 + 1.13777 w_k1_KO_s113 + 1.65289 w_k1_MAT_s113
                                 + 1.17059 w_k1_MSFT_s113
                                 + 1.17494 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.04319 w_k1_ADM_s113
                                 + 1.15861 w_k1_AXP_s113
                                 + 0.949735 w_k1_DIS_s113
                                 + 1.0094 w_k1_FDX_s113
                                 + 0.824641 w_k1_IBM_s113
                                 + 0.762767 w_k1_INTC_s113
                                 + 0.789206 w_k1_KO_s113
                                 + 0.944451 w_k1_MAT_s113
                                 + 0.824641 w_k1_MSFT_s113
                                 + 0.985682 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.790246 w_k1_ADM_s113
                                 + 0.819792 w_k1_AXP_s113
                                 + 0.724622 w_k1_DIS_s113
                                 + 0.772325 w_k1_FDX_s113
                                 + 1.25324 w_k1_IBM_s113
                                 + 0.76384 w_k1_INTC_s113
                                 + 0.905305 w_k1_KO_s113
                                 + 1.00381 w_k1_MAT_s113
                                 + 0.721272 w_k1_MSFT_s113
                                 + 0.901626 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.66767 w_k1_ADM_s113
                                 + 0.579575 w_k1_AXP_s113
                                 + 0.675919 w_k1_DIS_s113
                                 + 0.751841 w_k1_FDX_s113
                                 + 0.718366 w_k1_IBM_s113
                                 + 0.653357 w_k1_INTC_s113
                                 + 0.747891 w_k1_KO_s113
                                 + 0.81407 w_k1_MAT_s113
                                 + 0.678842 w_k1_MSFT_s113
                                 + 0.807145 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.988993 w_k1_ADM_s113
                                 + 1.09769 w_k1_AXP_s113
                                 + 1.27066 w_k1_DIS_s113
                                 + 1.02545 w_k1_FDX_s113
                                 + 1.60168 w_k1_IBM_s113
                                 + 0.861931 w_k1_INTC_s113
                                 + 1.28333 w_k1_KO_s113 + 1.8116 w_k1_MAT_s113
                                 + 1.01536 w_k1_MSFT_s113
                                 + 1.13999 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.942254 w_k1_ADM_s113
                                 + 0.854247 w_k1_AXP_s113
                                 + 0.881386 w_k1_DIS_s113
                                 + 0.961371 w_k1_FDX_s113
                                 + 0.869578 w_k1_IBM_s113
                                 + 0.835983 w_k1_INTC_s113
                                 + 0.866784 w_k1_KO_s113
                                 + 0.888497 w_k1_MAT_s113
                                 + 0.910786 w_k1_MSFT_s113
                                 + 0.994955 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.1185 w_k1_ADM_s113
                                 + 1.38582 w_k1_AXP_s113
                                 + 1.30704 w_k1_DIS_s113
                                 + 1.27024 w_k1_FDX_s113
                                 + 0.926485 w_k1_IBM_s113
                                 + 1.2634 w_k1_INTC_s113
                                 + 0.882825 w_k1_KO_s113
                                 + 1.08248 w_k1_MAT_s113
                                 + 1.1284 w_k1_MSFT_s113 + 1.03035 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.18716 w_k1_ADM_s113
                                 + 1.11297 w_k1_AXP_s113
                                 + 1.01686 w_k1_DIS_s113
                                 + 1.04555 w_k1_FDX_s113
                                 + 1.03239 w_k1_IBM_s113
                                 + 0.879905 w_k1_INTC_s113
                                 + 0.931071 w_k1_KO_s113
                                 + 1.21585 w_k1_MAT_s113
                                 + 0.915149 w_k1_MSFT_s113
                                 + 0.890263 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.989179 w_k1_ADM_s114
                                 + 0.727638 w_k1_AXP_s114
                                 + 0.917126 w_k1_DIS_s114
                                 + 0.928619 w_k1_FDX_s114
                                 + 0.866301 w_k1_IBM_s114
                                 + 0.777516 w_k1_INTC_s114
                                 + 0.801296 w_k1_KO_s114
                                 + 0.530028 w_k1_MAT_s114
                                 + 0.938005 w_k1_MSFT_s114
                                 + 0.774891 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.06417 w_k1_ADM_s114
                                 + 1.05294 w_k1_AXP_s114
                                 + 1.13997 w_k1_DIS_s114
                                 + 1.12565 w_k1_FDX_s114
                                 + 0.860355 w_k1_IBM_s114
                                 + 0.92472 w_k1_INTC_s114
                                 + 0.853248 w_k1_KO_s114
                                 + 1.03073 w_k1_MAT_s114
                                 + 1.04982 w_k1_MSFT_s114
                                 + 0.859105 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.1508 w_k1_ADM_s114
                                 + 1.40223 w_k1_AXP_s114
                                 + 1.36532 w_k1_DIS_s114
                                 + 1.12215 w_k1_FDX_s114
                                 + 1.27115 w_k1_IBM_s114
                                 + 1.14851 w_k1_INTC_s114
                                 + 1.13596 w_k1_KO_s114 + 1.25555 w_k1_MAT_s114
                                 + 1.13973 w_k1_MSFT_s114
                                 + 1.02804 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.94248 w_k1_ADM_s114
                                 + 0.750085 w_k1_AXP_s114
                                 + 0.653904 w_k1_DIS_s114
                                 + 0.862163 w_k1_FDX_s114
                                 + 0.768865 w_k1_IBM_s114
                                 + 0.711868 w_k1_INTC_s114
                                 + 0.64965 w_k1_KO_s114 + 0.76999 w_k1_MAT_s114
                                 + 0.757357 w_k1_MSFT_s114
                                 + 0.766644 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.946708 w_k1_ADM_s114
                                 + 1.21753 w_k1_AXP_s114
                                 + 1.15488 w_k1_DIS_s114 + 1.0053 w_k1_FDX_s114
                                 + 1.80476 w_k1_IBM_s114
                                 + 0.786894 w_k1_INTC_s114
                                 + 1.24697 w_k1_KO_s114 + 2.10165 w_k1_MAT_s114
                                 + 0.945513 w_k1_MSFT_s114
                                 + 0.992865 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.08457 w_k1_ADM_s114
                                 + 1.02801 w_k1_AXP_s114
                                 + 1.10867 w_k1_DIS_s114
                                 + 1.11725 w_k1_FDX_s114
                                 + 1.29631 w_k1_IBM_s114
                                 + 0.996224 w_k1_INTC_s114
                                 + 1.03053 w_k1_KO_s114 + 1.13969 w_k1_MAT_s114
                                 + 0.989455 w_k1_MSFT_s114
                                 + 1.00169 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.917424 w_k1_ADM_s114
                                 + 0.78735 w_k1_AXP_s114
                                 + 0.791903 w_k1_DIS_s114
                                 + 0.781688 w_k1_FDX_s114
                                 + 1.22246 w_k1_IBM_s114
                                 + 0.898887 w_k1_INTC_s114
                                 + 1.04382 w_k1_KO_s114 + 1.29376 w_k1_MAT_s114
                                 + 0.880153 w_k1_MSFT_s114
                                 + 0.785659 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.49958 w_k1_ADM_s114
                                 + 1.52043 w_k1_AXP_s114
                                 + 1.78497 w_k1_DIS_s114
                                 + 1.47915 w_k1_FDX_s114
                                 + 1.43453 w_k1_IBM_s114
                                 + 1.23197 w_k1_INTC_s114 + 1.4038 w_k1_KO_s114
                                 + 1.44447 w_k1_MAT_s114
                                 + 1.56348 w_k1_MSFT_s114
                                 + 1.24221 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.06163 w_k1_ADM_s115
                                 + 1.25406 w_k1_AXP_s115
                                 + 1.12434 w_k1_DIS_s115
                                 + 0.968788 w_k1_FDX_s115
                                 + 1.22935 w_k1_IBM_s115
                                 + 1.0534 w_k1_INTC_s115 + 1.21595 w_k1_KO_s115
                                 + 1.43598 w_k1_MAT_s115
                                 + 0.867746 w_k1_MSFT_s115
                                 + 1.0019 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.912047 w_k1_ADM_s115
                                 + 0.925124 w_k1_AXP_s115
                                 + 0.672976 w_k1_DIS_s115
                                 + 0.786415 w_k1_FDX_s115
                                 + 0.8434 w_k1_IBM_s115
                                 + 0.895488 w_k1_INTC_s115
                                 + 0.784909 w_k1_KO_s115
                                 + 0.808484 w_k1_MAT_s115
                                 + 0.804051 w_k1_MSFT_s115
                                 + 0.717302 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.968949 w_k1_ADM_s115
                                 + 1.5359 w_k1_AXP_s115 + 2.013 w_k1_DIS_s115
                                 + 1.40057 w_k1_FDX_s115
                                 + 1.08642 w_k1_IBM_s115
                                 + 1.29618 w_k1_INTC_s115
                                 + 1.16098 w_k1_KO_s115 + 1.23183 w_k1_MAT_s115
                                 + 1.17621 w_k1_MSFT_s115
                                 + 1.19803 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.960683 w_k1_ADM_s115
                                 + 1.00774 w_k1_AXP_s115
                                 + 0.993132 w_k1_DIS_s115
                                 + 1.03977 w_k1_FDX_s115
                                 + 0.664327 w_k1_IBM_s115
                                 + 0.961933 w_k1_INTC_s115
                                 + 0.77807 w_k1_KO_s115
                                 + 0.707883 w_k1_MAT_s115
                                 + 0.905143 w_k1_MSFT_s115
                                 + 0.859484 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.47304 w_k1_ADM_s115
                                 + 1.21478 w_k1_AXP_s115
                                 + 1.39882 w_k1_DIS_s115
                                 + 1.35898 w_k1_FDX_s115
                                 + 0.668797 w_k1_IBM_s115
                                 + 1.15532 w_k1_INTC_s115
                                 + 0.862805 w_k1_KO_s115
                                 + 0.81001 w_k1_MAT_s115
                                 + 1.21308 w_k1_MSFT_s115
                                 + 1.02076 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.97947 w_k1_ADM_s115
                                 + 1.4856 w_k1_AXP_s115 + 1.92016 w_k1_DIS_s115
                                 + 1.22909 w_k1_FDX_s115
                                 + 1.70158 w_k1_IBM_s115
                                 + 1.43704 w_k1_INTC_s115
                                 + 1.64487 w_k1_KO_s115 + 2.11378 w_k1_MAT_s115
                                 + 1.31791 w_k1_MSFT_s115
                                 + 1.14553 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.24472 w_k1_ADM_s115
                                 + 1.08212 w_k1_AXP_s115
                                 + 1.32755 w_k1_DIS_s115 + 1.22 w_k1_FDX_s115
                                 + 1.00429 w_k1_IBM_s115
                                 + 1.19731 w_k1_INTC_s115
                                 + 1.19598 w_k1_KO_s115 + 1.2474 w_k1_MAT_s115
                                 + 1.15384 w_k1_MSFT_s115
                                 + 1.04531 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.821844 w_k1_ADM_s115
                                 + 1.00796 w_k1_AXP_s115
                                 + 0.843299 w_k1_DIS_s115
                                 + 0.773338 w_k1_FDX_s115
                                 + 1.35951 w_k1_IBM_s115
                                 + 0.943593 w_k1_INTC_s115
                                 + 1.02204 w_k1_KO_s115 + 1.16429 w_k1_MAT_s115
                                 + 0.720756 w_k1_MSFT_s115
                                 + 0.883445 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.824695 w_k1_ADM_s116
                                 + 1.17729 w_k1_AXP_s116
                                 + 1.33614 w_k1_DIS_s116
                                 + 1.10577 w_k1_FDX_s116
                                 + 1.51116 w_k1_IBM_s116
                                 + 1.13924 w_k1_INTC_s116
                                 + 1.25771 w_k1_KO_s116 + 1.24282 w_k1_MAT_s116
                                 + 1.12425 w_k1_MSFT_s116
                                 + 1.05609 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.79563 w_k1_ADM_s116
                                 + 0.734598 w_k1_AXP_s116
                                 + 0.792566 w_k1_DIS_s116
                                 + 0.867565 w_k1_FDX_s116
                                 + 0.835194 w_k1_IBM_s116
                                 + 0.810726 w_k1_INTC_s116
                                 + 0.731115 w_k1_KO_s116
                                 + 0.431358 w_k1_MAT_s116
                                 + 0.853287 w_k1_MSFT_s116
                                 + 0.856984 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.04558 w_k1_ADM_s116
                                 + 1.23862 w_k1_AXP_s116
                                 + 1.22115 w_k1_DIS_s116
                                 + 1.16747 w_k1_FDX_s116
                                 + 1.09025 w_k1_IBM_s116
                                 + 1.18348 w_k1_INTC_s116
                                 + 1.14651 w_k1_KO_s116 + 1.06395 w_k1_MAT_s116
                                 + 0.989977 w_k1_MSFT_s116
                                 + 1.03116 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.914505 w_k1_ADM_s116
                                 + 0.837443 w_k1_AXP_s116
                                 + 0.996617 w_k1_DIS_s116
                                 + 1.05292 w_k1_FDX_s116
                                 + 0.742198 w_k1_IBM_s116
                                 + 1.03973 w_k1_INTC_s116
                                 + 0.915539 w_k1_KO_s116
                                 + 0.621748 w_k1_MAT_s116
                                 + 0.955258 w_k1_MSFT_s116
                                 + 0.834427 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.896104 w_k1_ADM_s116
                                 + 1.08481 w_k1_AXP_s116
                                 + 0.886248 w_k1_DIS_s116
                                 + 0.884624 w_k1_FDX_s116
                                 + 1.36303 w_k1_IBM_s116
                                 + 0.962907 w_k1_INTC_s116
                                 + 0.988929 w_k1_KO_s116
                                 + 1.17687 w_k1_MAT_s116
                                 + 0.782296 w_k1_MSFT_s116
                                 + 1.00351 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.524786 w_k1_ADM_s116
                                 + 0.646694 w_k1_AXP_s116
                                 + 0.651308 w_k1_DIS_s116
                                 + 0.691013 w_k1_FDX_s116
                                 + 1.1268 w_k1_IBM_s116
                                 + 0.777874 w_k1_INTC_s116
                                 + 0.759623 w_k1_KO_s116
                                 + 0.725452 w_k1_MAT_s116
                                 + 0.719123 w_k1_MSFT_s116
                                 + 0.864866 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.26434 w_k1_ADM_s116
                                 + 1.67567 w_k1_AXP_s116
                                 + 1.84308 w_k1_DIS_s116
                                 + 1.41027 w_k1_FDX_s116
                                 + 1.25682 w_k1_IBM_s116
                                 + 1.57376 w_k1_INTC_s116
                                 + 1.33666 w_k1_KO_s116 + 1.35842 w_k1_MAT_s116
                                 + 1.47438 w_k1_MSFT_s116
                                 + 1.16098 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.13076 w_k1_ADM_s116
                                 + 1.33333 w_k1_AXP_s116
                                 + 1.29632 w_k1_DIS_s116
                                 + 1.39517 w_k1_FDX_s116
                                 + 0.845611 w_k1_IBM_s116
                                 + 1.18216 w_k1_INTC_s116
                                 + 0.931163 w_k1_KO_s116
                                 + 0.681938 w_k1_MAT_s116
                                 + 1.26828 w_k1_MSFT_s116
                                 + 1.01178 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.784666 w_k1_ADM_s117
                                 + 0.844317 w_k1_AXP_s117
                                 + 1.03374 w_k1_DIS_s117
                                 + 0.935705 w_k1_FDX_s117
                                 + 1.16733 w_k1_IBM_s117
                                 + 1.17851 w_k1_INTC_s117
                                 + 1.03271 w_k1_KO_s117
                                 + 0.937771 w_k1_MAT_s117
                                 + 0.946842 w_k1_MSFT_s117
                                 + 0.941182 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.725222 w_k1_ADM_s117
                                 + 1.07664 w_k1_AXP_s117
                                 + 1.15622 w_k1_DIS_s117
                                 + 0.973305 w_k1_FDX_s117
                                 + 1.59996 w_k1_IBM_s117
                                 + 1.08949 w_k1_INTC_s117
                                 + 1.18937 w_k1_KO_s117 + 1.59796 w_k1_MAT_s117
                                 + 1.01392 w_k1_MSFT_s117
                                 + 1.00556 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.05305 w_k1_ADM_s117
                                 + 1.74245 w_k1_AXP_s117 + 1.6135 w_k1_DIS_s117
                                 + 1.52529 w_k1_FDX_s117
                                 + 1.16499 w_k1_IBM_s117
                                 + 1.6238 w_k1_INTC_s117 + 1.06912 w_k1_KO_s117
                                 + 0.904805 w_k1_MAT_s117
                                 + 1.11472 w_k1_MSFT_s117 + 1.09 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.774233 w_k1_ADM_s117
                                 + 0.996255 w_k1_AXP_s117
                                 + 0.982313 w_k1_DIS_s117
                                 + 0.952466 w_k1_FDX_s117
                                 + 0.88272 w_k1_IBM_s117
                                 + 1.23637 w_k1_INTC_s117
                                 + 0.986646 w_k1_KO_s117
                                 + 0.888584 w_k1_MAT_s117
                                 + 1.00384 w_k1_MSFT_s117 + 1.0316 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.59894 w_k1_ADM_s117
                                 + 0.608614 w_k1_AXP_s117
                                 + 0.551276 w_k1_DIS_s117
                                 + 0.579657 w_k1_FDX_s117
                                 + 0.908947 w_k1_IBM_s117
                                 + 0.864017 w_k1_INTC_s117
                                 + 0.666442 w_k1_KO_s117
                                 + 0.719947 w_k1_MAT_s117
                                 + 0.659282 w_k1_MSFT_s117
                                 + 0.724791 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.38479 w_k1_ADM_s117
                                 + 1.27859 w_k1_AXP_s117
                                 + 1.96736 w_k1_DIS_s117
                                 + 1.49773 w_k1_FDX_s117
                                 + 0.968161 w_k1_IBM_s117
                                 + 1.27091 w_k1_INTC_s117
                                 + 1.22345 w_k1_KO_s117
                                 + 0.886207 w_k1_MAT_s117
                                 + 1.40518 w_k1_MSFT_s117
                                 + 1.12132 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.836067 w_k1_ADM_s117
                                 + 0.821751 w_k1_AXP_s117
                                 + 0.770027 w_k1_DIS_s117
                                 + 0.962821 w_k1_FDX_s117
                                 + 0.665821 w_k1_IBM_s117
                                 + 1.06782 w_k1_INTC_s117
                                 + 0.754682 w_k1_KO_s117
                                 + 0.599067 w_k1_MAT_s117
                                 + 0.908611 w_k1_MSFT_s117
                                 + 0.896257 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.935707 w_k1_ADM_s117
                                 + 0.798933 w_k1_AXP_s117
                                 + 0.702043 w_k1_DIS_s117
                                 + 0.754949 w_k1_FDX_s117
                                 + 1.29291 w_k1_IBM_s117
                                 + 1.10964 w_k1_INTC_s117
                                 + 0.962164 w_k1_KO_s117
                                 + 0.87481 w_k1_MAT_s117
                                 + 0.794613 w_k1_MSFT_s117
                                 + 0.944199 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0283950599259027 w_k1_ADM_s000
                                 + 0.0378874671868805 w_k1_ADM_s001
                                 + 0.0459266029520881 w_k1_ADM_s002
                                 + 0.0105749991771398 w_k1_ADM_s003
                                 + 0.0450573198985505 w_k1_ADM_s004
                                 + 0.025362594020417 w_k1_ADM_s005
                                 + 0.041165710395808 w_k1_ADM_s006
                                 + 0.0383815542966333 w_k1_ADM_s007
                                 + 0.00857246290027622 w_k1_ADM_s010
                                 + 0.0174472750795393 w_k1_ADM_s011
                                 + 0.020276540789993 w_k1_ADM_s012
                                 + 0.00775347720212535 w_k1_ADM_s013
                                 + 0.0147187194998937 w_k1_ADM_s014
                                 + 0.0180048911875469 w_k1_ADM_s015
                                 + 0.0256215454462432 w_k1_ADM_s016
                                 + 0.0101218601104385 w_k1_ADM_s017
                                 + 0.0941086099885303 w_k1_ADM_s100
                                 + 0.0289180889352873 w_k1_ADM_s101
                                 + 0.0557258612547157 w_k1_ADM_s102
                                 + 0.0332210297713747 w_k1_ADM_s103
                                 + 0.022569241200093 w_k1_ADM_s104
                                 + 0.0445440288577921 w_k1_ADM_s105
                                 + 0.0251849554017807 w_k1_ADM_s106
                                 + 0.0331142960664868 w_k1_ADM_s107
                                 + 0.0232077449296568 w_k1_ADM_s110
                                 + 0.0184505714629111 w_k1_ADM_s111
                                 + 0.0422792193721465 w_k1_ADM_s112
                                 + 0.0641642171904081 w_k1_ADM_s113
                                 + 0.046982254379918 w_k1_ADM_s114
                                 + 0.047787286693829 w_k1_ADM_s115
                                 + 0.0264029808124906 w_k1_ADM_s116
                                 + 0.0198633961322689 w_k1_ADM_s117
                                 + 0.0276846494789003 w_k1_AXP_s000
                                 + 0.0409253456349041 w_k1_AXP_s001
                                 + 0.0557685760740743 w_k1_AXP_s002
                                 + 0.0111883886834592 w_k1_AXP_s003
                                 + 0.0435228345405596 w_k1_AXP_s004
                                 + 0.0278753142439732 w_k1_AXP_s005
                                 + 0.0410809874344628 w_k1_AXP_s006
                                 + 0.039500893841291 w_k1_AXP_s007
                                 + 0.00700869806412908 w_k1_AXP_s010
                                 + 0.0156651584643845 w_k1_AXP_s011
                                 + 0.0171341685588064 w_k1_AXP_s012
                                 + 0.00693351517789553 w_k1_AXP_s013
                                 + 0.015620187889092 w_k1_AXP_s014
                                 + 0.0163346887815642 w_k1_AXP_s015
                                 + 0.0249888802353579 w_k1_AXP_s016
                                 + 0.00886169532379175 w_k1_AXP_s017
                                 + 0.0950353306266185 w_k1_AXP_s100
                                 + 0.0285493356574322 w_k1_AXP_s101
                                 + 0.0488110580792012 w_k1_AXP_s102
                                 + 0.0325842502824467 w_k1_AXP_s103
                                 + 0.0205354216223558 w_k1_AXP_s104
                                 + 0.047219157143102 w_k1_AXP_s105
                                 + 0.0224121912969032 w_k1_AXP_s106
                                 + 0.034023631693144 w_k1_AXP_s107
                                 + 0.0251219664025491 w_k1_AXP_s110
                                 + 0.0161575982328883 w_k1_AXP_s111
                                 + 0.0393462279294655 w_k1_AXP_s112
                                 + 0.0689903680099008 w_k1_AXP_s113
                                 + 0.0441278561720069 w_k1_AXP_s114
                                 + 0.0521701129074442 w_k1_AXP_s115
                                 + 0.029646572406883 w_k1_AXP_s116
                                 + 0.0230695014389703 w_k1_AXP_s117
                                 + 0.0290709334547604 w_k1_DIS_s000
                                 + 0.0399800053708846 w_k1_DIS_s001
                                 + 0.0565200792315842 w_k1_DIS_s002
                                 + 0.0130596421179141 w_k1_DIS_s003
                                 + 0.0430489304254724 w_k1_DIS_s004
                                 + 0.0290381482637061 w_k1_DIS_s005
                                 + 0.0392321452560784 w_k1_DIS_s006
                                 + 0.0407637527134998 w_k1_DIS_s007
                                 + 0.00789252809976566 w_k1_DIS_s010
                                 + 0.0174031124824109 w_k1_DIS_s011
                                 + 0.0208194837229182 w_k1_DIS_s012
                                 + 0.00694784996106446 w_k1_DIS_s013
                                 + 0.0158090622371931 w_k1_DIS_s014
                                 + 0.0176068301164233 w_k1_DIS_s015
                                 + 0.0251196700319035 w_k1_DIS_s016
                                 + 0.00895709324764994 w_k1_DIS_s017
                                 + 0.0857446373314168 w_k1_DIS_s100
                                 + 0.0285530132026751 w_k1_DIS_s101
                                 + 0.0515694663187352 w_k1_DIS_s102
                                 + 0.0354287638797021 w_k1_DIS_s103
                                 + 0.0209557094830722 w_k1_DIS_s104
                                 + 0.0450215239227348 w_k1_DIS_s105
                                 + 0.0272584076307317 w_k1_DIS_s106
                                 + 0.0356511381067243 w_k1_DIS_s107
                                 + 0.0255986857738592 w_k1_DIS_s110
                                 + 0.0168698222652732 w_k1_DIS_s111
                                 + 0.0391150349712567 w_k1_DIS_s112
                                 + 0.0676458965587018 w_k1_DIS_s113
                                 + 0.0463247216590172 w_k1_DIS_s114
                                 + 0.0510082201384919 w_k1_DIS_s115
                                 + 0.0307885450706957 w_k1_DIS_s116
                                 + 0.0240428020340833 w_k1_DIS_s117
                                 + 0.0290724907518305 w_k1_FDX_s000
                                 + 0.0396353950945483 w_k1_FDX_s001
                                 + 0.0511728379126541 w_k1_FDX_s002
                                 + 0.0114693526088529 w_k1_FDX_s003
                                 + 0.0434769580103553 w_k1_FDX_s004
                                 + 0.0281031120117337 w_k1_FDX_s005
                                 + 0.0381449011479794 w_k1_FDX_s006
                                 + 0.0405813672555672 w_k1_FDX_s007
                                 + 0.00742698208181282 w_k1_FDX_s010
                                 + 0.0173222331360563 w_k1_FDX_s011
                                 + 0.0183915311042609 w_k1_FDX_s012
                                 + 0.00683473142491188 w_k1_FDX_s013
                                 + 0.0150212823767226 w_k1_FDX_s014
                                 + 0.0168960157617464 w_k1_FDX_s015
                                 + 0.0248119652666454 w_k1_FDX_s016
                                 + 0.00930315057144235 w_k1_FDX_s017
                                 + 0.0905855564707644 w_k1_FDX_s100
                                 + 0.0278723933393943 w_k1_FDX_s101
                                 + 0.0511639400635728 w_k1_FDX_s102
                                 + 0.032257006417519 w_k1_FDX_s103
                                 + 0.0218730389753661 w_k1_FDX_s104
                                 + 0.0452390232816181 w_k1_FDX_s105
                                 + 0.0252009183134433 w_k1_FDX_s106
                                 + 0.0350471014891723 w_k1_FDX_s107
                                 + 0.024869618202654 w_k1_FDX_s110
                                 + 0.016568551237889 w_k1_FDX_s111
                                 + 0.0374751180274648 w_k1_FDX_s112
                                 + 0.0672790314046004 w_k1_FDX_s113
                                 + 0.0453820324344487 w_k1_FDX_s114
                                 + 0.0471213964485019 w_k1_FDX_s115
                                 + 0.0300553792494682 w_k1_FDX_s116
                                 + 0.0231395668963001 w_k1_FDX_s117
                                 + 0.0301818106892854 w_k1_IBM_s000
                                 + 0.0396177045348779 w_k1_IBM_s001
                                 + 0.0524772317042926 w_k1_IBM_s002
                                 + 0.0117128288006761 w_k1_IBM_s003
                                 + 0.0433659923884724 w_k1_IBM_s004
                                 + 0.0277867106716182 w_k1_IBM_s005
                                 + 0.0410529449732285 w_k1_IBM_s006
                                 + 0.0404639674799036 w_k1_IBM_s007
                                 + 0.00795914000718342 w_k1_IBM_s010
                                 + 0.0182262985622882 w_k1_IBM_s011
                                 + 0.0190986602212044 w_k1_IBM_s012
                                 + 0.00721528414147199 w_k1_IBM_s013
                                 + 0.0154007286358727 w_k1_IBM_s014
                                 + 0.0178236764839192 w_k1_IBM_s015
                                 + 0.0239747348362309 w_k1_IBM_s016
                                 + 0.00852512450991744 w_k1_IBM_s017
                                 + 0.088753484997514 w_k1_IBM_s100
                                 + 0.0275288582867057 w_k1_IBM_s101
                                 + 0.0501129733105531 w_k1_IBM_s102
                                 + 0.0334865485502259 w_k1_IBM_s103
                                 + 0.0216164820397099 w_k1_IBM_s104
                                 + 0.043934968259523 w_k1_IBM_s105
                                 + 0.0268989385512837 w_k1_IBM_s106
                                 + 0.0361382663272834 w_k1_IBM_s107
                                 + 0.0247443494717451 w_k1_IBM_s110
                                 + 0.0160747529653868 w_k1_IBM_s111
                                 + 0.040478493693906 w_k1_IBM_s112
                                 + 0.067900666731255 w_k1_IBM_s113
                                 + 0.048768942193907 w_k1_IBM_s114
                                 + 0.0479568954777593 w_k1_IBM_s115
                                 + 0.0306202029178523 w_k1_IBM_s116
                                 + 0.0233629699609022 w_k1_IBM_s117
                                 + 0.0305072347511996 w_k1_INTC_s000
                                 + 0.0354779682591768 w_k1_INTC_s001
                                 + 0.0573142481266191 w_k1_INTC_s002
                                 + 0.0115518760856531 w_k1_INTC_s003
                                 + 0.0461315452507493 w_k1_INTC_s004
                                 + 0.0268516078458007 w_k1_INTC_s005
                                 + 0.0380943543532765 w_k1_INTC_s006
                                 + 0.0363857215291328 w_k1_INTC_s007
                                 + 0.00807829076599191 w_k1_INTC_s010
                                 + 0.0160966575770995 w_k1_INTC_s011
                                 + 0.0203389013606849 w_k1_INTC_s012
                                 + 0.00785723387639871 w_k1_INTC_s013
                                 + 0.0196967837159751 w_k1_INTC_s014
                                 + 0.018659610620442 w_k1_INTC_s015
                                 + 0.0241165147482552 w_k1_INTC_s016
                                 + 0.00928693005148909 w_k1_INTC_s017
                                 + 0.0949231514684081 w_k1_INTC_s100
                                 + 0.0258806788269537 w_k1_INTC_s101
                                 + 0.0477103770170326 w_k1_INTC_s102
                                 + 0.0339288764889525 w_k1_INTC_s103
                                 + 0.0211882500194271 w_k1_INTC_s104
                                 + 0.0481509458821927 w_k1_INTC_s105
                                 + 0.0248102573487925 w_k1_INTC_s106
                                 + 0.038893997838049 w_k1_INTC_s107
                                 + 0.0230213178361652 w_k1_INTC_s110
                                 + 0.0176316107104561 w_k1_INTC_s111
                                 + 0.039022689098311 w_k1_INTC_s112
                                 + 0.0595403908487778 w_k1_INTC_s113
                                 + 0.0410387459248304 w_k1_INTC_s114
                                 + 0.0489395043893952 w_k1_INTC_s115
                                 + 0.0302369300488095 w_k1_INTC_s116
                                 + 0.0270680907133358 w_k1_INTC_s117
                                 + 0.0289368674129387 w_k1_KO_s000
                                 + 0.040544585340387 w_k1_KO_s001
                                 + 0.04910818574279 w_k1_KO_s002
                                 + 0.0118633467858481 w_k1_KO_s003
                                 + 0.0418082851016127 w_k1_KO_s004
                                 + 0.0273014872123827 w_k1_KO_s005
                                 + 0.0397328313262354 w_k1_KO_s006
                                 + 0.0392865233576033 w_k1_KO_s007
                                 + 0.00799134405698678 w_k1_KO_s010
                                 + 0.0174066318609244 w_k1_KO_s011
                                 + 0.0202368361525636 w_k1_KO_s012
                                 + 0.00665309345595563 w_k1_KO_s013
                                 + 0.0161071178197272 w_k1_KO_s014
                                 + 0.0173124491542509 w_k1_KO_s015
                                 + 0.024708625051591 w_k1_KO_s016
                                 + 0.00893673654879419 w_k1_KO_s017
                                 + 0.0859109280220025 w_k1_KO_s100
                                 + 0.0281800737248178 w_k1_KO_s101
                                 + 0.0506449556529741 w_k1_KO_s102
                                 + 0.0356129120344942 w_k1_KO_s103
                                 + 0.018795027148651 w_k1_KO_s104
                                 + 0.0452547452720072 w_k1_KO_s105
                                 + 0.0272748920725202 w_k1_KO_s106
                                 + 0.0347111910772266 w_k1_KO_s107
                                 + 0.028435048579789 w_k1_KO_s110
                                 + 0.0159816273930895 w_k1_KO_s111
                                 + 0.0392921316747788 w_k1_KO_s112
                                 + 0.0608951375959127 w_k1_KO_s113
                                 + 0.0428997126425322 w_k1_KO_s114
                                 + 0.0474732797670957 w_k1_KO_s115
                                 + 0.028218636054121 w_k1_KO_s116
                                 + 0.0220800480869568 w_k1_KO_s117
                                 + 0.0341037427488756 w_k1_MAT_s000
                                 + 0.0466290236759271 w_k1_MAT_s001
                                 + 0.0480293857573883 w_k1_MAT_s002
                                 + 0.0160605496089787 w_k1_MAT_s003
                                 + 0.0402393313882527 w_k1_MAT_s004
                                 + 0.0264151394892622 w_k1_MAT_s005
                                 + 0.0435725753081085 w_k1_MAT_s006
                                 + 0.0392428409286867 w_k1_MAT_s007
                                 + 0.00693236500964305 w_k1_MAT_s010
                                 + 0.0189517430273013 w_k1_MAT_s011
                                 + 0.0206485945538969 w_k1_MAT_s012
                                 + 0.00600043237572926 w_k1_MAT_s013
                                 + 0.014055259505947 w_k1_MAT_s014
                                 + 0.0169836521564638 w_k1_MAT_s015
                                 + 0.0278372808935218 w_k1_MAT_s016
                                 + 0.00811789660159568 w_k1_MAT_s017
                                 + 0.0810712250622768 w_k1_MAT_s100
                                 + 0.0239431981723092 w_k1_MAT_s101
                                 + 0.0519609212989752 w_k1_MAT_s102
                                 + 0.0356732124117638 w_k1_MAT_s103
                                 + 0.0235871761130307 w_k1_MAT_s104
                                 + 0.0376660454966972 w_k1_MAT_s105
                                 + 0.0267365951384537 w_k1_MAT_s106
                                 + 0.0386963533080402 w_k1_MAT_s107
                                 + 0.0309900711717467 w_k1_MAT_s110
                                 + 0.0116276886007001 w_k1_MAT_s111
                                 + 0.0423712973585319 w_k1_MAT_s112
                                 + 0.0758944472642496 w_k1_MAT_s113
                                 + 0.0484749844954028 w_k1_MAT_s114
                                 + 0.0513666257464548 w_k1_MAT_s115
                                 + 0.0243882735061115 w_k1_MAT_s116
                                 + 0.020550600663336 w_k1_MAT_s117
                                 + 0.0315312172968512 w_k1_MSFT_s000
                                 + 0.0371642126640713 w_k1_MSFT_s001
                                 + 0.0469872686021747 w_k1_MSFT_s002
                                 + 0.012338461309703 w_k1_MSFT_s003
                                 + 0.0412258175288604 w_k1_MSFT_s004
                                 + 0.025863836712991 w_k1_MSFT_s005
                                 + 0.0412427134708488 w_k1_MSFT_s006
                                 + 0.0387379538215214 w_k1_MSFT_s007
                                 + 0.00736921121710927 w_k1_MSFT_s010
                                 + 0.0174070745692083 w_k1_MSFT_s011
                                 + 0.0199655155579433 w_k1_MSFT_s012
                                 + 0.00711101205767684 w_k1_MSFT_s013
                                 + 0.0158123437045706 w_k1_MSFT_s014
                                 + 0.0166987699211236 w_k1_MSFT_s015
                                 + 0.0243627752656518 w_k1_MSFT_s016
                                 + 0.0086903240016704 w_k1_MSFT_s017
                                 + 0.0834685741183042 w_k1_MSFT_s100
                                 + 0.0261114620272818 w_k1_MSFT_s101
                                 + 0.0507988133263831 w_k1_MSFT_s102
                                 + 0.0284468120992761 w_k1_MSFT_s103
                                 + 0.0204296664305899 w_k1_MSFT_s104
                                 + 0.0452656626193104 w_k1_MSFT_s105
                                 + 0.0280638023685405 w_k1_MSFT_s106
                                 + 0.0364266318783985 w_k1_MSFT_s107
                                 + 0.0256408268252995 w_k1_MSFT_s110
                                 + 0.0168631783417138 w_k1_MSFT_s111
                                 + 0.0372886514505643 w_k1_MSFT_s112
                                 + 0.0602553209860261 w_k1_MSFT_s113
                                 + 0.0445603693454642 w_k1_MSFT_s114
                                 + 0.0437482541871563 w_k1_MSFT_s115
                                 + 0.0284479156453377 w_k1_MSFT_s116
                                 + 0.0224251753661775 w_k1_MSFT_s117
                                 + 0.0267172363759064 w_k1_PG_s000
                                 + 0.0378288155697352 w_k1_PG_s001
                                 + 0.0526170714735023 w_k1_PG_s002
                                 + 0.0111741112190154 w_k1_PG_s003
                                 + 0.0425864191780179 w_k1_PG_s004
                                 + 0.0293289958165804 w_k1_PG_s005
                                 + 0.0377018945427644 w_k1_PG_s006
                                 + 0.0367866330987471 w_k1_PG_s007
                                 + 0.00767543392037042 w_k1_PG_s010
                                 + 0.017676722276988 w_k1_PG_s011
                                 + 0.0175094129415817 w_k1_PG_s012
                                 + 0.00771656708985994 w_k1_PG_s013
                                 + 0.0146021355740893 w_k1_PG_s014
                                 + 0.0169240394091681 w_k1_PG_s015
                                 + 0.0250018985308087 w_k1_PG_s016
                                 + 0.00918753263545606 w_k1_PG_s017
                                 + 0.0911891345947828 w_k1_PG_s100
                                 + 0.0272841814866605 w_k1_PG_s101
                                 + 0.0523320230110468 w_k1_PG_s102
                                 + 0.0341271604546587 w_k1_PG_s103
                                 + 0.0211396479556279 w_k1_PG_s104
                                 + 0.0422873971438235 w_k1_PG_s105
                                 + 0.0266257236404984 w_k1_PG_s106
                                 + 0.0329829057252063 w_k1_PG_s107
                                 + 0.0247962427278962 w_k1_PG_s110
                                 + 0.0155430942169327 w_k1_PG_s111
                                 + 0.0381943466848133 w_k1_PG_s112
                                 + 0.0646740217739143 w_k1_PG_s113
                                 + 0.0397690702274768 w_k1_PG_s114
                                 + 0.0433633531621337 w_k1_PG_s115
                                 + 0.0278818751824403 w_k1_PG_s116
                                 + 0.0223250704866659 w_k1_PG_s117 >= 1.13
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
