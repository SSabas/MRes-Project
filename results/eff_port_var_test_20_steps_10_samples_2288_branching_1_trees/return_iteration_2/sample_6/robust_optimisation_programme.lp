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
 w_k1_ADM_s0_return_constraint:  1.20481 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.990108 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.853112 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.06886 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.96593 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.15315 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.19732 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.801724 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.703469 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.963752 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.958956 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.98207 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.41497 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.03617 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.0622 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.30238 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.00432 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.02434 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.11997 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.05907 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.822316 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.82507 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.751871 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.8244 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.06699 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.898125 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.03746 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.912254 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.13955 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.15832 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.4244 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.15206 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.17877 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.08098 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.29973 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.00694 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.911826 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.06749 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.21238 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.924284 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.795317 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.20655 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.890908 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.44328 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.19151 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.864508 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.685096 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.959461 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.988361 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.757911 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.50263 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.23512 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.15852 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.44347 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.869046 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.96775 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.108 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.17738 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.56558 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.708842 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.60355 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.781898 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.20268 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.917938 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.6042 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.970535 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.41208 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.1248 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.66649 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.04326 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.12681 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.1377 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.24834 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.898827 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.846589 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.76865 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.31533 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.889605 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.694475 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.17067 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.944073 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.64517 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.07217 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.779208 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.63795 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.34184 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.00125 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.916794 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.75777 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.46784 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.35002 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.79801 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.833902 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.19262 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.00548 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.22977 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.543481 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.768148 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.456397 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.709714 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.07718 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.881816 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.32741 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.00169 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.42405 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.38082 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.62537 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.901963 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.27827 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.938049 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.33618 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.813947 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.673723 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.90186 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.2384 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.917705 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.846966 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.03809 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.945231 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.29441 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.17326 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.814072 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.743693 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.08383 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.976395 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.890899 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.54323 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.11923 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.17455 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.52696 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.08404 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.27218 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.05796 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.04868 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.725105 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.857242 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.62994 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.870518 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.25054 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.801259 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.18945 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.930761 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.12108 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.23655 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.28012 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.02659 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.1313 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.957349 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.37261 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.806577 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.857215 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.982082 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.04163 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.01926 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.706161 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.18158 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.930008 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.43369 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.808632 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.13566 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.776399 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.02846 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.13759 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.714912 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.916457 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.54697 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.27586 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.09299 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.606447 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.743519 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.890786 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.76449 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.63763 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.959104 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.603679 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.698125 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.682057 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.38479 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.29532 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.12901 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.8925 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.977972 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.5354 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.858915 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.0545 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.35469 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.814406 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.1999 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.719883 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.584891 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09852 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.945222 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.87671 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.07128 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.847838 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.17622 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.12568 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.861497 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.81182 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.12273 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.986068 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.854768 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.32362 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.35224 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.18344 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.23954 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.983942 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.23109 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.01778 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.07315 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.794246 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.952555 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.880437 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.00303 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.31807 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.82251 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.2386 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.09745 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.28398 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.20298 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.29888 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.955142 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.32425 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.09519 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.40993 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.924919 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.867849 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.885375 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1152 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.898292 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.705999 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.07119 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.00787 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.51637 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.868875 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.874456 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.800017 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.22334 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.11404 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.925124 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.41598 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.54002 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.24578 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.42207 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.788658 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.09383 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.07455 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.28522 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.691795 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.950828 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.607383 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.750744 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.901408 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.03822 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.26847 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.08849 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.37595 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.09694 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.49353 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.898752 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.30736 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.1714 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.0314 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.06942 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.734013 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.789512 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.16039 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.935151 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.599141 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.20244 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.02714 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.95019 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.892048 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.47046 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.655534 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.04255 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.33429 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.927129 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.49392 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.95663 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.32362 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.951163 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.46622 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.741371 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.01126 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.50155 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.498101 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.789557 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.558912 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.618537 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.857395 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.17571 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.45477 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.03481 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.7026 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.03758 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.44762 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.700353 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.884908 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.35708 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.59412 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.836871 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.673236 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.438418 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14797 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.938149 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.859563 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.05317 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.900683 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.30723 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.13674 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.875294 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.715251 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.1342 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.07395 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.966977 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.67627 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.16719 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.26478 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.34946 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.0136 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.22104 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.02726 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.17676 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.721564 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.954359 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.659738 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.932862 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.1669 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.928795 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.24372 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.00704 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.22532 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.37851 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.31952 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.89287 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.09605 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.976092 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.27964 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.895016 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.812013 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.01404 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09624 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.93697 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.844596 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.10998 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.971592 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.33723 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.04354 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.940043 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.804183 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.06104 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.9877 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.965729 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.34027 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.15447 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.06363 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.18852 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.90196 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.12464 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.03625 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.0348 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.760963 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.833772 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.712387 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.844028 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.14873 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.973016 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.0901 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.02338 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.32688 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.12813 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.16424 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.85174 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.00451 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.00066 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.03325 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.877406 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.844377 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.856525 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.102999704283863 z_k1_s0000
                                 - 0.419999447097709 z_k1_s0001
                                 - 0.165999475148863 z_k1_s0002
                                 - 0.173999520312498 z_k1_s0003
                                 - 0.0969998166107925 z_k1_s0004
                                 - 0.158999435630683 z_k1_s0005
                                 - 0.198999661448857 z_k1_s0006
                                 - 0.143999643348179 z_k1_s0007
                                 - 0.132000186268601 z_k1_s0010
                                 - 0.26399906953946 z_k1_s0011
                                 - 0.188999822498994 z_k1_s0012
                                 - 0.12899977336775 z_k1_s0013
                                 - 0.065999832534752 z_k1_s0014
                                 - 0.191999323301425 z_k1_s0015
                                 - 0.208999534843268 z_k1_s0016
                                 - 0.121999808897367 z_k1_s0017
                                 - 0.071000073605826 z_k1_s0020
                                 - 0.173999908205492 z_k1_s0021
                                 - 0.133000147332052 z_k1_s0022
                                 - 0.281000383930214 z_k1_s0023
                                 - 0.160000146970852 z_k1_s0024
                                 - 0.274000503312894 z_k1_s0025
                                 - 0.153000266353532 z_k1_s0026
                                 - 0.0960000881825113 z_k1_s0027
                                 - 0.282000179435425 z_k1_s0030
                                 - 0.0669999490929001 z_k1_s0031
                                 - 0.189999904442507 z_k1_s0032
                                 - 0.0619999057128765 z_k1_s0033
                                 - 0.109999428361929 z_k1_s0034
                                 - 0.0959999172972987 z_k1_s0035
                                 - 0.0659999840168551 z_k1_s0036
                                 - 0.217000269494514 z_k1_s0037
                                 - 0.197999820679519 z_k1_s0040
                                 - 0.0800000296140419 z_k1_s0041
                                 - 0.105000058005948 z_k1_s0042
                                 - 0.0990000634399017 z_k1_s0043
                                 - 0.251000027846496 z_k1_s0044
                                 - 0.261000529123713 z_k1_s0045
                                 - 0.349000959759528 z_k1_s0046
                                 - 0.188000850403723 z_k1_s0047
                                 - 0.0280000053180937 z_k1_s0050
                                 - 0.0610001262001593 z_k1_s0051
                                 - 0.0230000253541533 z_k1_s0052
                                 - 0.0390000154787752 z_k1_s0053
                                 - 0.0929999256493611 z_k1_s0054
                                 - 0.0990002270461654 z_k1_s0055
                                 - 0.0280000053180937 z_k1_s0056
                                 - 0.0810002268559631 z_k1_s0057
                                 - 0.0420000246713483 z_k1_s0060
                                 - 0.0340000199720438 z_k1_s0061
                                 - 0.0340000199720438 z_k1_s0062
                                 - 0.0570000334825441 z_k1_s0063
                                 - 0.0620000364196094 z_k1_s0064
                                 - 0.0500000293706527 z_k1_s0065
                                 - 0.0500000293706527 z_k1_s0066
                                 - 0.0710000417063268 z_k1_s0067
                                 - 0.0660000722600846 z_k1_s0070
                                 - 0.262000596017773 z_k1_s0071
                                 - 0.0830001129558974 z_k1_s0072
                                 - 0.185999806141456 z_k1_s0073
                                 - 0.128000228474237 z_k1_s0074
                                 - 0.220000682534283 z_k1_s0075
                                 - 0.0720000788291832 z_k1_s0076
                                 - 0.308000690547596 z_k1_s0077
                                 - 0.214999965895515 z_k1_s0100
                                 - 0.211999712331615 z_k1_s0101
                                 - 0.776999126308637 z_k1_s0102
                                 - 0.451999907861287 z_k1_s0103
                                 - 0.392001114577766 z_k1_s0104
                                 - 0.540999896663614 z_k1_s0105
                                 - 0.197999784632311 z_k1_s0106
                                 - 0.352000872723004 z_k1_s0107
                                 - 0.23599971506307 z_k1_s0110
                                 - 0.431999569471265 z_k1_s0111
                                 - 0.469998362694775 z_k1_s0112
                                 - 0.331000727116673 z_k1_s0113
                                 - 0.197999847440939 z_k1_s0114
                                 - 0.1689996798665 z_k1_s0115
                                 - 0.332999108551485 z_k1_s0116
                                 - 0.51700061636115 z_k1_s0117
                                 - 0.0850000214742538 z_k1_s0120
                                 - 0.305000653336827 z_k1_s0121
                                 - 0.094999913177146 z_k1_s0122
                                 - 0.201000121707328 z_k1_s0123
                                 - 0.136999910489135 z_k1_s0124
                                 - 0.417000646168797 z_k1_s0125
                                 - 0.245999445970409 z_k1_s0126
                                 - 0.398000512813421 z_k1_s0127
                                 - 0.184999608174739 z_k1_s0130
                                 - 0.106000081981381 z_k1_s0131
                                 - 0.212999974547381 z_k1_s0132
                                 - 0.0880000264562167 z_k1_s0133
                                 - 0.375999969858297 z_k1_s0134
                                 - 0.159999618556506 z_k1_s0135
                                 - 0.265000204953453 z_k1_s0136
                                 - 0.182000806421251 z_k1_s0137
                                 - 0.145999993904852 z_k1_s0140
                                 - 0.255000017655716 z_k1_s0141
                                 - 0.317000637697101 z_k1_s0142
                                 - 0.228999424412769 z_k1_s0143
                                 - 0.120999963817055 z_k1_s0144
                                 - 0.484999385424035 z_k1_s0145
                                 - 0.141999807282921 z_k1_s0146
                                 - 0.370999644895425 z_k1_s0147
                                 - 0.156000445957236 z_k1_s0150
                                 - 0.149000491896533 z_k1_s0151
                                 - 0.544002082734513 z_k1_s0152
                                 - 0.463002049030389 z_k1_s0153
                                 - 0.323000329808388 z_k1_s0154
                                 - 0.448002335900878 z_k1_s0155
                                 - 0.297000123581786 z_k1_s0156
                                 - 0.258000869445056 z_k1_s0157
                                 - 0.139999882856141 z_k1_s0160
                                 - 0.333999426828335 z_k1_s0161
                                 - 0.110999956071636 z_k1_s0162
                                 - 0.155999869468271 z_k1_s0163
                                 - 0.221999912143271 z_k1_s0164
                                 - 0.24599910886177 z_k1_s0165
                                 - 0.442999286673492 z_k1_s0166
                                 - 0.30600003765688 z_k1_s0167
                                 - 1.23299878371619 z_k1_s0170
                                 - 0.454000465226942 z_k1_s0171
                                 - 0.293000355818967 z_k1_s0172
                                 - 1.4179983800076 z_k1_s0173
                                 - 1.01800238594723 z_k1_s0174
                                 - 1.21500343043152 z_k1_s0175
                                 - 0.600000128194745 z_k1_s0176
                                 - 0.58800072570326 z_k1_s0177
                                 - 0.311000114374782 z_k1_s1000
                                 - 0.19200013405649 z_k1_s1001
                                 - 0.22600007983312 z_k1_s1002
                                 - 0.365000768690989 z_k1_s1003
                                 - 0.65200044105849 z_k1_s1004
                                 - 0.606998761794398 z_k1_s1005
                                 - 0.459001479285646 z_k1_s1006
                                 - 0.589998959006171 z_k1_s1007
                                 - 1.3210015120614 z_k1_s1010
                                 - 2.13000027097422 z_k1_s1011
                                 - 1.33600139538086 z_k1_s1012
                                 - 2.9399995834287 z_k1_s1013
                                 - 0.70300042543584 z_k1_s1014
                                 - 3.38300764453082 z_k1_s1015
                                 - 0.946000219172184 z_k1_s1016
                                 - 1.27400075501971 z_k1_s1017
                                 - 0.752000372584792 z_k1_s1020
                                 - 0.421000371673372 z_k1_s1021
                                 - 0.901002349070447 z_k1_s1022
                                 - 1.43700416782995 z_k1_s1023
                                 - 0.885000725821615 z_k1_s1024
                                 - 0.445000470543226 z_k1_s1025
                                 - 0.592001660121928 z_k1_s1026
                                 - 0.407000703333189 z_k1_s1027
                                 - 1.14000300636694 z_k1_s1030
                                 - 0.455000452414478 z_k1_s1031
                                 - 1.22000066491765 z_k1_s1032
                                 - 0.349000437233131 z_k1_s1033
                                 - 0.553000182140477 z_k1_s1034
                                 - 0.575000057842198 z_k1_s1035
                                 - 0.25800024283018 z_k1_s1036
                                 - 0.64599823960609 z_k1_s1037
                                 - 0.717997539541702 z_k1_s1040
                                 - 0.574999010444409 z_k1_s1041
                                 - 0.557998860076843 z_k1_s1042
                                 - 0.459998845956244 z_k1_s1043
                                 - 0.472999316236621 z_k1_s1044
                                 - 0.648000118021868 z_k1_s1045
                                 - 2.01199853954065 z_k1_s1046
                                 - 0.595998840899164 z_k1_s1047
                                 - 1.07300309159667 z_k1_s1050
                                 - 1.41200480109724 z_k1_s1051
                                 - 0.533002141106597 z_k1_s1052
                                 - 0.763005007287219 z_k1_s1053
                                 - 1.06800483346885 z_k1_s1054
                                 - 0.545001704626939 z_k1_s1055
                                 - 0.318000913334323 z_k1_s1056
                                 - 0.52800201097199 z_k1_s1057
                                 - 1.27600006198985 z_k1_s1060
                                 - 0.603999342370953 z_k1_s1061
                                 - 0.26699997115553 z_k1_s1062
                                 - 0.527999204271163 z_k1_s1063
                                 - 0.622002509291225 z_k1_s1064
                                 - 0.579998295812516 z_k1_s1065
                                 - 0.447000197939183 z_k1_s1066
                                 - 0.44000044849331 z_k1_s1067
                                 - 1.10400358608681 z_k1_s1070
                                 - 0.651999660263445 z_k1_s1071
                                 - 0.851999194700384 z_k1_s1072
                                 - 3.20999984270681 z_k1_s1073
                                 - 0.650999311091551 z_k1_s1074
                                 - 1.68600160429882 z_k1_s1075
                                 - 0.693999125495564 z_k1_s1076
                                 - 0.650999311091551 z_k1_s1077
                                 - 0.108999683241507 z_k1_s1100
                                 - 0.143999731782946 z_k1_s1101
                                 - 0.232999850342615 z_k1_s1102
                                 - 0.23499927795196 z_k1_s1103
                                 - 0.214999883869746 z_k1_s1104
                                 - 0.16699963207618 z_k1_s1105
                                 - 0.212998750264147 z_k1_s1106
                                 - 0.389999273578812 z_k1_s1107
                                 - 0.212999666495156 z_k1_s1110
                                 - 0.0559999281597489 z_k1_s1111
                                 - 0.0959995946740413 z_k1_s1112
                                 - 0.134999892947565 z_k1_s1113
                                 - 0.146000103691296 z_k1_s1114
                                 - 0.0489999885772463 z_k1_s1115
                                 - 0.0719999429053681 z_k1_s1116
                                 - 0.0559999281597489 z_k1_s1117
                                 - 0.212998833496142 z_k1_s1120
                                 - 0.137999720128777 z_k1_s1121
                                 - 0.258999653403589 z_k1_s1122
                                 - 0.142999630580006 z_k1_s1123
                                 - 0.113999662924021 z_k1_s1124
                                 - 0.257999701753271 z_k1_s1125
                                 - 0.220000017044788 z_k1_s1126
                                 - 0.176999052088293 z_k1_s1127
                                 - 0.16700044976562 z_k1_s1130
                                 - 0.108999971279334 z_k1_s1131
                                 - 0.116000039572507 z_k1_s1132
                                 - 0.400000189181063 z_k1_s1133
                                 - 0.295000693783573 z_k1_s1134
                                 - 0.139999989763367 z_k1_s1135
                                 - 0.102000055886172 z_k1_s1136
                                 - 0.199999330090477 z_k1_s1137
                                 - 0.203000253944763 z_k1_s1140
                                 - 0.177999593725764 z_k1_s1141
                                 - 0.101999902137064 z_k1_s1142
                                 - 0.100999919008249 z_k1_s1143
                                 - 0.157000164420374 z_k1_s1144
                                 - 0.225999216708332 z_k1_s1145
                                 - 0.0629999109141053 z_k1_s1146
                                 - 0.0519998800974157 z_k1_s1147
                                 - 0.64199693631908 z_k1_s1150
                                 - 0.295999102226689 z_k1_s1151
                                 - 0.489998588253436 z_k1_s1152
                                 - 0.23799947166552 z_k1_s1153
                                 - 0.195999435301374 z_k1_s1154
                                 - 0.322999243631885 z_k1_s1155
                                 - 0.432999207728819 z_k1_s1156
                                 - 0.135999580066337 z_k1_s1157
                                 - 0.415000003831989 z_k1_s1160
                                 - 0.144000079775504 z_k1_s1161
                                 - 0.0860000423242548 z_k1_s1162
                                 - 0.121000144165593 z_k1_s1163
                                 - 0.267001017646988 z_k1_s1164
                                 - 0.428000949916689 z_k1_s1165
                                 - 0.177000153911605 z_k1_s1166
                                 - 0.277001156173178 z_k1_s1167
                                 - 0.278000050640016 z_k1_s1170
                                 - 0.278000050640016 z_k1_s1171
                                 - 0.188999997317135 z_k1_s1172
                                 - 0.120000051833099 z_k1_s1173
                                 - 0.217999925523471 z_k1_s1174
                                 - 0.21600005361958 z_k1_s1175
                                 - 0.518000551106195 z_k1_s1176
                                 - 0.167000017574399 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.953867 w_k1_ADM_s000
                                 + 1.25877 w_k1_AXP_s000
                                 + 0.906865 w_k1_DIS_s000
                                 + 0.947615 w_k1_FDX_s000
                                 + 1.03692 w_k1_IBM_s000
                                 + 0.919315 w_k1_INTC_s000
                                 + 1.0657 w_k1_KO_s000 + 1.82261 w_k1_MAT_s000
                                 + 0.794714 w_k1_MSFT_s000
                                 + 1.00437 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.926619 w_k1_ADM_s000
                                 + 1.19212 w_k1_AXP_s000
                                 + 1.36746 w_k1_DIS_s000
                                 + 1.23551 w_k1_FDX_s000
                                 + 0.961086 w_k1_IBM_s000
                                 + 0.990584 w_k1_INTC_s000
                                 + 1.24619 w_k1_KO_s000 + 1.23861 w_k1_MAT_s000
                                 + 1.11285 w_k1_MSFT_s000
                                 + 1.10823 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.08404 w_k1_ADM_s000
                                 + 1.48465 w_k1_AXP_s000
                                 + 1.69379 w_k1_DIS_s000
                                 + 1.23479 w_k1_FDX_s000
                                 + 1.56422 w_k1_IBM_s000
                                 + 1.23511 w_k1_INTC_s000
                                 + 1.60605 w_k1_KO_s000 + 1.58848 w_k1_MAT_s000
                                 + 1.19318 w_k1_MSFT_s000
                                 + 1.25569 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.897053 w_k1_ADM_s000
                                 + 0.973924 w_k1_AXP_s000
                                 + 0.949111 w_k1_DIS_s000
                                 + 0.913714 w_k1_FDX_s000 + 1.209 w_k1_IBM_s000
                                 + 0.94218 w_k1_INTC_s000
                                 + 1.16365 w_k1_KO_s000 + 1.26754 w_k1_MAT_s000
                                 + 0.814434 w_k1_MSFT_s000
                                 + 0.952469 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.695064 w_k1_ADM_s000
                                 + 0.725051 w_k1_AXP_s000
                                 + 0.883211 w_k1_DIS_s000
                                 + 0.82518 w_k1_FDX_s000
                                 + 1.15733 w_k1_IBM_s000
                                 + 0.821478 w_k1_INTC_s000
                                 + 1.10502 w_k1_KO_s000
                                 + 0.864031 w_k1_MAT_s000
                                 + 0.757144 w_k1_MSFT_s000
                                 + 0.910823 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.09186 w_k1_ADM_s000
                                 + 0.724403 w_k1_AXP_s000
                                 + 0.969988 w_k1_DIS_s000
                                 + 0.940527 w_k1_FDX_s000
                                 + 0.64879 w_k1_IBM_s000
                                 + 0.897498 w_k1_INTC_s000
                                 + 0.938529 w_k1_KO_s000
                                 + 0.422758 w_k1_MAT_s000
                                 + 0.954235 w_k1_MSFT_s000
                                 + 0.914109 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.975889 w_k1_ADM_s000
                                 + 0.913885 w_k1_AXP_s000
                                 + 0.692741 w_k1_DIS_s000
                                 + 0.915809 w_k1_FDX_s000
                                 + 0.778878 w_k1_IBM_s000
                                 + 0.890689 w_k1_INTC_s000
                                 + 0.822981 w_k1_KO_s000
                                 + 1.02805 w_k1_MAT_s000
                                 + 0.851294 w_k1_MSFT_s000
                                 + 0.812353 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.72866 w_k1_ADM_s000
                                 + 0.710723 w_k1_AXP_s000
                                 + 0.681543 w_k1_DIS_s000
                                 + 0.689828 w_k1_FDX_s000
                                 + 0.777705 w_k1_IBM_s000
                                 + 0.68756 w_k1_INTC_s000
                                 + 0.911922 w_k1_KO_s000
                                 + 0.804979 w_k1_MAT_s000
                                 + 0.675717 w_k1_MSFT_s000
                                 + 0.838222 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.15718 w_k1_ADM_s001
                                 + 1.18141 w_k1_AXP_s001
                                 + 1.01253 w_k1_DIS_s001
                                 + 1.03548 w_k1_FDX_s001
                                 + 1.02761 w_k1_IBM_s001
                                 + 1.09355 w_k1_INTC_s001
                                 + 1.13452 w_k1_KO_s001
                                 + 0.808649 w_k1_MAT_s001
                                 + 1.08172 w_k1_MSFT_s001
                                 + 1.04602 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.34682 w_k1_ADM_s001
                                 + 1.06449 w_k1_AXP_s001
                                 + 1.14905 w_k1_DIS_s001
                                 + 1.19191 w_k1_FDX_s001
                                 + 0.881754 w_k1_IBM_s001
                                 + 1.17998 w_k1_INTC_s001
                                 + 1.11149 w_k1_KO_s001
                                 + 0.429276 w_k1_MAT_s001
                                 + 1.18492 w_k1_MSFT_s001
                                 + 1.13984 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.05765 w_k1_ADM_s001
                                 + 0.841289 w_k1_AXP_s001
                                 + 0.73559 w_k1_DIS_s001
                                 + 0.844056 w_k1_FDX_s001
                                 + 0.693592 w_k1_IBM_s001
                                 + 0.752929 w_k1_INTC_s001
                                 + 0.687517 w_k1_KO_s001
                                 + 0.738107 w_k1_MAT_s001
                                 + 0.78051 w_k1_MSFT_s001
                                 + 0.833951 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.21841 w_k1_ADM_s001
                                 + 1.04724 w_k1_AXP_s001
                                 + 0.896016 w_k1_DIS_s001
                                 + 0.988703 w_k1_FDX_s001
                                 + 0.865891 w_k1_IBM_s001
                                 + 0.972646 w_k1_INTC_s001
                                 + 0.987106 w_k1_KO_s001
                                 + 0.87111 w_k1_MAT_s001
                                 + 0.996205 w_k1_MSFT_s001
                                 + 1.05646 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.0091 w_k1_ADM_s001
                                 + 0.979011 w_k1_AXP_s001
                                 + 0.708992 w_k1_DIS_s001
                                 + 0.811719 w_k1_FDX_s001
                                 + 0.879859 w_k1_IBM_s001
                                 + 0.90187 w_k1_INTC_s001
                                 + 0.911937 w_k1_KO_s001
                                 + 0.784486 w_k1_MAT_s001
                                 + 0.750563 w_k1_MSFT_s001
                                 + 0.977249 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.98489 w_k1_ADM_s001
                                 + 0.965674 w_k1_AXP_s001
                                 + 0.936713 w_k1_DIS_s001
                                 + 0.863067 w_k1_FDX_s001
                                 + 1.30675 w_k1_IBM_s001
                                 + 0.851218 w_k1_INTC_s001
                                 + 1.2095 w_k1_KO_s001 + 0.970129 w_k1_MAT_s001
                                 + 0.804157 w_k1_MSFT_s001
                                 + 1.01606 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.32961 w_k1_ADM_s001
                                 + 1.57138 w_k1_AXP_s001
                                 + 1.60586 w_k1_DIS_s001
                                 + 1.25204 w_k1_FDX_s001
                                 + 1.34912 w_k1_IBM_s001
                                 + 1.09571 w_k1_INTC_s001
                                 + 1.38531 w_k1_KO_s001 + 1.04208 w_k1_MAT_s001
                                 + 1.08346 w_k1_MSFT_s001
                                 + 1.19951 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.69629 w_k1_ADM_s001
                                 + 0.664733 w_k1_AXP_s001
                                 + 0.81068 w_k1_DIS_s001
                                 + 0.683298 w_k1_FDX_s001
                                 + 1.01922 w_k1_IBM_s001
                                 + 0.903644 w_k1_INTC_s001
                                 + 1.0606 w_k1_KO_s001 + 1.01385 w_k1_MAT_s001
                                 + 0.911705 w_k1_MSFT_s001
                                 + 0.835573 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.40498 w_k1_ADM_s002
                                 + 1.22435 w_k1_AXP_s002
                                 + 1.47537 w_k1_DIS_s002
                                 + 1.33844 w_k1_FDX_s002
                                 + 0.768776 w_k1_IBM_s002
                                 + 1.33718 w_k1_INTC_s002
                                 + 1.02647 w_k1_KO_s002
                                 + 0.929839 w_k1_MAT_s002
                                 + 1.50278 w_k1_MSFT_s002
                                 + 0.991364 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.989325 w_k1_ADM_s002
                                 + 0.639132 w_k1_AXP_s002
                                 + 0.75081 w_k1_DIS_s002
                                 + 0.777423 w_k1_FDX_s002
                                 + 0.619787 w_k1_IBM_s002
                                 + 0.895572 w_k1_INTC_s002
                                 + 0.631203 w_k1_KO_s002
                                 + 0.629326 w_k1_MAT_s002
                                 + 0.841225 w_k1_MSFT_s002
                                 + 0.852229 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.831413 w_k1_ADM_s002
                                 + 0.692893 w_k1_AXP_s002
                                 + 0.839512 w_k1_DIS_s002
                                 + 0.745842 w_k1_FDX_s002
                                 + 0.951631 w_k1_IBM_s002
                                 + 0.841241 w_k1_INTC_s002
                                 + 0.95827 w_k1_KO_s002 + 1.29012 w_k1_MAT_s002
                                 + 0.936235 w_k1_MSFT_s002
                                 + 0.881436 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.972129 w_k1_ADM_s002
                                 + 0.971015 w_k1_AXP_s002
                                 + 1.20384 w_k1_DIS_s002
                                 + 0.944778 w_k1_FDX_s002 + 1.345 w_k1_IBM_s002
                                 + 0.863397 w_k1_INTC_s002
                                 + 1.11003 w_k1_KO_s002 + 1.47008 w_k1_MAT_s002
                                 + 1.03641 w_k1_MSFT_s002
                                 + 1.08235 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.1566 w_k1_ADM_s002
                                 + 0.890138 w_k1_AXP_s002
                                 + 0.984971 w_k1_DIS_s002
                                 + 1.06317 w_k1_FDX_s002
                                 + 0.682281 w_k1_IBM_s002
                                 + 1.03812 w_k1_INTC_s002
                                 + 0.842532 w_k1_KO_s002
                                 + 0.641673 w_k1_MAT_s002
                                 + 1.16987 w_k1_MSFT_s002
                                 + 0.922974 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.10421 w_k1_ADM_s002
                                 + 1.07089 w_k1_AXP_s002
                                 + 1.24946 w_k1_DIS_s002 + 1.145 w_k1_FDX_s002
                                 + 0.984838 w_k1_IBM_s002
                                 + 1.27109 w_k1_INTC_s002
                                 + 1.06672 w_k1_KO_s002 + 1.25332 w_k1_MAT_s002
                                 + 1.29758 w_k1_MSFT_s002
                                 + 1.08186 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.943434 w_k1_ADM_s002
                                 + 0.86496 w_k1_AXP_s002
                                 + 0.940118 w_k1_DIS_s002
                                 + 0.871526 w_k1_FDX_s002
                                 + 0.938807 w_k1_IBM_s002
                                 + 0.986562 w_k1_INTC_s002
                                 + 0.968562 w_k1_KO_s002
                                 + 1.25518 w_k1_MAT_s002
                                 + 1.0565 w_k1_MSFT_s002
                                 + 0.989097 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.57417 w_k1_ADM_s002
                                 + 1.4749 w_k1_AXP_s002 + 1.83468 w_k1_DIS_s002
                                 + 1.38675 w_k1_FDX_s002
                                 + 1.22838 w_k1_IBM_s002
                                 + 1.34257 w_k1_INTC_s002
                                 + 1.34036 w_k1_KO_s002 + 1.436 w_k1_MAT_s002
                                 + 1.48552 w_k1_MSFT_s002
                                 + 1.33893 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.05283 w_k1_ADM_s003
                                 + 0.96724 w_k1_AXP_s003
                                 + 1.21995 w_k1_DIS_s003
                                 + 1.04088 w_k1_FDX_s003
                                 + 0.884301 w_k1_IBM_s003
                                 + 1.14029 w_k1_INTC_s003
                                 + 0.999477 w_k1_KO_s003
                                 + 0.730917 w_k1_MAT_s003
                                 + 1.19415 w_k1_MSFT_s003
                                 + 1.03598 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.08415 w_k1_ADM_s003
                                 + 0.995267 w_k1_AXP_s003
                                 + 0.924049 w_k1_DIS_s003
                                 + 0.870482 w_k1_FDX_s003
                                 + 1.44311 w_k1_IBM_s003
                                 + 0.835217 w_k1_INTC_s003
                                 + 1.16717 w_k1_KO_s003 + 1.61363 w_k1_MAT_s003
                                 + 0.884996 w_k1_MSFT_s003
                                 + 1.00961 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.18213 w_k1_ADM_s003
                                 + 1.28138 w_k1_AXP_s003 + 1.4177 w_k1_DIS_s003
                                 + 1.3583 w_k1_FDX_s003
                                 + 0.968992 w_k1_IBM_s003
                                 + 1.22261 w_k1_INTC_s003
                                 + 1.20948 w_k1_KO_s003 + 1.39989 w_k1_MAT_s003
                                 + 1.35838 w_k1_MSFT_s003
                                 + 1.31483 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.950538 w_k1_ADM_s003
                                 + 0.914087 w_k1_AXP_s003
                                 + 1.16736 w_k1_DIS_s003
                                 + 0.846542 w_k1_FDX_s003
                                 + 1.44109 w_k1_IBM_s003
                                 + 0.977473 w_k1_INTC_s003
                                 + 1.3394 w_k1_KO_s003 + 1.10128 w_k1_MAT_s003
                                 + 1.05471 w_k1_MSFT_s003
                                 + 1.07337 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.24291 w_k1_ADM_s003
                                 + 1.08258 w_k1_AXP_s003
                                 + 0.942178 w_k1_DIS_s003
                                 + 0.967372 w_k1_FDX_s003
                                 + 0.983916 w_k1_IBM_s003
                                 + 0.996465 w_k1_INTC_s003
                                 + 0.980282 w_k1_KO_s003
                                 + 1.18873 w_k1_MAT_s003
                                 + 1.06235 w_k1_MSFT_s003
                                 + 1.04013 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.39003 w_k1_ADM_s003
                                 + 1.40566 w_k1_AXP_s003
                                 + 1.54342 w_k1_DIS_s003
                                 + 1.27829 w_k1_FDX_s003
                                 + 1.70388 w_k1_IBM_s003
                                 + 1.23163 w_k1_INTC_s003
                                 + 1.34621 w_k1_KO_s003 + 1.4533 w_k1_MAT_s003
                                 + 1.13899 w_k1_MSFT_s003
                                 + 1.27392 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.961656 w_k1_ADM_s003
                                 + 0.608381 w_k1_AXP_s003
                                 + 0.634346 w_k1_DIS_s003
                                 + 0.673741 w_k1_FDX_s003
                                 + 0.975767 w_k1_IBM_s003
                                 + 0.63771 w_k1_INTC_s003
                                 + 0.823739 w_k1_KO_s003
                                 + 0.955992 w_k1_MAT_s003
                                 + 0.743618 w_k1_MSFT_s003
                                 + 0.834271 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.810877 w_k1_ADM_s003
                                 + 0.863118 w_k1_AXP_s003
                                 + 0.824844 w_k1_DIS_s003
                                 + 0.833109 w_k1_FDX_s003
                                 + 0.860961 w_k1_IBM_s003
                                 + 0.942058 w_k1_INTC_s003
                                 + 1.00112 w_k1_KO_s003 + 0.76591 w_k1_MAT_s003
                                 + 0.963969 w_k1_MSFT_s003
                                 + 0.874454 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.09145 w_k1_ADM_s004
                                 + 1.29301 w_k1_AXP_s004
                                 + 1.15131 w_k1_DIS_s004
                                 + 1.11899 w_k1_FDX_s004
                                 + 1.40487 w_k1_IBM_s004
                                 + 1.22089 w_k1_INTC_s004
                                 + 1.27872 w_k1_KO_s004 + 1.48325 w_k1_MAT_s004
                                 + 1.10258 w_k1_MSFT_s004
                                 + 1.26615 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.0826 w_k1_ADM_s004
                                 + 0.995875 w_k1_AXP_s004
                                 + 1.14895 w_k1_DIS_s004
                                 + 1.20479 w_k1_FDX_s004
                                 + 0.735267 w_k1_IBM_s004
                                 + 1.11729 w_k1_INTC_s004
                                 + 1.02433 w_k1_KO_s004
                                 + 0.612749 w_k1_MAT_s004
                                 + 1.044 w_k1_MSFT_s004 + 1.12654 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.15941 w_k1_ADM_s004
                                 + 1.45457 w_k1_AXP_s004
                                 + 1.72732 w_k1_DIS_s004
                                 + 1.23092 w_k1_FDX_s004
                                 + 1.39445 w_k1_IBM_s004 + 1.34 w_k1_INTC_s004
                                 + 1.45405 w_k1_KO_s004 + 1.11162 w_k1_MAT_s004
                                 + 1.08295 w_k1_MSFT_s004
                                 + 1.29304 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.07228 w_k1_ADM_s004
                                 + 1.53709 w_k1_AXP_s004
                                 + 1.53773 w_k1_DIS_s004
                                 + 1.38757 w_k1_FDX_s004
                                 + 0.961391 w_k1_IBM_s004
                                 + 1.21178 w_k1_INTC_s004
                                 + 1.13618 w_k1_KO_s004
                                 + 0.986904 w_k1_MAT_s004
                                 + 1.08468 w_k1_MSFT_s004
                                 + 1.17782 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.983039 w_k1_ADM_s004
                                 + 0.685221 w_k1_AXP_s004
                                 + 0.842513 w_k1_DIS_s004
                                 + 0.78356 w_k1_FDX_s004
                                 + 1.19877 w_k1_IBM_s004
                                 + 0.876698 w_k1_INTC_s004
                                 + 1.05822 w_k1_KO_s004
                                 + 0.689846 w_k1_MAT_s004
                                 + 0.813192 w_k1_MSFT_s004
                                 + 0.946537 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.849659 w_k1_ADM_s004
                                 + 1.03268 w_k1_AXP_s004
                                 + 1.07548 w_k1_DIS_s004
                                 + 1.04045 w_k1_FDX_s004
                                 + 0.945804 w_k1_IBM_s004
                                 + 1.14433 w_k1_INTC_s004
                                 + 1.04419 w_k1_KO_s004
                                 + 0.933012 w_k1_MAT_s004
                                 + 0.908639 w_k1_MSFT_s004
                                 + 1.08019 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.936376 w_k1_ADM_s004
                                 + 0.764734 w_k1_AXP_s004
                                 + 0.787934 w_k1_DIS_s004
                                 + 0.833411 w_k1_FDX_s004
                                 + 0.737391 w_k1_IBM_s004
                                 + 0.899785 w_k1_INTC_s004
                                 + 0.796544 w_k1_KO_s004
                                 + 0.690998 w_k1_MAT_s004
                                 + 0.86939 w_k1_MSFT_s004
                                 + 0.909973 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.07783 w_k1_ADM_s004
                                 + 1.1504 w_k1_AXP_s004
                                 + 0.997216 w_k1_DIS_s004
                                 + 1.07324 w_k1_FDX_s004
                                 + 0.899855 w_k1_IBM_s004
                                 + 0.966963 w_k1_INTC_s004
                                 + 0.979191 w_k1_KO_s004 + 1.1127 w_k1_MAT_s004
                                 + 0.91349 w_k1_MSFT_s004
                                 + 1.19077 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.805432 w_k1_ADM_s005
                                 + 0.974758 w_k1_AXP_s005
                                 + 0.941153 w_k1_DIS_s005
                                 + 0.786882 w_k1_FDX_s005
                                 + 1.22413 w_k1_IBM_s005
                                 + 0.966562 w_k1_INTC_s005
                                 + 0.961071 w_k1_KO_s005
                                 + 0.566306 w_k1_MAT_s005
                                 + 0.785141 w_k1_MSFT_s005
                                 + 0.972506 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.06013 w_k1_ADM_s005
                                 + 0.922309 w_k1_AXP_s005
                                 + 0.854472 w_k1_DIS_s005
                                 + 0.881926 w_k1_FDX_s005
                                 + 0.941652 w_k1_IBM_s005
                                 + 0.86277 w_k1_INTC_s005
                                 + 0.901686 w_k1_KO_s005
                                 + 0.744788 w_k1_MAT_s005
                                 + 0.863919 w_k1_MSFT_s005
                                 + 0.962902 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.985358 w_k1_ADM_s005
                                 + 1.45448 w_k1_AXP_s005
                                 + 0.879798 w_k1_DIS_s005
                                 + 0.975633 w_k1_FDX_s005
                                 + 1.19354 w_k1_IBM_s005
                                 + 1.0546 w_k1_INTC_s005
                                 + 0.878865 w_k1_KO_s005
                                 + 0.953858 w_k1_MAT_s005
                                 + 0.840737 w_k1_MSFT_s005
                                 + 0.912351 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.690772 w_k1_ADM_s005
                                 + 0.75462 w_k1_AXP_s005
                                 + 0.514345 w_k1_DIS_s005
                                 + 0.775454 w_k1_FDX_s005
                                 + 0.882373 w_k1_IBM_s005
                                 + 0.986649 w_k1_INTC_s005
                                 + 0.688298 w_k1_KO_s005
                                 + 0.527025 w_k1_MAT_s005
                                 + 0.830074 w_k1_MSFT_s005
                                 + 0.798612 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.01293 w_k1_ADM_s005
                                 + 1.30225 w_k1_AXP_s005
                                 + 1.30061 w_k1_DIS_s005
                                 + 1.24316 w_k1_FDX_s005
                                 + 1.22647 w_k1_IBM_s005
                                 + 1.45016 w_k1_INTC_s005
                                 + 1.27458 w_k1_KO_s005
                                 + 0.991564 w_k1_MAT_s005
                                 + 1.1719 w_k1_MSFT_s005 + 1.14501 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.969648 w_k1_ADM_s005
                                 + 1.03125 w_k1_AXP_s005
                                 + 1.05837 w_k1_DIS_s005
                                 + 1.04634 w_k1_FDX_s005
                                 + 1.28424 w_k1_IBM_s005
                                 + 0.964001 w_k1_INTC_s005
                                 + 1.30333 w_k1_KO_s005 + 1.23398 w_k1_MAT_s005
                                 + 1.0778 w_k1_MSFT_s005 + 1.11438 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.764084 w_k1_ADM_s005
                                 + 0.692523 w_k1_AXP_s005
                                 + 0.626881 w_k1_DIS_s005
                                 + 0.823031 w_k1_FDX_s005
                                 + 0.751824 w_k1_IBM_s005
                                 + 0.966128 w_k1_INTC_s005
                                 + 0.764547 w_k1_KO_s005
                                 + 0.698559 w_k1_MAT_s005
                                 + 1.00229 w_k1_MSFT_s005
                                 + 0.824708 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.845318 w_k1_ADM_s005
                                 + 1.02241 w_k1_AXP_s005
                                 + 0.962999 w_k1_DIS_s005
                                 + 1.04592 w_k1_FDX_s005
                                 + 0.641111 w_k1_IBM_s005
                                 + 1.26558 w_k1_INTC_s005
                                 + 0.889198 w_k1_KO_s005
                                 + 0.508415 w_k1_MAT_s005
                                 + 1.07268 w_k1_MSFT_s005
                                 + 0.92817 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.658964 w_k1_ADM_s006
                                 + 0.7292 w_k1_AXP_s006
                                 + 0.918049 w_k1_DIS_s006
                                 + 0.891785 w_k1_FDX_s006
                                 + 0.895242 w_k1_IBM_s006
                                 + 0.979622 w_k1_INTC_s006
                                 + 0.836775 w_k1_KO_s006
                                 + 0.874787 w_k1_MAT_s006
                                 + 0.901489 w_k1_MSFT_s006
                                 + 0.955921 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.27513 w_k1_ADM_s006
                                 + 1.58055 w_k1_AXP_s006
                                 + 1.55953 w_k1_DIS_s006
                                 + 1.15361 w_k1_FDX_s006 + 1.7522 w_k1_IBM_s006
                                 + 1.08701 w_k1_INTC_s006 + 1.5558 w_k1_KO_s006
                                 + 1.19047 w_k1_MAT_s006
                                 + 1.06681 w_k1_MSFT_s006
                                 + 1.20738 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.74457 w_k1_ADM_s006
                                 + 0.7324 w_k1_AXP_s006
                                 + 0.586693 w_k1_DIS_s006
                                 + 0.751287 w_k1_FDX_s006
                                 + 1.12654 w_k1_IBM_s006
                                 + 0.896635 w_k1_INTC_s006
                                 + 0.819789 w_k1_KO_s006
                                 + 0.985866 w_k1_MAT_s006
                                 + 0.641576 w_k1_MSFT_s006
                                 + 0.799005 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.01159 w_k1_ADM_s006
                                 + 1.1528 w_k1_AXP_s006 + 1.22471 w_k1_DIS_s006
                                 + 1.13916 w_k1_FDX_s006
                                 + 1.07996 w_k1_IBM_s006
                                 + 1.13457 w_k1_INTC_s006 + 1.0315 w_k1_KO_s006
                                 + 0.815197 w_k1_MAT_s006
                                 + 1.00007 w_k1_MSFT_s006 + 1.0106 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.849602 w_k1_ADM_s006
                                 + 0.993576 w_k1_AXP_s006
                                 + 0.899382 w_k1_DIS_s006
                                 + 1.08705 w_k1_FDX_s006
                                 + 0.913927 w_k1_IBM_s006
                                 + 0.933489 w_k1_INTC_s006
                                 + 0.766701 w_k1_KO_s006
                                 + 0.810008 w_k1_MAT_s006
                                 + 0.808568 w_k1_MSFT_s006
                                 + 0.891013 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.19839 w_k1_ADM_s006
                                 + 1.44357 w_k1_AXP_s006
                                 + 1.62835 w_k1_DIS_s006
                                 + 1.37555 w_k1_FDX_s006
                                 + 1.29966 w_k1_IBM_s006
                                 + 1.1002 w_k1_INTC_s006 + 1.21172 w_k1_KO_s006
                                 + 0.995562 w_k1_MAT_s006
                                 + 1.01879 w_k1_MSFT_s006
                                 + 1.07228 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.757711 w_k1_ADM_s006
                                 + 0.709177 w_k1_AXP_s006
                                 + 0.600579 w_k1_DIS_s006
                                 + 0.746056 w_k1_FDX_s006
                                 + 0.615951 w_k1_IBM_s006
                                 + 0.771329 w_k1_INTC_s006
                                 + 0.539883 w_k1_KO_s006
                                 + 0.447397 w_k1_MAT_s006
                                 + 0.632666 w_k1_MSFT_s006
                                 + 0.685138 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.13277 w_k1_ADM_s006
                                 + 1.0734 w_k1_AXP_s006 + 0.90072 w_k1_DIS_s006
                                 + 0.943264 w_k1_FDX_s006
                                 + 1.09973 w_k1_IBM_s006
                                 + 1.09802 w_k1_INTC_s006
                                 + 0.932042 w_k1_KO_s006
                                 + 0.59965 w_k1_MAT_s006
                                 + 0.775897 w_k1_MSFT_s006
                                 + 1.01421 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.38004 w_k1_ADM_s007
                                 + 1.57534 w_k1_AXP_s007
                                 + 2.02221 w_k1_DIS_s007
                                 + 1.43353 w_k1_FDX_s007
                                 + 1.27752 w_k1_IBM_s007
                                 + 0.85824 w_k1_INTC_s007
                                 + 1.42946 w_k1_KO_s007 + 1.50231 w_k1_MAT_s007
                                 + 1.29562 w_k1_MSFT_s007
                                 + 1.28454 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.795907 w_k1_ADM_s007
                                 + 0.778821 w_k1_AXP_s007
                                 + 0.621068 w_k1_DIS_s007
                                 + 0.710681 w_k1_FDX_s007
                                 + 0.864084 w_k1_IBM_s007
                                 + 0.749614 w_k1_INTC_s007
                                 + 0.765546 w_k1_KO_s007
                                 + 0.609191 w_k1_MAT_s007
                                 + 0.727711 w_k1_MSFT_s007
                                 + 0.850617 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.856496 w_k1_ADM_s007
                                 + 1.17831 w_k1_AXP_s007
                                 + 1.04773 w_k1_DIS_s007
                                 + 0.869925 w_k1_FDX_s007
                                 + 1.70212 w_k1_IBM_s007
                                 + 0.841788 w_k1_INTC_s007
                                 + 1.19794 w_k1_KO_s007 + 1.47237 w_k1_MAT_s007
                                 + 0.956827 w_k1_MSFT_s007
                                 + 1.04958 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.11827 w_k1_ADM_s007
                                 + 0.925589 w_k1_AXP_s007
                                 + 1.13934 w_k1_DIS_s007 + 1.0194 w_k1_FDX_s007
                                 + 0.926101 w_k1_IBM_s007
                                 + 0.897091 w_k1_INTC_s007
                                 + 0.987661 w_k1_KO_s007
                                 + 0.776869 w_k1_MAT_s007
                                 + 1.01807 w_k1_MSFT_s007
                                 + 0.96365 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.33406 w_k1_ADM_s007
                                 + 1.05618 w_k1_AXP_s007
                                 + 0.845089 w_k1_DIS_s007
                                 + 1.04875 w_k1_FDX_s007
                                 + 0.749615 w_k1_IBM_s007
                                 + 0.799711 w_k1_INTC_s007
                                 + 0.759753 w_k1_KO_s007
                                 + 0.595796 w_k1_MAT_s007
                                 + 0.968517 w_k1_MSFT_s007
                                 + 0.969268 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.23065 w_k1_ADM_s007
                                 + 1.55517 w_k1_AXP_s007
                                 + 1.20116 w_k1_DIS_s007
                                 + 1.29519 w_k1_FDX_s007
                                 + 1.18657 w_k1_IBM_s007
                                 + 0.927946 w_k1_INTC_s007
                                 + 1.27461 w_k1_KO_s007 + 1.26351 w_k1_MAT_s007
                                 + 1.10903 w_k1_MSFT_s007
                                 + 1.14285 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.14007 w_k1_ADM_s007
                                 + 1.19112 w_k1_AXP_s007
                                 + 1.50371 w_k1_DIS_s007
                                 + 1.28991 w_k1_FDX_s007
                                 + 0.749751 w_k1_IBM_s007
                                 + 0.932118 w_k1_INTC_s007
                                 + 1.08758 w_k1_KO_s007
                                 + 0.811049 w_k1_MAT_s007
                                 + 1.29514 w_k1_MSFT_s007
                                 + 1.12193 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.04491 w_k1_ADM_s007
                                 + 0.941121 w_k1_AXP_s007
                                 + 0.790273 w_k1_DIS_s007
                                 + 0.800357 w_k1_FDX_s007
                                 + 1.25219 w_k1_IBM_s007
                                 + 0.725769 w_k1_INTC_s007
                                 + 0.954334 w_k1_KO_s007
                                 + 0.87079 w_k1_MAT_s007
                                 + 0.882981 w_k1_MSFT_s007
                                 + 1.04245 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.787726 w_k1_ADM_s010
                                 + 0.929885 w_k1_AXP_s010
                                 + 0.972001 w_k1_DIS_s010
                                 + 0.872194 w_k1_FDX_s010
                                 + 1.47616 w_k1_IBM_s010
                                 + 0.915238 w_k1_INTC_s010
                                 + 1.20913 w_k1_KO_s010 + 1.28131 w_k1_MAT_s010
                                 + 0.903608 w_k1_MSFT_s010
                                 + 1.13753 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.902062 w_k1_ADM_s010
                                 + 0.828043 w_k1_AXP_s010
                                 + 0.95476 w_k1_DIS_s010
                                 + 0.995501 w_k1_FDX_s010
                                 + 0.849056 w_k1_IBM_s010
                                 + 1.11517 w_k1_INTC_s010
                                 + 0.997162 w_k1_KO_s010
                                 + 0.95734 w_k1_MAT_s010
                                 + 1.08646 w_k1_MSFT_s010
                                 + 1.10707 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.12516 w_k1_ADM_s010
                                 + 1.32825 w_k1_AXP_s010
                                 + 1.39787 w_k1_DIS_s010
                                 + 1.25462 w_k1_FDX_s010
                                 + 1.32774 w_k1_IBM_s010
                                 + 1.31646 w_k1_INTC_s010
                                 + 1.38186 w_k1_KO_s010 + 1.38048 w_k1_MAT_s010
                                 + 1.22794 w_k1_MSFT_s010
                                 + 1.27495 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.967867 w_k1_ADM_s010
                                 + 0.942913 w_k1_AXP_s010
                                 + 0.934593 w_k1_DIS_s010
                                 + 1.0028 w_k1_FDX_s010
                                 + 0.846166 w_k1_IBM_s010
                                 + 1.08871 w_k1_INTC_s010 + 1.025 w_k1_KO_s010
                                 + 0.726718 w_k1_MAT_s010
                                 + 0.961919 w_k1_MSFT_s010
                                 + 1.03373 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.822278 w_k1_ADM_s010
                                 + 0.777832 w_k1_AXP_s010
                                 + 0.666081 w_k1_DIS_s010
                                 + 0.819659 w_k1_FDX_s010
                                 + 0.736629 w_k1_IBM_s010
                                 + 0.826896 w_k1_INTC_s010
                                 + 0.785277 w_k1_KO_s010
                                 + 0.755787 w_k1_MAT_s010
                                 + 0.773646 w_k1_MSFT_s010
                                 + 0.811908 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.776821 w_k1_ADM_s010
                                 + 0.950073 w_k1_AXP_s010
                                 + 0.801205 w_k1_DIS_s010
                                 + 0.819892 w_k1_FDX_s010
                                 + 1.08072 w_k1_IBM_s010
                                 + 0.847127 w_k1_INTC_s010
                                 + 0.960077 w_k1_KO_s010
                                 + 0.998887 w_k1_MAT_s010
                                 + 0.770119 w_k1_MSFT_s010
                                 + 0.924626 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.708098 w_k1_ADM_s010
                                 + 0.642162 w_k1_AXP_s010
                                 + 0.580153 w_k1_DIS_s010
                                 + 0.775907 w_k1_FDX_s010
                                 + 0.832822 w_k1_IBM_s010
                                 + 0.872367 w_k1_INTC_s010
                                 + 0.823707 w_k1_KO_s010
                                 + 0.614005 w_k1_MAT_s010
                                 + 0.713247 w_k1_MSFT_s010
                                 + 0.866956 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.2929 w_k1_ADM_s010
                                 + 1.38158 w_k1_AXP_s010
                                 + 1.19417 w_k1_DIS_s010 + 1.2642 w_k1_FDX_s010
                                 + 0.873886 w_k1_IBM_s010
                                 + 1.30225 w_k1_INTC_s010
                                 + 1.09289 w_k1_KO_s010
                                 + 0.843662 w_k1_MAT_s010
                                 + 1.1628 w_k1_MSFT_s010 + 1.05412 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.986471 w_k1_ADM_s011
                                 + 1.24604 w_k1_AXP_s011
                                 + 0.832444 w_k1_DIS_s011
                                 + 1.10635 w_k1_FDX_s011
                                 + 0.82255 w_k1_IBM_s011
                                 + 1.42391 w_k1_INTC_s011
                                 + 0.948603 w_k1_KO_s011
                                 + 1.33311 w_k1_MAT_s011
                                 + 1.09531 w_k1_MSFT_s011
                                 + 1.09341 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.854283 w_k1_ADM_s011
                                 + 0.81775 w_k1_AXP_s011
                                 + 0.714363 w_k1_DIS_s011
                                 + 0.753049 w_k1_FDX_s011
                                 + 0.99037 w_k1_IBM_s011
                                 + 0.866951 w_k1_INTC_s011
                                 + 0.874045 w_k1_KO_s011
                                 + 1.56223 w_k1_MAT_s011
                                 + 0.887606 w_k1_MSFT_s011
                                 + 0.88297 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.00941 w_k1_ADM_s011
                                 + 0.936731 w_k1_AXP_s011
                                 + 0.987402 w_k1_DIS_s011
                                 + 0.871497 w_k1_FDX_s011
                                 + 1.21027 w_k1_IBM_s011
                                 + 1.23782 w_k1_INTC_s011
                                 + 1.17407 w_k1_KO_s011 + 1.32715 w_k1_MAT_s011
                                 + 1.05833 w_k1_MSFT_s011
                                 + 1.01703 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.769169 w_k1_ADM_s011
                                 + 0.780354 w_k1_AXP_s011
                                 + 0.580398 w_k1_DIS_s011
                                 + 0.816142 w_k1_FDX_s011
                                 + 0.593873 w_k1_IBM_s011
                                 + 0.88142 w_k1_INTC_s011
                                 + 0.653828 w_k1_KO_s011
                                 + 0.79876 w_k1_MAT_s011
                                 + 0.842639 w_k1_MSFT_s011
                                 + 0.786452 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.0223 w_k1_ADM_s011
                                 + 0.899493 w_k1_AXP_s011
                                 + 0.769486 w_k1_DIS_s011
                                 + 0.958527 w_k1_FDX_s011
                                 + 0.807614 w_k1_IBM_s011
                                 + 1.06679 w_k1_INTC_s011
                                 + 0.796071 w_k1_KO_s011
                                 + 0.894054 w_k1_MAT_s011
                                 + 1.00816 w_k1_MSFT_s011
                                 + 0.948274 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.883807 w_k1_ADM_s011
                                 + 1.59329 w_k1_AXP_s011 + 1.7655 w_k1_DIS_s011
                                 + 1.29911 w_k1_FDX_s011
                                 + 2.06523 w_k1_IBM_s011
                                 + 1.28785 w_k1_INTC_s011 + 1.6462 w_k1_KO_s011
                                 + 3.08075 w_k1_MAT_s011
                                 + 1.29962 w_k1_MSFT_s011 + 1.2586 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.02081 w_k1_ADM_s011
                                 + 0.9066 w_k1_AXP_s011 + 1.10475 w_k1_DIS_s011
                                 + 1.0496 w_k1_FDX_s011
                                 + 0.962354 w_k1_IBM_s011
                                 + 1.1233 w_k1_INTC_s011
                                 + 0.961975 w_k1_KO_s011
                                 + 0.84015 w_k1_MAT_s011
                                 + 1.11584 w_k1_MSFT_s011
                                 + 0.973722 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.18431 w_k1_ADM_s011
                                 + 1.39409 w_k1_AXP_s011
                                 + 1.38972 w_k1_DIS_s011
                                 + 1.28668 w_k1_FDX_s011
                                 + 1.17226 w_k1_IBM_s011
                                 + 1.33058 w_k1_INTC_s011
                                 + 1.19175 w_k1_KO_s011 + 1.14346 w_k1_MAT_s011
                                 + 1.16822 w_k1_MSFT_s011
                                 + 1.09547 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.766536 w_k1_ADM_s012
                                 + 0.67044 w_k1_AXP_s012
                                 + 0.579342 w_k1_DIS_s012
                                 + 0.648729 w_k1_FDX_s012
                                 + 0.742016 w_k1_IBM_s012
                                 + 0.603291 w_k1_INTC_s012
                                 + 0.604636 w_k1_KO_s012
                                 + 1.14075 w_k1_MAT_s012
                                 + 0.576746 w_k1_MSFT_s012
                                 + 0.826295 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.0298 w_k1_ADM_s012
                                 + 1.27009 w_k1_AXP_s012
                                 + 1.07514 w_k1_DIS_s012
                                 + 1.13197 w_k1_FDX_s012
                                 + 0.897716 w_k1_IBM_s012
                                 + 1.11852 w_k1_INTC_s012
                                 + 1.23824 w_k1_KO_s012 + 1.9011 w_k1_MAT_s012
                                 + 1.1177 w_k1_MSFT_s012 + 1.08848 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.04511 w_k1_ADM_s012
                                 + 0.984941 w_k1_AXP_s012
                                 + 1.33175 w_k1_DIS_s012
                                 + 1.04263 w_k1_FDX_s012
                                 + 1.46112 w_k1_IBM_s012
                                 + 1.33375 w_k1_INTC_s012
                                 + 1.50702 w_k1_KO_s012 + 1.87123 w_k1_MAT_s012
                                 + 1.19724 w_k1_MSFT_s012 + 1.1635 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.913935 w_k1_ADM_s012
                                 + 1.39729 w_k1_AXP_s012
                                 + 1.38131 w_k1_DIS_s012
                                 + 1.11492 w_k1_FDX_s012
                                 + 1.32823 w_k1_IBM_s012
                                 + 1.17005 w_k1_INTC_s012
                                 + 1.25426 w_k1_KO_s012 + 1.99754 w_k1_MAT_s012
                                 + 1.14035 w_k1_MSFT_s012 + 1.1332 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.835787 w_k1_ADM_s012
                                 + 0.867835 w_k1_AXP_s012
                                 + 0.577637 w_k1_DIS_s012
                                 + 0.775752 w_k1_FDX_s012
                                 + 0.748637 w_k1_IBM_s012
                                 + 0.852973 w_k1_INTC_s012
                                 + 0.746757 w_k1_KO_s012
                                 + 1.09528 w_k1_MAT_s012
                                 + 0.812997 w_k1_MSFT_s012
                                 + 0.876593 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.889984 w_k1_ADM_s012
                                 + 0.707419 w_k1_AXP_s012
                                 + 0.985718 w_k1_DIS_s012
                                 + 0.898643 w_k1_FDX_s012
                                 + 0.850023 w_k1_IBM_s012
                                 + 0.972732 w_k1_INTC_s012
                                 + 1.07087 w_k1_KO_s012
                                 + 0.817159 w_k1_MAT_s012
                                 + 0.96751 w_k1_MSFT_s012
                                 + 0.876182 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.31446 w_k1_ADM_s012
                                 + 1.12884 w_k1_AXP_s012
                                 + 1.58252 w_k1_DIS_s012
                                 + 1.31346 w_k1_FDX_s012
                                 + 1.10896 w_k1_IBM_s012
                                 + 1.17812 w_k1_INTC_s012
                                 + 1.31025 w_k1_KO_s012 + 1.55798 w_k1_MAT_s012
                                 + 1.30928 w_k1_MSFT_s012
                                 + 1.28357 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.816994 w_k1_ADM_s012
                                 + 0.90354 w_k1_AXP_s012
                                 + 0.80877 w_k1_DIS_s012
                                 + 0.852168 w_k1_FDX_s012
                                 + 1.1225 w_k1_IBM_s012
                                 + 0.884201 w_k1_INTC_s012
                                 + 1.11322 w_k1_KO_s012 + 1.54728 w_k1_MAT_s012
                                 + 0.877839 w_k1_MSFT_s012
                                 + 1.00767 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.49287 w_k1_ADM_s013
                                 + 1.49991 w_k1_AXP_s013
                                 + 1.55216 w_k1_DIS_s013
                                 + 1.34022 w_k1_FDX_s013
                                 + 1.18719 w_k1_IBM_s013
                                 + 1.09311 w_k1_INTC_s013
                                 + 1.05186 w_k1_KO_s013 + 1.29055 w_k1_MAT_s013
                                 + 1.34791 w_k1_MSFT_s013
                                 + 1.08944 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.889783 w_k1_ADM_s013
                                 + 0.630471 w_k1_AXP_s013
                                 + 0.844191 w_k1_DIS_s013
                                 + 0.793655 w_k1_FDX_s013
                                 + 0.768382 w_k1_IBM_s013
                                 + 0.91464 w_k1_INTC_s013
                                 + 0.844229 w_k1_KO_s013
                                 + 0.556805 w_k1_MAT_s013
                                 + 1.01271 w_k1_MSFT_s013
                                 + 0.712946 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.18701 w_k1_ADM_s013
                                 + 1.06304 w_k1_AXP_s013
                                 + 1.32317 w_k1_DIS_s013
                                 + 1.25952 w_k1_FDX_s013
                                 + 0.862139 w_k1_IBM_s013
                                 + 1.0483 w_k1_INTC_s013
                                 + 0.938256 w_k1_KO_s013
                                 + 0.821073 w_k1_MAT_s013
                                 + 1.19276 w_k1_MSFT_s013
                                 + 0.92543 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.944644 w_k1_ADM_s013
                                 + 0.780369 w_k1_AXP_s013
                                 + 0.733686 w_k1_DIS_s013
                                 + 0.919958 w_k1_FDX_s013
                                 + 0.7451 w_k1_IBM_s013
                                 + 0.789521 w_k1_INTC_s013
                                 + 0.664814 w_k1_KO_s013
                                 + 0.76974 w_k1_MAT_s013
                                 + 0.932169 w_k1_MSFT_s013
                                 + 0.785065 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.792429 w_k1_ADM_s013
                                 + 0.92748 w_k1_AXP_s013
                                 + 0.981494 w_k1_DIS_s013
                                 + 0.91792 w_k1_FDX_s013
                                 + 0.968325 w_k1_IBM_s013
                                 + 0.954381 w_k1_INTC_s013
                                 + 0.862794 w_k1_KO_s013
                                 + 0.72936 w_k1_MAT_s013
                                 + 0.88359 w_k1_MSFT_s013
                                 + 0.867346 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.13912 w_k1_ADM_s013
                                 + 1.19359 w_k1_AXP_s013
                                 + 0.987514 w_k1_DIS_s013
                                 + 1.09336 w_k1_FDX_s013
                                 + 0.872778 w_k1_IBM_s013
                                 + 1.2462 w_k1_INTC_s013
                                 + 0.934731 w_k1_KO_s013
                                 + 0.800974 w_k1_MAT_s013
                                 + 1.07432 w_k1_MSFT_s013
                                 + 0.994948 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.815211 w_k1_ADM_s013
                                 + 0.893319 w_k1_AXP_s013
                                 + 0.855462 w_k1_DIS_s013
                                 + 0.762143 w_k1_FDX_s013
                                 + 1.19553 w_k1_IBM_s013
                                 + 0.962575 w_k1_INTC_s013
                                 + 0.956968 w_k1_KO_s013
                                 + 0.870836 w_k1_MAT_s013
                                 + 0.804826 w_k1_MSFT_s013
                                 + 0.825044 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.21219 w_k1_ADM_s013
                                 + 1.31841 w_k1_AXP_s013
                                 + 1.15596 w_k1_DIS_s013
                                 + 1.08056 w_k1_FDX_s013
                                 + 1.42821 w_k1_IBM_s013
                                 + 0.861095 w_k1_INTC_s013
                                 + 1.14374 w_k1_KO_s013 + 1.45713 w_k1_MAT_s013
                                 + 1.01077 w_k1_MSFT_s013 + 1.0709 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.19061 w_k1_ADM_s014
                                 + 1.16941 w_k1_AXP_s014
                                 + 1.61803 w_k1_DIS_s014
                                 + 1.29414 w_k1_FDX_s014 + 1.5583 w_k1_IBM_s014
                                 + 1.24223 w_k1_INTC_s014
                                 + 1.46139 w_k1_KO_s014 + 1.66853 w_k1_MAT_s014
                                 + 1.30064 w_k1_MSFT_s014 + 1.132 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.672613 w_k1_ADM_s014
                                 + 0.845032 w_k1_AXP_s014
                                 + 0.819308 w_k1_DIS_s014
                                 + 0.947377 w_k1_FDX_s014
                                 + 0.63183 w_k1_IBM_s014
                                 + 1.14664 w_k1_INTC_s014
                                 + 0.715829 w_k1_KO_s014
                                 + 0.506714 w_k1_MAT_s014
                                 + 0.93465 w_k1_MSFT_s014
                                 + 0.800653 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.835382 w_k1_ADM_s014
                                 + 1.21024 w_k1_AXP_s014
                                 + 0.813777 w_k1_DIS_s014
                                 + 1.06111 w_k1_FDX_s014
                                 + 0.865006 w_k1_IBM_s014
                                 + 1.0922 w_k1_INTC_s014
                                 + 0.768569 w_k1_KO_s014
                                 + 0.924643 w_k1_MAT_s014
                                 + 1.02336 w_k1_MSFT_s014
                                 + 0.921945 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.00292 w_k1_ADM_s014
                                 + 1.40377 w_k1_AXP_s014
                                 + 1.11106 w_k1_DIS_s014 + 1.1743 w_k1_FDX_s014
                                 + 1.3706 w_k1_IBM_s014
                                 + 1.19791 w_k1_INTC_s014 + 1.0944 w_k1_KO_s014
                                 + 1.65381 w_k1_MAT_s014
                                 + 1.13934 w_k1_MSFT_s014
                                 + 1.04364 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.20317 w_k1_ADM_s014
                                 + 1.18426 w_k1_AXP_s014
                                 + 1.20539 w_k1_DIS_s014
                                 + 1.35776 w_k1_FDX_s014
                                 + 0.873301 w_k1_IBM_s014
                                 + 1.13794 w_k1_INTC_s014
                                 + 0.885116 w_k1_KO_s014
                                 + 0.775686 w_k1_MAT_s014
                                 + 1.32348 w_k1_MSFT_s014
                                 + 1.15044 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.904827 w_k1_ADM_s014
                                 + 0.901383 w_k1_AXP_s014
                                 + 1.01863 w_k1_DIS_s014
                                 + 0.947802 w_k1_FDX_s014
                                 + 1.04624 w_k1_IBM_s014
                                 + 1.16668 w_k1_INTC_s014
                                 + 0.887115 w_k1_KO_s014
                                 + 0.623596 w_k1_MAT_s014
                                 + 1.00451 w_k1_MSFT_s014
                                 + 0.925619 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.04422 w_k1_ADM_s014
                                 + 1.26637 w_k1_AXP_s014
                                 + 1.36114 w_k1_DIS_s014
                                 + 1.23853 w_k1_FDX_s014
                                 + 0.95981 w_k1_IBM_s014
                                 + 0.979034 w_k1_INTC_s014
                                 + 1.02195 w_k1_KO_s014 + 1.02048 w_k1_MAT_s014
                                 + 1.2284 w_k1_MSFT_s014 + 1.00055 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.756719 w_k1_ADM_s014
                                 + 0.811666 w_k1_AXP_s014
                                 + 0.635795 w_k1_DIS_s014
                                 + 0.794538 w_k1_FDX_s014
                                 + 0.953182 w_k1_IBM_s014
                                 + 0.930518 w_k1_INTC_s014
                                 + 0.731194 w_k1_KO_s014
                                 + 0.702456 w_k1_MAT_s014
                                 + 0.857722 w_k1_MSFT_s014
                                 + 0.809619 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.09694 w_k1_ADM_s015
                                 + 1.31831 w_k1_AXP_s015
                                 + 0.790996 w_k1_DIS_s015
                                 + 0.953203 w_k1_FDX_s015
                                 + 1.3361 w_k1_IBM_s015
                                 + 0.935784 w_k1_INTC_s015
                                 + 1.11261 w_k1_KO_s015 + 1.97667 w_k1_MAT_s015
                                 + 0.747016 w_k1_MSFT_s015
                                 + 1.04573 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.01012 w_k1_ADM_s015
                                 + 0.777821 w_k1_AXP_s015
                                 + 0.628614 w_k1_DIS_s015
                                 + 0.956003 w_k1_FDX_s015
                                 + 0.706046 w_k1_IBM_s015
                                 + 1.02462 w_k1_INTC_s015
                                 + 0.871973 w_k1_KO_s015
                                 + 0.911389 w_k1_MAT_s015
                                 + 0.847875 w_k1_MSFT_s015
                                 + 1.02923 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.981108 w_k1_ADM_s015
                                 + 1.01774 w_k1_AXP_s015
                                 + 1.09963 w_k1_DIS_s015
                                 + 1.06212 w_k1_FDX_s015
                                 + 1.33769 w_k1_IBM_s015
                                 + 0.990793 w_k1_INTC_s015
                                 + 1.25843 w_k1_KO_s015 + 1.33327 w_k1_MAT_s015
                                 + 0.826887 w_k1_MSFT_s015
                                 + 1.0969 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.856153 w_k1_ADM_s015
                                 + 0.716243 w_k1_AXP_s015
                                 + 0.614512 w_k1_DIS_s015
                                 + 0.717761 w_k1_FDX_s015
                                 + 0.9489 w_k1_IBM_s015
                                 + 0.732758 w_k1_INTC_s015
                                 + 0.899594 w_k1_KO_s015
                                 + 1.08196 w_k1_MAT_s015
                                 + 0.729955 w_k1_MSFT_s015
                                 + 0.897686 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.16393 w_k1_ADM_s015
                                 + 1.23176 w_k1_AXP_s015
                                 + 1.05422 w_k1_DIS_s015
                                 + 1.17375 w_k1_FDX_s015
                                 + 0.827127 w_k1_IBM_s015
                                 + 1.19125 w_k1_INTC_s015
                                 + 1.15746 w_k1_KO_s015 + 1.00313 w_k1_MAT_s015
                                 + 1.05756 w_k1_MSFT_s015
                                 + 1.28534 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.912136 w_k1_ADM_s015
                                 + 0.866881 w_k1_AXP_s015
                                 + 1.1503 w_k1_DIS_s015 + 1.02576 w_k1_FDX_s015
                                 + 0.909264 w_k1_IBM_s015
                                 + 0.916294 w_k1_INTC_s015
                                 + 1.17531 w_k1_KO_s015
                                 + 0.910387 w_k1_MAT_s015
                                 + 1.0213 w_k1_MSFT_s015 + 1.00438 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.34464 w_k1_ADM_s015
                                 + 1.89174 w_k1_AXP_s015
                                 + 1.79741 w_k1_DIS_s015
                                 + 1.47402 w_k1_FDX_s015
                                 + 1.62224 w_k1_IBM_s015
                                 + 1.35976 w_k1_INTC_s015
                                 + 1.84519 w_k1_KO_s015 + 2.84264 w_k1_MAT_s015
                                 + 1.25777 w_k1_MSFT_s015
                                 + 1.48217 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.767999 w_k1_ADM_s015
                                 + 0.680422 w_k1_AXP_s015
                                 + 0.642326 w_k1_DIS_s015
                                 + 0.755354 w_k1_FDX_s015
                                 + 0.646001 w_k1_IBM_s015
                                 + 0.803042 w_k1_INTC_s015
                                 + 0.762712 w_k1_KO_s015
                                 + 0.49429 w_k1_MAT_s015
                                 + 0.692027 w_k1_MSFT_s015
                                 + 0.816937 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.748761 w_k1_ADM_s016
                                 + 0.668504 w_k1_AXP_s016
                                 + 0.65643 w_k1_DIS_s016
                                 + 0.753496 w_k1_FDX_s016
                                 + 0.76565 w_k1_IBM_s016
                                 + 0.704253 w_k1_INTC_s016
                                 + 0.735467 w_k1_KO_s016
                                 + 0.655863 w_k1_MAT_s016
                                 + 0.781427 w_k1_MSFT_s016
                                 + 0.760991 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.672909 w_k1_ADM_s016
                                 + 0.655367 w_k1_AXP_s016
                                 + 0.793475 w_k1_DIS_s016
                                 + 0.844132 w_k1_FDX_s016
                                 + 0.878332 w_k1_IBM_s016
                                 + 0.78602 w_k1_INTC_s016
                                 + 0.814295 w_k1_KO_s016
                                 + 0.994245 w_k1_MAT_s016
                                 + 0.911021 w_k1_MSFT_s016
                                 + 0.942053 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.34767 w_k1_ADM_s016
                                 + 1.80903 w_k1_AXP_s016
                                 + 1.73693 w_k1_DIS_s016
                                 + 1.37665 w_k1_FDX_s016
                                 + 1.51652 w_k1_IBM_s016
                                 + 1.10861 w_k1_INTC_s016
                                 + 1.65862 w_k1_KO_s016 + 1.79103 w_k1_MAT_s016
                                 + 1.30291 w_k1_MSFT_s016
                                 + 1.44352 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.07018 w_k1_ADM_s016
                                 + 1.46355 w_k1_AXP_s016
                                 + 1.15787 w_k1_DIS_s016
                                 + 1.27012 w_k1_FDX_s016
                                 + 0.789698 w_k1_IBM_s016
                                 + 1.14352 w_k1_INTC_s016
                                 + 0.902636 w_k1_KO_s016
                                 + 0.932083 w_k1_MAT_s016
                                 + 1.1065 w_k1_MSFT_s016 + 1.02594 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.773998 w_k1_ADM_s016
                                 + 0.994088 w_k1_AXP_s016
                                 + 0.916372 w_k1_DIS_s016
                                 + 0.970161 w_k1_FDX_s016
                                 + 0.557637 w_k1_IBM_s016
                                 + 0.979036 w_k1_INTC_s016
                                 + 0.762991 w_k1_KO_s016
                                 + 0.583498 w_k1_MAT_s016
                                 + 1.02547 w_k1_MSFT_s016
                                 + 0.851366 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.13021 w_k1_ADM_s016
                                 + 1.38634 w_k1_AXP_s016
                                 + 1.72769 w_k1_DIS_s016
                                 + 1.38371 w_k1_FDX_s016
                                 + 0.971642 w_k1_IBM_s016
                                 + 1.20146 w_k1_INTC_s016
                                 + 1.24086 w_k1_KO_s016 + 1.20669 w_k1_MAT_s016
                                 + 1.35621 w_k1_MSFT_s016
                                 + 1.20637 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.10173 w_k1_ADM_s016
                                 + 1.04553 w_k1_AXP_s016 + 1.1915 w_k1_DIS_s016
                                 + 1.04694 w_k1_FDX_s016
                                 + 1.25299 w_k1_IBM_s016
                                 + 1.10463 w_k1_INTC_s016
                                 + 1.22783 w_k1_KO_s016 + 1.16585 w_k1_MAT_s016
                                 + 1.24612 w_k1_MSFT_s016 + 1.0393 w_k1_PG_s016
                                 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.825368 w_k1_ADM_s016
                                 + 1.1495 w_k1_AXP_s016
                                 + 0.826752 w_k1_DIS_s016
                                 + 0.89728 w_k1_FDX_s016
                                 + 1.23334 w_k1_IBM_s016
                                 + 0.815457 w_k1_INTC_s016
                                 + 1.10362 w_k1_KO_s016 + 1.67688 w_k1_MAT_s016
                                 + 0.990241 w_k1_MSFT_s016
                                 + 0.94677 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.821061 w_k1_ADM_s017
                                 + 0.91718 w_k1_AXP_s017
                                 + 0.579575 w_k1_DIS_s017
                                 + 0.723891 w_k1_FDX_s017
                                 + 0.934692 w_k1_IBM_s017
                                 + 0.799018 w_k1_INTC_s017
                                 + 0.804097 w_k1_KO_s017
                                 + 1.29053 w_k1_MAT_s017
                                 + 0.728665 w_k1_MSFT_s017
                                 + 0.907761 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.5275 w_k1_ADM_s017
                                 + 1.46022 w_k1_AXP_s017
                                 + 1.67135 w_k1_DIS_s017
                                 + 1.44224 w_k1_FDX_s017
                                 + 1.13891 w_k1_IBM_s017
                                 + 1.14937 w_k1_INTC_s017
                                 + 1.39722 w_k1_KO_s017 + 1.35743 w_k1_MAT_s017
                                 + 1.34341 w_k1_MSFT_s017 + 1.3175 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.15551 w_k1_ADM_s017
                                 + 1.53635 w_k1_AXP_s017
                                 + 1.77127 w_k1_DIS_s017
                                 + 1.29912 w_k1_FDX_s017
                                 + 1.39079 w_k1_IBM_s017
                                 + 1.16393 w_k1_INTC_s017
                                 + 1.34708 w_k1_KO_s017 + 1.47846 w_k1_MAT_s017
                                 + 1.15722 w_k1_MSFT_s017 + 1.2848 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.919073 w_k1_ADM_s017
                                 + 0.923458 w_k1_AXP_s017
                                 + 0.951477 w_k1_DIS_s017
                                 + 0.916461 w_k1_FDX_s017
                                 + 0.765791 w_k1_IBM_s017
                                 + 1.05239 w_k1_INTC_s017
                                 + 0.997387 w_k1_KO_s017
                                 + 0.899835 w_k1_MAT_s017
                                 + 0.93174 w_k1_MSFT_s017
                                 + 1.03857 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.900893 w_k1_ADM_s017
                                 + 1.05147 w_k1_AXP_s017
                                 + 1.01905 w_k1_DIS_s017
                                 + 0.872301 w_k1_FDX_s017
                                 + 1.19847 w_k1_IBM_s017
                                 + 1.04481 w_k1_INTC_s017
                                 + 1.25755 w_k1_KO_s017 + 1.16592 w_k1_MAT_s017
                                 + 0.947199 w_k1_MSFT_s017
                                 + 1.04875 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.17702 w_k1_ADM_s017
                                 + 1.19604 w_k1_AXP_s017
                                 + 1.25002 w_k1_DIS_s017
                                 + 1.18127 w_k1_FDX_s017
                                 + 0.991059 w_k1_IBM_s017
                                 + 0.990048 w_k1_INTC_s017
                                 + 1.10774 w_k1_KO_s017 + 1.06249 w_k1_MAT_s017
                                 + 1.05152 w_k1_MSFT_s017
                                 + 1.16454 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.14721 w_k1_ADM_s017
                                 + 1.21353 w_k1_AXP_s017
                                 + 1.26543 w_k1_DIS_s017
                                 + 1.11689 w_k1_FDX_s017
                                 + 1.58186 w_k1_IBM_s017
                                 + 1.14808 w_k1_INTC_s017
                                 + 1.46014 w_k1_KO_s017 + 1.84677 w_k1_MAT_s017
                                 + 1.06935 w_k1_MSFT_s017
                                 + 1.25634 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.03591 w_k1_ADM_s017
                                 + 1.26119 w_k1_AXP_s017
                                 + 0.88796 w_k1_DIS_s017
                                 + 1.08616 w_k1_FDX_s017
                                 + 0.823017 w_k1_IBM_s017
                                 + 0.840723 w_k1_INTC_s017
                                 + 0.895145 w_k1_KO_s017
                                 + 0.996987 w_k1_MAT_s017
                                 + 0.914359 w_k1_MSFT_s017
                                 + 1.10775 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.778885 w_k1_ADM_s100
                                 + 1.03062 w_k1_AXP_s100
                                 + 1.07163 w_k1_DIS_s100
                                 + 0.837747 w_k1_FDX_s100
                                 + 1.78845 w_k1_IBM_s100
                                 + 1.05294 w_k1_INTC_s100 + 1.25 w_k1_KO_s100
                                 + 1.61575 w_k1_MAT_s100
                                 + 1.13436 w_k1_MSFT_s100
                                 + 1.13946 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.00017 w_k1_ADM_s100
                                 + 1.02756 w_k1_AXP_s100
                                 + 1.13965 w_k1_DIS_s100
                                 + 1.40667 w_k1_FDX_s100
                                 + 0.586641 w_k1_IBM_s100
                                 + 0.984421 w_k1_INTC_s100
                                 + 0.927583 w_k1_KO_s100
                                 + 0.828946 w_k1_MAT_s100
                                 + 1.24799 w_k1_MSFT_s100
                                 + 1.08344 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.942902 w_k1_ADM_s100
                                 + 1.44199 w_k1_AXP_s100
                                 + 1.56145 w_k1_DIS_s100
                                 + 1.42248 w_k1_FDX_s100
                                 + 1.07439 w_k1_IBM_s100
                                 + 1.10807 w_k1_INTC_s100
                                 + 1.16223 w_k1_KO_s100 + 1.03686 w_k1_MAT_s100
                                 + 1.45267 w_k1_MSFT_s100
                                 + 1.14666 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.09096 w_k1_ADM_s100
                                 + 1.14788 w_k1_AXP_s100
                                 + 1.34384 w_k1_DIS_s100
                                 + 1.23868 w_k1_FDX_s100
                                 + 0.845496 w_k1_IBM_s100
                                 + 1.04843 w_k1_INTC_s100
                                 + 1.04499 w_k1_KO_s100
                                 + 0.750013 w_k1_MAT_s100
                                 + 1.26103 w_k1_MSFT_s100
                                 + 1.07527 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.92174 w_k1_ADM_s100
                                 + 1.08646 w_k1_AXP_s100 + 1.0606 w_k1_DIS_s100
                                 + 0.965201 w_k1_FDX_s100
                                 + 1.34767 w_k1_IBM_s100
                                 + 0.861062 w_k1_INTC_s100
                                 + 1.11531 w_k1_KO_s100 + 1.14641 w_k1_MAT_s100
                                 + 1.14951 w_k1_MSFT_s100
                                 + 1.04243 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.81973 w_k1_ADM_s100
                                 + 0.690452 w_k1_AXP_s100
                                 + 0.737898 w_k1_DIS_s100
                                 + 0.74085 w_k1_FDX_s100
                                 + 0.866534 w_k1_IBM_s100
                                 + 0.748316 w_k1_INTC_s100
                                 + 0.773521 w_k1_KO_s100
                                 + 0.466043 w_k1_MAT_s100
                                 + 0.938079 w_k1_MSFT_s100
                                 + 0.879438 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.849924 w_k1_ADM_s100
                                 + 1.04415 w_k1_AXP_s100
                                 + 0.970502 w_k1_DIS_s100
                                 + 1.03729 w_k1_FDX_s100
                                 + 0.988139 w_k1_IBM_s100
                                 + 0.834789 w_k1_INTC_s100
                                 + 1.09936 w_k1_KO_s100 + 1.05335 w_k1_MAT_s100
                                 + 1.06251 w_k1_MSFT_s100
                                 + 1.02306 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.768579 w_k1_ADM_s100
                                 + 0.794325 w_k1_AXP_s100
                                 + 0.758609 w_k1_DIS_s100
                                 + 0.995046 w_k1_FDX_s100
                                 + 0.636012 w_k1_IBM_s100
                                 + 0.75997 w_k1_INTC_s100 + 0.7145 w_k1_KO_s100
                                 + 0.702545 w_k1_MAT_s100
                                 + 0.914803 w_k1_MSFT_s100
                                 + 0.878115 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.10996 w_k1_ADM_s101
                                 + 1.18279 w_k1_AXP_s101
                                 + 1.48716 w_k1_DIS_s101
                                 + 1.16468 w_k1_FDX_s101
                                 + 1.17305 w_k1_IBM_s101
                                 + 1.17647 w_k1_INTC_s101
                                 + 1.26528 w_k1_KO_s101 + 1.16364 w_k1_MAT_s101
                                 + 1.19536 w_k1_MSFT_s101
                                 + 1.14742 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.10856 w_k1_ADM_s101
                                 + 1.41386 w_k1_AXP_s101
                                 + 1.00071 w_k1_DIS_s101
                                 + 1.09256 w_k1_FDX_s101
                                 + 1.03494 w_k1_IBM_s101
                                 + 1.14074 w_k1_INTC_s101
                                 + 0.996409 w_k1_KO_s101
                                 + 1.37226 w_k1_MAT_s101
                                 + 0.925646 w_k1_MSFT_s101
                                 + 1.09157 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.2679 w_k1_ADM_s101
                                 + 1.38924 w_k1_AXP_s101
                                 + 1.58927 w_k1_DIS_s101
                                 + 1.38068 w_k1_FDX_s101
                                 + 0.844774 w_k1_IBM_s101
                                 + 1.24873 w_k1_INTC_s101
                                 + 1.18674 w_k1_KO_s101
                                 + 0.958791 w_k1_MAT_s101
                                 + 1.26921 w_k1_MSFT_s101
                                 + 1.24614 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.919732 w_k1_ADM_s101
                                 + 0.898546 w_k1_AXP_s101
                                 + 1.11301 w_k1_DIS_s101
                                 + 0.997189 w_k1_FDX_s101
                                 + 0.942037 w_k1_IBM_s101
                                 + 1.13099 w_k1_INTC_s101
                                 + 0.953359 w_k1_KO_s101
                                 + 0.811803 w_k1_MAT_s101
                                 + 1.06428 w_k1_MSFT_s101
                                 + 0.997848 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.00317 w_k1_ADM_s101
                                 + 1.31319 w_k1_AXP_s101
                                 + 1.79014 w_k1_DIS_s101
                                 + 1.18984 w_k1_FDX_s101
                                 + 1.53649 w_k1_IBM_s101
                                 + 1.3393 w_k1_INTC_s101 + 1.48658 w_k1_KO_s101
                                 + 1.60069 w_k1_MAT_s101
                                 + 1.29059 w_k1_MSFT_s101
                                 + 1.24264 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.792542 w_k1_ADM_s101
                                 + 0.831859 w_k1_AXP_s101
                                 + 0.741592 w_k1_DIS_s101
                                 + 0.7916 w_k1_FDX_s101 + 0.85508 w_k1_IBM_s101
                                 + 0.775368 w_k1_INTC_s101
                                 + 0.929172 w_k1_KO_s101
                                 + 0.795948 w_k1_MAT_s101
                                 + 0.882606 w_k1_MSFT_s101
                                 + 0.931205 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.997147 w_k1_ADM_s101
                                 + 1.26263 w_k1_AXP_s101
                                 + 1.12281 w_k1_DIS_s101
                                 + 1.01679 w_k1_FDX_s101
                                 + 1.73557 w_k1_IBM_s101
                                 + 1.01849 w_k1_INTC_s101
                                 + 1.34411 w_k1_KO_s101 + 2.09577 w_k1_MAT_s101
                                 + 0.926944 w_k1_MSFT_s101
                                 + 1.11128 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.900158 w_k1_ADM_s101
                                 + 0.924608 w_k1_AXP_s101
                                 + 0.830994 w_k1_DIS_s101
                                 + 0.768245 w_k1_FDX_s101
                                 + 1.36178 w_k1_IBM_s101
                                 + 0.823636 w_k1_INTC_s101
                                 + 1.03094 w_k1_KO_s101 + 1.23221 w_k1_MAT_s101
                                 + 0.802842 w_k1_MSFT_s101
                                 + 0.882308 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.859266 w_k1_ADM_s102
                                 + 0.966568 w_k1_AXP_s102
                                 + 1.00214 w_k1_DIS_s102
                                 + 0.908337 w_k1_FDX_s102
                                 + 1.33668 w_k1_IBM_s102
                                 + 0.82233 w_k1_INTC_s102
                                 + 1.14173 w_k1_KO_s102 + 1.46195 w_k1_MAT_s102
                                 + 0.955446 w_k1_MSFT_s102
                                 + 0.948068 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.746929 w_k1_ADM_s102
                                 + 0.63843 w_k1_AXP_s102
                                 + 0.623013 w_k1_DIS_s102
                                 + 0.794851 w_k1_FDX_s102
                                 + 0.632025 w_k1_IBM_s102
                                 + 0.784695 w_k1_INTC_s102
                                 + 0.622914 w_k1_KO_s102
                                 + 0.444191 w_k1_MAT_s102
                                 + 0.812965 w_k1_MSFT_s102
                                 + 0.703534 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.06716 w_k1_ADM_s102
                                 + 0.954871 w_k1_AXP_s102
                                 + 0.833157 w_k1_DIS_s102
                                 + 0.999669 w_k1_FDX_s102
                                 + 1.00631 w_k1_IBM_s102
                                 + 0.71448 w_k1_INTC_s102
                                 + 0.842894 w_k1_KO_s102
                                 + 0.946778 w_k1_MAT_s102
                                 + 0.800855 w_k1_MSFT_s102
                                 + 0.820336 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.25273 w_k1_ADM_s102
                                 + 1.28647 w_k1_AXP_s102
                                 + 1.28911 w_k1_DIS_s102
                                 + 1.18336 w_k1_FDX_s102
                                 + 0.997239 w_k1_IBM_s102
                                 + 0.861884 w_k1_INTC_s102
                                 + 1.01764 w_k1_KO_s102 + 1.04115 w_k1_MAT_s102
                                 + 1.1236 w_k1_MSFT_s102
                                 + 0.990211 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.02243 w_k1_ADM_s102
                                 + 0.872296 w_k1_AXP_s102
                                 + 0.678093 w_k1_DIS_s102
                                 + 0.759096 w_k1_FDX_s102
                                 + 1.00395 w_k1_IBM_s102
                                 + 0.704141 w_k1_INTC_s102
                                 + 0.808101 w_k1_KO_s102
                                 + 0.864863 w_k1_MAT_s102
                                 + 0.723558 w_k1_MSFT_s102
                                 + 0.781046 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.92785 w_k1_ADM_s102
                                 + 0.944803 w_k1_AXP_s102
                                 + 1.02021 w_k1_DIS_s102
                                 + 0.943504 w_k1_FDX_s102
                                 + 0.837984 w_k1_IBM_s102
                                 + 1.14056 w_k1_INTC_s102
                                 + 0.927616 w_k1_KO_s102
                                 + 0.537787 w_k1_MAT_s102
                                 + 1.11392 w_k1_MSFT_s102
                                 + 0.917028 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.40425 w_k1_ADM_s102
                                 + 1.68103 w_k1_AXP_s102
                                 + 1.70892 w_k1_DIS_s102 + 1.4189 w_k1_FDX_s102
                                 + 1.33304 w_k1_IBM_s102
                                 + 0.953878 w_k1_INTC_s102
                                 + 1.23165 w_k1_KO_s102 + 1.29791 w_k1_MAT_s102
                                 + 1.26687 w_k1_MSFT_s102 + 1.0715 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.97465 w_k1_ADM_s102
                                 + 0.834257 w_k1_AXP_s102
                                 + 1.17361 w_k1_DIS_s102
                                 + 0.999534 w_k1_FDX_s102
                                 + 0.893202 w_k1_IBM_s102
                                 + 1.14904 w_k1_INTC_s102
                                 + 0.957265 w_k1_KO_s102
                                 + 0.480809 w_k1_MAT_s102
                                 + 1.11009 w_k1_MSFT_s102
                                 + 0.840577 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.851199 w_k1_ADM_s103
                                 + 1.19385 w_k1_AXP_s103
                                 + 1.07976 w_k1_DIS_s103
                                 + 0.993434 w_k1_FDX_s103
                                 + 1.39408 w_k1_IBM_s103
                                 + 1.38895 w_k1_INTC_s103
                                 + 1.33812 w_k1_KO_s103 + 1.74858 w_k1_MAT_s103
                                 + 1.1284 w_k1_MSFT_s103 + 1.07261 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.850015 w_k1_ADM_s103
                                 + 1.01947 w_k1_AXP_s103
                                 + 0.957956 w_k1_DIS_s103
                                 + 1.1076 w_k1_FDX_s103
                                 + 0.696763 w_k1_IBM_s103
                                 + 1.13097 w_k1_INTC_s103
                                 + 0.817464 w_k1_KO_s103
                                 + 0.82133 w_k1_MAT_s103
                                 + 0.949033 w_k1_MSFT_s103
                                 + 0.951961 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05099 w_k1_ADM_s103
                                 + 1.3574 w_k1_AXP_s103 + 1.26857 w_k1_DIS_s103
                                 + 1.29346 w_k1_FDX_s103
                                 + 1.08598 w_k1_IBM_s103
                                 + 1.63297 w_k1_INTC_s103 + 1.276 w_k1_KO_s103
                                 + 1.10289 w_k1_MAT_s103
                                 + 1.34174 w_k1_MSFT_s103
                                 + 1.19141 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.82291 w_k1_ADM_s103
                                 + 0.769358 w_k1_AXP_s103
                                 + 0.994044 w_k1_DIS_s103
                                 + 1.04308 w_k1_FDX_s103
                                 + 1.02291 w_k1_IBM_s103
                                 + 1.21716 w_k1_INTC_s103
                                 + 1.11091 w_k1_KO_s103 + 0.90849 w_k1_MAT_s103
                                 + 1.00545 w_k1_MSFT_s103
                                 + 0.995481 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.770683 w_k1_ADM_s103
                                 + 0.797738 w_k1_AXP_s103
                                 + 0.546232 w_k1_DIS_s103
                                 + 0.846254 w_k1_FDX_s103
                                 + 0.759439 w_k1_IBM_s103
                                 + 1.04583 w_k1_INTC_s103
                                 + 0.716466 w_k1_KO_s103
                                 + 0.879359 w_k1_MAT_s103
                                 + 0.828644 w_k1_MSFT_s103
                                 + 0.875423 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.755488 w_k1_ADM_s103
                                 + 0.785713 w_k1_AXP_s103
                                 + 0.561286 w_k1_DIS_s103
                                 + 0.742544 w_k1_FDX_s103
                                 + 0.864238 w_k1_IBM_s103
                                 + 0.905088 w_k1_INTC_s103
                                 + 0.713361 w_k1_KO_s103
                                 + 0.801848 w_k1_MAT_s103
                                 + 0.655585 w_k1_MSFT_s103
                                 + 0.803503 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.772047 w_k1_ADM_s103
                                 + 0.730816 w_k1_AXP_s103
                                 + 1.06201 w_k1_DIS_s103
                                 + 0.912014 w_k1_FDX_s103
                                 + 0.830817 w_k1_IBM_s103
                                 + 1.18895 w_k1_INTC_s103
                                 + 1.01774 w_k1_KO_s103 + 0.91114 w_k1_MAT_s103
                                 + 0.973758 w_k1_MSFT_s103
                                 + 0.9559 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.875232 w_k1_ADM_s103
                                 + 0.809664 w_k1_AXP_s103
                                 + 0.912138 w_k1_DIS_s103
                                 + 0.890098 w_k1_FDX_s103
                                 + 1.17452 w_k1_IBM_s103
                                 + 1.21645 w_k1_INTC_s103
                                 + 1.04256 w_k1_KO_s103
                                 + 0.939148 w_k1_MAT_s103
                                 + 0.946069 w_k1_MSFT_s103
                                 + 0.923273 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.10768 w_k1_ADM_s104
                                 + 1.15503 w_k1_AXP_s104
                                 + 1.18916 w_k1_DIS_s104 + 1.2595 w_k1_FDX_s104
                                 + 1.01029 w_k1_IBM_s104
                                 + 1.09055 w_k1_INTC_s104
                                 + 1.02916 w_k1_KO_s104
                                 + 0.943844 w_k1_MAT_s104
                                 + 1.0859 w_k1_MSFT_s104 + 1.18982 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.13558 w_k1_ADM_s104
                                 + 1.12455 w_k1_AXP_s104
                                 + 0.917812 w_k1_DIS_s104
                                 + 1.03346 w_k1_FDX_s104
                                 + 1.56065 w_k1_IBM_s104
                                 + 1.10288 w_k1_INTC_s104
                                 + 1.13653 w_k1_KO_s104 + 1.58757 w_k1_MAT_s104
                                 + 0.958972 w_k1_MSFT_s104
                                 + 1.21004 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.32842 w_k1_ADM_s104
                                 + 1.62985 w_k1_AXP_s104
                                 + 1.70611 w_k1_DIS_s104 + 1.591 w_k1_FDX_s104
                                 + 1.38511 w_k1_IBM_s104
                                 + 1.37668 w_k1_INTC_s104
                                 + 1.40335 w_k1_KO_s104 + 1.52272 w_k1_MAT_s104
                                 + 1.406 w_k1_MSFT_s104 + 1.42701 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.09254 w_k1_ADM_s104
                                 + 0.934254 w_k1_AXP_s104
                                 + 1.25306 w_k1_DIS_s104 + 1.1126 w_k1_FDX_s104
                                 + 1.31209 w_k1_IBM_s104
                                 + 1.12579 w_k1_INTC_s104 + 1.1885 w_k1_KO_s104
                                 + 0.991395 w_k1_MAT_s104
                                 + 1.13178 w_k1_MSFT_s104
                                 + 1.10903 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.2302 w_k1_ADM_s104
                                 + 1.0527 w_k1_AXP_s104 + 1.07491 w_k1_DIS_s104
                                 + 1.2271 w_k1_FDX_s104
                                 + 0.788819 w_k1_IBM_s104
                                 + 1.24434 w_k1_INTC_s104
                                 + 0.933729 w_k1_KO_s104
                                 + 0.816466 w_k1_MAT_s104
                                 + 1.11217 w_k1_MSFT_s104
                                 + 1.12518 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.944224 w_k1_ADM_s104
                                 + 1.10082 w_k1_AXP_s104
                                 + 0.928093 w_k1_DIS_s104
                                 + 0.98802 w_k1_FDX_s104
                                 + 1.06721 w_k1_IBM_s104
                                 + 0.95747 w_k1_INTC_s104
                                 + 0.836455 w_k1_KO_s104
                                 + 1.09246 w_k1_MAT_s104
                                 + 0.850139 w_k1_MSFT_s104
                                 + 1.05398 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.96798 w_k1_ADM_s104
                                 + 0.794708 w_k1_AXP_s104
                                 + 0.78342 w_k1_DIS_s104
                                 + 0.927227 w_k1_FDX_s104
                                 + 0.883372 w_k1_IBM_s104
                                 + 1.03431 w_k1_INTC_s104
                                 + 0.780219 w_k1_KO_s104
                                 + 0.615719 w_k1_MAT_s104
                                 + 0.814661 w_k1_MSFT_s104
                                 + 0.875745 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.710774 w_k1_ADM_s104
                                 + 0.607333 w_k1_AXP_s104
                                 + 0.557719 w_k1_DIS_s104
                                 + 0.696148 w_k1_FDX_s104
                                 + 0.852341 w_k1_IBM_s104
                                 + 0.724589 w_k1_INTC_s104
                                 + 0.698385 w_k1_KO_s104
                                 + 0.938524 w_k1_MAT_s104
                                 + 0.687089 w_k1_MSFT_s104
                                 + 0.882596 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.29575 w_k1_ADM_s105
                                 + 1.06866 w_k1_AXP_s105
                                 + 1.14631 w_k1_DIS_s105
                                 + 1.22116 w_k1_FDX_s105
                                 + 0.899403 w_k1_IBM_s105
                                 + 1.06843 w_k1_INTC_s105 + 1.0861 w_k1_KO_s105
                                 + 0.662787 w_k1_MAT_s105
                                 + 1.23201 w_k1_MSFT_s105
                                 + 1.07428 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.15454 w_k1_ADM_s105
                                 + 1.23888 w_k1_AXP_s105
                                 + 1.35156 w_k1_DIS_s105
                                 + 1.15542 w_k1_FDX_s105
                                 + 1.29785 w_k1_IBM_s105
                                 + 1.13335 w_k1_INTC_s105
                                 + 1.12016 w_k1_KO_s105 + 1.21596 w_k1_MAT_s105
                                 + 1.16101 w_k1_MSFT_s105 + 1.0324 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.963754 w_k1_ADM_s105
                                 + 0.723115 w_k1_AXP_s105
                                 + 0.992786 w_k1_DIS_s105
                                 + 0.983761 w_k1_FDX_s105
                                 + 1.00163 w_k1_IBM_s105
                                 + 1.04295 w_k1_INTC_s105
                                 + 1.01883 w_k1_KO_s105
                                 + 0.726405 w_k1_MAT_s105
                                 + 1.08913 w_k1_MSFT_s105
                                 + 1.01816 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.855142 w_k1_ADM_s105
                                 + 1.04223 w_k1_AXP_s105
                                 + 0.63637 w_k1_DIS_s105
                                 + 0.84032 w_k1_FDX_s105
                                 + 0.749297 w_k1_IBM_s105
                                 + 0.889741 w_k1_INTC_s105
                                 + 0.674798 w_k1_KO_s105
                                 + 0.864625 w_k1_MAT_s105
                                 + 0.808748 w_k1_MSFT_s105
                                 + 0.879776 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.903078 w_k1_ADM_s105
                                 + 0.658161 w_k1_AXP_s105
                                 + 0.783589 w_k1_DIS_s105
                                 + 0.824256 w_k1_FDX_s105
                                 + 0.826587 w_k1_IBM_s105
                                 + 0.847892 w_k1_INTC_s105
                                 + 0.800974 w_k1_KO_s105
                                 + 0.651558 w_k1_MAT_s105
                                 + 0.911304 w_k1_MSFT_s105
                                 + 0.914793 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.1047 w_k1_ADM_s105
                                 + 1.12869 w_k1_AXP_s105
                                 + 0.800516 w_k1_DIS_s105
                                 + 0.929918 w_k1_FDX_s105
                                 + 0.993792 w_k1_IBM_s105
                                 + 0.943251 w_k1_INTC_s105
                                 + 0.921911 w_k1_KO_s105
                                 + 1.12642 w_k1_MAT_s105
                                 + 1.00838 w_k1_MSFT_s105 + 1.0612 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.676465 w_k1_ADM_s105
                                 + 0.696962 w_k1_AXP_s105
                                 + 0.456049 w_k1_DIS_s105
                                 + 0.6621 w_k1_FDX_s105
                                 + 0.803709 w_k1_IBM_s105
                                 + 0.829912 w_k1_INTC_s105
                                 + 0.590381 w_k1_KO_s105
                                 + 0.508733 w_k1_MAT_s105
                                 + 0.677184 w_k1_MSFT_s105
                                 + 0.689298 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.941905 w_k1_ADM_s105
                                 + 1.0707 w_k1_AXP_s105
                                 + 0.938254 w_k1_DIS_s105
                                 + 0.909703 w_k1_FDX_s105
                                 + 1.44353 w_k1_IBM_s105
                                 + 1.18878 w_k1_INTC_s105
                                 + 1.30418 w_k1_KO_s105 + 1.37779 w_k1_MAT_s105
                                 + 1.00771 w_k1_MSFT_s105
                                 + 1.02437 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.958842 w_k1_ADM_s106
                                 + 0.938348 w_k1_AXP_s106
                                 + 0.973933 w_k1_DIS_s106
                                 + 0.963754 w_k1_FDX_s106
                                 + 0.869929 w_k1_IBM_s106
                                 + 0.791651 w_k1_INTC_s106
                                 + 0.913556 w_k1_KO_s106
                                 + 0.870747 w_k1_MAT_s106
                                 + 0.853784 w_k1_MSFT_s106
                                 + 0.83585 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.06483 w_k1_ADM_s106
                                 + 1.22343 w_k1_AXP_s106
                                 + 1.18448 w_k1_DIS_s106
                                 + 1.27895 w_k1_FDX_s106
                                 + 0.90116 w_k1_IBM_s106
                                 + 1.11756 w_k1_INTC_s106
                                 + 1.23112 w_k1_KO_s106 + 1.38997 w_k1_MAT_s106
                                 + 1.21845 w_k1_MSFT_s106
                                 + 0.977763 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.56597 w_k1_ADM_s106
                                 + 0.56682 w_k1_AXP_s106
                                 + 0.554235 w_k1_DIS_s106
                                 + 0.647757 w_k1_FDX_s106
                                 + 0.97418 w_k1_IBM_s106
                                 + 0.777299 w_k1_INTC_s106
                                 + 0.821953 w_k1_KO_s106
                                 + 0.836627 w_k1_MAT_s106
                                 + 0.651784 w_k1_MSFT_s106
                                 + 0.690539 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.994049 w_k1_ADM_s106
                                 + 1.15465 w_k1_AXP_s106
                                 + 1.03397 w_k1_DIS_s106
                                 + 1.00484 w_k1_FDX_s106
                                 + 1.25888 w_k1_IBM_s106
                                 + 0.838376 w_k1_INTC_s106
                                 + 1.05223 w_k1_KO_s106 + 1.24685 w_k1_MAT_s106
                                 + 0.874338 w_k1_MSFT_s106
                                 + 1.02777 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.803974 w_k1_ADM_s106
                                 + 0.683977 w_k1_AXP_s106
                                 + 0.587854 w_k1_DIS_s106
                                 + 0.793449 w_k1_FDX_s106
                                 + 0.702885 w_k1_IBM_s106
                                 + 0.846744 w_k1_INTC_s106
                                 + 0.787737 w_k1_KO_s106
                                 + 0.672633 w_k1_MAT_s106
                                 + 0.791906 w_k1_MSFT_s106
                                 + 0.807876 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.34354 w_k1_ADM_s106
                                 + 1.535 w_k1_AXP_s106 + 1.67925 w_k1_DIS_s106
                                 + 1.47702 w_k1_FDX_s106 + 1.1127 w_k1_IBM_s106
                                 + 1.20931 w_k1_INTC_s106 + 1.4081 w_k1_KO_s106
                                 + 1.5646 w_k1_MAT_s106
                                 + 1.38825 w_k1_MSFT_s106
                                 + 1.22672 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.840701 w_k1_ADM_s106
                                 + 0.804199 w_k1_AXP_s106
                                 + 0.779104 w_k1_DIS_s106
                                 + 0.709289 w_k1_FDX_s106
                                 + 1.23531 w_k1_IBM_s106
                                 + 0.724787 w_k1_INTC_s106
                                 + 1.09865 w_k1_KO_s106 + 1.06326 w_k1_MAT_s106
                                 + 0.790232 w_k1_MSFT_s106
                                 + 0.741006 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.17085 w_k1_ADM_s106
                                 + 1.2825 w_k1_AXP_s106 + 1.34958 w_k1_DIS_s106
                                 + 1.0132 w_k1_FDX_s106 + 1.67497 w_k1_IBM_s106
                                 + 1.06457 w_k1_INTC_s106
                                 + 1.71585 w_k1_KO_s106 + 1.72863 w_k1_MAT_s106
                                 + 1.13166 w_k1_MSFT_s106 + 1.121 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.2671 w_k1_ADM_s107
                                 + 1.30026 w_k1_AXP_s107
                                 + 1.23028 w_k1_DIS_s107
                                 + 1.24319 w_k1_FDX_s107
                                 + 0.908407 w_k1_IBM_s107
                                 + 1.2769 w_k1_INTC_s107 + 1.14759 w_k1_KO_s107
                                 + 0.941588 w_k1_MAT_s107
                                 + 1.15483 w_k1_MSFT_s107
                                 + 1.10682 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.03287 w_k1_ADM_s107
                                 + 1.32095 w_k1_AXP_s107
                                 + 0.996968 w_k1_DIS_s107
                                 + 0.963772 w_k1_FDX_s107
                                 + 1.59517 w_k1_IBM_s107
                                 + 1.15063 w_k1_INTC_s107
                                 + 1.14812 w_k1_KO_s107 + 1.7371 w_k1_MAT_s107
                                 + 0.917971 w_k1_MSFT_s107
                                 + 1.10347 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.48726 w_k1_ADM_s107
                                 + 1.71656 w_k1_AXP_s107
                                 + 1.81431 w_k1_DIS_s107
                                 + 1.51059 w_k1_FDX_s107
                                 + 1.23436 w_k1_IBM_s107
                                 + 1.39395 w_k1_INTC_s107
                                 + 1.39162 w_k1_KO_s107 + 1.01801 w_k1_MAT_s107
                                 + 1.3143 w_k1_MSFT_s107 + 1.22255 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.81664 w_k1_ADM_s107
                                 + 0.814948 w_k1_AXP_s107
                                 + 0.704888 w_k1_DIS_s107
                                 + 0.801505 w_k1_FDX_s107
                                 + 0.903892 w_k1_IBM_s107
                                 + 0.93105 w_k1_INTC_s107
                                 + 0.841495 w_k1_KO_s107
                                 + 0.754732 w_k1_MAT_s107
                                 + 0.721913 w_k1_MSFT_s107
                                 + 0.840668 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.9365 w_k1_ADM_s107
                                 + 0.896986 w_k1_AXP_s107
                                 + 0.908071 w_k1_DIS_s107
                                 + 0.923708 w_k1_FDX_s107
                                 + 1.21158 w_k1_IBM_s107
                                 + 1.08969 w_k1_INTC_s107
                                 + 1.04042 w_k1_KO_s107
                                 + 0.932359 w_k1_MAT_s107
                                 + 0.991451 w_k1_MSFT_s107
                                 + 0.938376 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.04085 w_k1_ADM_s107
                                 + 1.08482 w_k1_AXP_s107
                                 + 0.920254 w_k1_DIS_s107
                                 + 1.02957 w_k1_FDX_s107
                                 + 0.919014 w_k1_IBM_s107
                                 + 1.00986 w_k1_INTC_s107
                                 + 0.91134 w_k1_KO_s107
                                 + 0.901569 w_k1_MAT_s107
                                 + 0.990721 w_k1_MSFT_s107
                                 + 0.999706 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.11287 w_k1_ADM_s107
                                 + 1.30737 w_k1_AXP_s107
                                 + 1.49611 w_k1_DIS_s107
                                 + 1.19267 w_k1_FDX_s107 + 1.4798 w_k1_IBM_s107
                                 + 1.39675 w_k1_INTC_s107
                                 + 1.46118 w_k1_KO_s107 + 1.40906 w_k1_MAT_s107
                                 + 1.25228 w_k1_MSFT_s107
                                 + 1.22626 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.14175 w_k1_ADM_s107
                                 + 1.19 w_k1_AXP_s107 + 1.17728 w_k1_DIS_s107
                                 + 1.04484 w_k1_FDX_s107
                                 + 1.18524 w_k1_IBM_s107
                                 + 1.11019 w_k1_INTC_s107
                                 + 1.12423 w_k1_KO_s107
                                 + 0.963068 w_k1_MAT_s107
                                 + 0.917851 w_k1_MSFT_s107
                                 + 1.05158 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.713718 w_k1_ADM_s110
                                 + 0.620661 w_k1_AXP_s110
                                 + 0.657042 w_k1_DIS_s110
                                 + 0.835194 w_k1_FDX_s110
                                 + 0.567386 w_k1_IBM_s110
                                 + 0.834146 w_k1_INTC_s110
                                 + 0.687503 w_k1_KO_s110
                                 + 0.490313 w_k1_MAT_s110
                                 + 0.771862 w_k1_MSFT_s110
                                 + 0.937359 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.686137 w_k1_ADM_s110
                                 + 0.664542 w_k1_AXP_s110
                                 + 0.763802 w_k1_DIS_s110
                                 + 0.804198 w_k1_FDX_s110
                                 + 0.997846 w_k1_IBM_s110
                                 + 0.874821 w_k1_INTC_s110
                                 + 0.88105 w_k1_KO_s110
                                 + 0.782616 w_k1_MAT_s110
                                 + 0.812826 w_k1_MSFT_s110
                                 + 0.875705 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.811492 w_k1_ADM_s110
                                 + 0.693091 w_k1_AXP_s110
                                 + 0.831708 w_k1_DIS_s110
                                 + 0.928892 w_k1_FDX_s110
                                 + 1.22223 w_k1_IBM_s110
                                 + 0.922965 w_k1_INTC_s110
                                 + 0.946238 w_k1_KO_s110
                                 + 0.943127 w_k1_MAT_s110
                                 + 0.887228 w_k1_MSFT_s110
                                 + 0.951575 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.02055 w_k1_ADM_s110
                                 + 1.10188 w_k1_AXP_s110
                                 + 1.13386 w_k1_DIS_s110
                                 + 1.29072 w_k1_FDX_s110
                                 + 0.659804 w_k1_IBM_s110
                                 + 1.18894 w_k1_INTC_s110
                                 + 0.806957 w_k1_KO_s110
                                 + 0.686198 w_k1_MAT_s110
                                 + 1.08606 w_k1_MSFT_s110
                                 + 0.974431 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.941246 w_k1_ADM_s110
                                 + 1.32072 w_k1_AXP_s110 + 1.6243 w_k1_DIS_s110
                                 + 1.5161 w_k1_FDX_s110 + 1.05471 w_k1_IBM_s110
                                 + 1.45376 w_k1_INTC_s110
                                 + 1.14486 w_k1_KO_s110 + 1.03534 w_k1_MAT_s110
                                 + 1.25207 w_k1_MSFT_s110
                                 + 1.25366 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.933802 w_k1_ADM_s110
                                 + 0.814271 w_k1_AXP_s110
                                 + 0.766953 w_k1_DIS_s110
                                 + 0.919527 w_k1_FDX_s110
                                 + 0.715134 w_k1_IBM_s110
                                 + 1.00585 w_k1_INTC_s110
                                 + 0.761537 w_k1_KO_s110
                                 + 0.755995 w_k1_MAT_s110
                                 + 0.839531 w_k1_MSFT_s110
                                 + 1.00311 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.854396 w_k1_ADM_s110
                                 + 0.796133 w_k1_AXP_s110
                                 + 1.11105 w_k1_DIS_s110
                                 + 0.992978 w_k1_FDX_s110
                                 + 0.908984 w_k1_IBM_s110
                                 + 1.02897 w_k1_INTC_s110
                                 + 1.00591 w_k1_KO_s110
                                 + 0.755139 w_k1_MAT_s110
                                 + 1.04804 w_k1_MSFT_s110
                                 + 0.906908 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.05645 w_k1_ADM_s110
                                 + 1.09003 w_k1_AXP_s110
                                 + 1.27388 w_k1_DIS_s110
                                 + 1.07369 w_k1_FDX_s110
                                 + 1.51517 w_k1_IBM_s110
                                 + 1.04192 w_k1_INTC_s110 + 1.3667 w_k1_KO_s110
                                 + 1.37076 w_k1_MAT_s110
                                 + 0.945878 w_k1_MSFT_s110
                                 + 1.13793 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.04979 w_k1_ADM_s111
                                 + 1.0483 w_k1_AXP_s111 + 1.1341 w_k1_DIS_s111
                                 + 1.02336 w_k1_FDX_s111
                                 + 1.11553 w_k1_IBM_s111
                                 + 1.19958 w_k1_INTC_s111
                                 + 1.17772 w_k1_KO_s111 + 0.98114 w_k1_MAT_s111
                                 + 1.05296 w_k1_MSFT_s111
                                 + 1.03588 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.788906 w_k1_ADM_s111
                                 + 0.58477 w_k1_AXP_s111
                                 + 0.795123 w_k1_DIS_s111
                                 + 0.761957 w_k1_FDX_s111
                                 + 0.804326 w_k1_IBM_s111
                                 + 0.825472 w_k1_INTC_s111
                                 + 0.797343 w_k1_KO_s111
                                 + 0.651042 w_k1_MAT_s111
                                 + 0.955392 w_k1_MSFT_s111
                                 + 0.87651 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.10118 w_k1_ADM_s111
                                 + 1.14093 w_k1_AXP_s111 + 1.0996 w_k1_DIS_s111
                                 + 1.14601 w_k1_FDX_s111
                                 + 0.922424 w_k1_IBM_s111
                                 + 1.08587 w_k1_INTC_s111
                                 + 0.939741 w_k1_KO_s111
                                 + 0.814006 w_k1_MAT_s111
                                 + 0.970001 w_k1_MSFT_s111
                                 + 1.01034 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.07235 w_k1_ADM_s111
                                 + 0.785576 w_k1_AXP_s111
                                 + 0.830103 w_k1_DIS_s111
                                 + 0.848128 w_k1_FDX_s111
                                 + 0.857388 w_k1_IBM_s111
                                 + 1.02848 w_k1_INTC_s111
                                 + 0.868664 w_k1_KO_s111
                                 + 0.646685 w_k1_MAT_s111
                                 + 0.934973 w_k1_MSFT_s111
                                 + 0.972833 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.29032 w_k1_ADM_s111
                                 + 1.48077 w_k1_AXP_s111
                                 + 1.59912 w_k1_DIS_s111
                                 + 1.28912 w_k1_FDX_s111 + 1.3601 w_k1_IBM_s111
                                 + 1.55078 w_k1_INTC_s111
                                 + 1.45639 w_k1_KO_s111 + 1.52576 w_k1_MAT_s111
                                 + 1.44825 w_k1_MSFT_s111
                                 + 1.35467 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.87721 w_k1_ADM_s111
                                 + 0.842697 w_k1_AXP_s111
                                 + 0.709143 w_k1_DIS_s111
                                 + 0.846061 w_k1_FDX_s111
                                 + 1.16241 w_k1_IBM_s111
                                 + 0.941426 w_k1_INTC_s111
                                 + 0.862486 w_k1_KO_s111
                                 + 0.97795 w_k1_MAT_s111
                                 + 0.83822 w_k1_MSFT_s111
                                 + 0.937299 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.973698 w_k1_ADM_s111
                                 + 0.87836 w_k1_AXP_s111
                                 + 1.22118 w_k1_DIS_s111
                                 + 0.86268 w_k1_FDX_s111
                                 + 1.40922 w_k1_IBM_s111
                                 + 0.884224 w_k1_INTC_s111
                                 + 1.12323 w_k1_KO_s111 + 1.23764 w_k1_MAT_s111
                                 + 0.984393 w_k1_MSFT_s111
                                 + 1.14188 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.01252 w_k1_ADM_s111
                                 + 0.898332 w_k1_AXP_s111
                                 + 0.993128 w_k1_DIS_s111
                                 + 1.07838 w_k1_FDX_s111
                                 + 0.860616 w_k1_IBM_s111
                                 + 1.16445 w_k1_INTC_s111
                                 + 0.904536 w_k1_KO_s111
                                 + 0.860662 w_k1_MAT_s111
                                 + 1.14437 w_k1_MSFT_s111
                                 + 1.11681 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.953381 w_k1_ADM_s112
                                 + 1.18032 w_k1_AXP_s112
                                 + 1.35145 w_k1_DIS_s112
                                 + 1.11466 w_k1_FDX_s112
                                 + 1.52654 w_k1_IBM_s112
                                 + 1.04876 w_k1_INTC_s112 + 1.1054 w_k1_KO_s112
                                 + 1.78175 w_k1_MAT_s112
                                 + 1.31849 w_k1_MSFT_s112
                                 + 1.01307 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.799815 w_k1_ADM_s112
                                 + 0.77879 w_k1_AXP_s112
                                 + 0.643862 w_k1_DIS_s112
                                 + 0.746511 w_k1_FDX_s112
                                 + 0.809533 w_k1_IBM_s112
                                 + 0.619344 w_k1_INTC_s112
                                 + 0.65057 w_k1_KO_s112
                                 + 0.874483 w_k1_MAT_s112
                                 + 0.746323 w_k1_MSFT_s112
                                 + 0.872746 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.979264 w_k1_ADM_s112
                                 + 0.975418 w_k1_AXP_s112
                                 + 0.871323 w_k1_DIS_s112
                                 + 0.871135 w_k1_FDX_s112
                                 + 1.12606 w_k1_IBM_s112
                                 + 0.792079 w_k1_INTC_s112
                                 + 0.897159 w_k1_KO_s112
                                 + 0.956658 w_k1_MAT_s112
                                 + 0.90458 w_k1_MSFT_s112
                                 + 1.08514 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.772627 w_k1_ADM_s112
                                 + 0.938789 w_k1_AXP_s112
                                 + 0.698437 w_k1_DIS_s112
                                 + 1.00583 w_k1_FDX_s112
                                 + 0.634553 w_k1_IBM_s112
                                 + 0.886574 w_k1_INTC_s112
                                 + 0.697142 w_k1_KO_s112
                                 + 0.999221 w_k1_MAT_s112
                                 + 0.979321 w_k1_MSFT_s112
                                 + 0.918135 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.07459 w_k1_ADM_s112
                                 + 1.18589 w_k1_AXP_s112
                                 + 0.847208 w_k1_DIS_s112
                                 + 0.997471 w_k1_FDX_s112 + 1.145 w_k1_IBM_s112
                                 + 0.963668 w_k1_INTC_s112
                                 + 0.867587 w_k1_KO_s112
                                 + 1.05122 w_k1_MAT_s112
                                 + 0.972618 w_k1_MSFT_s112
                                 + 0.971633 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.03379 w_k1_ADM_s112
                                 + 0.886221 w_k1_AXP_s112
                                 + 1.01225 w_k1_DIS_s112
                                 + 1.00076 w_k1_FDX_s112
                                 + 0.885441 w_k1_IBM_s112
                                 + 0.786835 w_k1_INTC_s112
                                 + 0.757662 w_k1_KO_s112
                                 + 0.852562 w_k1_MAT_s112
                                 + 1.00282 w_k1_MSFT_s112
                                 + 0.994027 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.13929 w_k1_ADM_s112
                                 + 1.37758 w_k1_AXP_s112
                                 + 1.28287 w_k1_DIS_s112
                                 + 1.30467 w_k1_FDX_s112
                                 + 1.05629 w_k1_IBM_s112
                                 + 1.19959 w_k1_INTC_s112
                                 + 0.957375 w_k1_KO_s112 + 1.1156 w_k1_MAT_s112
                                 + 1.23745 w_k1_MSFT_s112
                                 + 1.20113 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.828126 w_k1_ADM_s112
                                 + 0.90362 w_k1_AXP_s112
                                 + 0.86214 w_k1_DIS_s112
                                 + 0.933598 w_k1_FDX_s112
                                 + 0.771674 w_k1_IBM_s112
                                 + 0.944024 w_k1_INTC_s112
                                 + 0.712376 w_k1_KO_s112
                                 + 0.838514 w_k1_MAT_s112
                                 + 1.10734 w_k1_MSFT_s112
                                 + 0.894674 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.06749 w_k1_ADM_s113
                                 + 1.0866 w_k1_AXP_s113 + 1.08213 w_k1_DIS_s113
                                 + 1.07762 w_k1_FDX_s113
                                 + 0.905606 w_k1_IBM_s113
                                 + 0.870808 w_k1_INTC_s113
                                 + 1.04534 w_k1_KO_s113
                                 + 0.879149 w_k1_MAT_s113
                                 + 0.980502 w_k1_MSFT_s113
                                 + 1.05279 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.10357 w_k1_ADM_s113
                                 + 0.99311 w_k1_AXP_s113
                                 + 0.905523 w_k1_DIS_s113
                                 + 1.04374 w_k1_FDX_s113
                                 + 0.914027 w_k1_IBM_s113
                                 + 0.824188 w_k1_INTC_s113
                                 + 0.864725 w_k1_KO_s113
                                 + 0.773275 w_k1_MAT_s113
                                 + 0.912345 w_k1_MSFT_s113
                                 + 0.952854 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.29236 w_k1_ADM_s113
                                 + 1.32956 w_k1_AXP_s113
                                 + 1.22232 w_k1_DIS_s113
                                 + 1.30854 w_k1_FDX_s113
                                 + 0.823693 w_k1_IBM_s113
                                 + 1.17725 w_k1_INTC_s113
                                 + 0.974461 w_k1_KO_s113
                                 + 0.725594 w_k1_MAT_s113
                                 + 1.16851 w_k1_MSFT_s113
                                 + 1.07453 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.826968 w_k1_ADM_s113
                                 + 0.824835 w_k1_AXP_s113
                                 + 0.786345 w_k1_DIS_s113
                                 + 0.860186 w_k1_FDX_s113
                                 + 0.777542 w_k1_IBM_s113
                                 + 1.00883 w_k1_INTC_s113
                                 + 0.714175 w_k1_KO_s113
                                 + 0.523596 w_k1_MAT_s113
                                 + 0.836737 w_k1_MSFT_s113
                                 + 0.825497 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.960745 w_k1_ADM_s113
                                 + 1.50817 w_k1_AXP_s113
                                 + 1.00728 w_k1_DIS_s113
                                 + 1.05596 w_k1_FDX_s113
                                 + 1.09667 w_k1_IBM_s113
                                 + 1.07767 w_k1_INTC_s113
                                 + 0.951511 w_k1_KO_s113
                                 + 1.11714 w_k1_MAT_s113
                                 + 0.846094 w_k1_MSFT_s113
                                 + 0.977527 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.35244 w_k1_ADM_s113
                                 + 1.23849 w_k1_AXP_s113 + 1.4368 w_k1_DIS_s113
                                 + 1.34631 w_k1_FDX_s113
                                 + 0.977134 w_k1_IBM_s113
                                 + 1.15372 w_k1_INTC_s113
                                 + 1.10326 w_k1_KO_s113
                                 + 0.975351 w_k1_MAT_s113
                                 + 1.25555 w_k1_MSFT_s113
                                 + 1.14875 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.839993 w_k1_ADM_s113
                                 + 1.4817 w_k1_AXP_s113 + 1.59793 w_k1_DIS_s113
                                 + 1.25275 w_k1_FDX_s113
                                 + 1.70864 w_k1_IBM_s113
                                 + 1.44961 w_k1_INTC_s113
                                 + 1.46681 w_k1_KO_s113 + 1.89919 w_k1_MAT_s113
                                 + 1.43254 w_k1_MSFT_s113
                                 + 1.10547 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.01409 w_k1_ADM_s113
                                 + 1.10305 w_k1_AXP_s113
                                 + 1.12826 w_k1_DIS_s113
                                 + 0.886935 w_k1_FDX_s113
                                 + 1.58503 w_k1_IBM_s113
                                 + 0.946123 w_k1_INTC_s113
                                 + 1.11013 w_k1_KO_s113 + 1.36577 w_k1_MAT_s113
                                 + 0.938828 w_k1_MSFT_s113
                                 + 0.963645 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.18128 w_k1_ADM_s114
                                 + 1.08141 w_k1_AXP_s114
                                 + 1.39292 w_k1_DIS_s114
                                 + 1.07202 w_k1_FDX_s114
                                 + 1.26797 w_k1_IBM_s114
                                 + 0.982736 w_k1_INTC_s114
                                 + 1.23019 w_k1_KO_s114 + 1.55429 w_k1_MAT_s114
                                 + 1.15447 w_k1_MSFT_s114 + 1.1152 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.875093 w_k1_ADM_s114
                                 + 0.982527 w_k1_AXP_s114
                                 + 0.90111 w_k1_DIS_s114
                                 + 1.01409 w_k1_FDX_s114
                                 + 0.80984 w_k1_IBM_s114
                                 + 1.0734 w_k1_INTC_s114
                                 + 0.879695 w_k1_KO_s114
                                 + 0.709702 w_k1_MAT_s114
                                 + 0.905478 w_k1_MSFT_s114
                                 + 0.91226 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.979811 w_k1_ADM_s114
                                 + 1.36904 w_k1_AXP_s114
                                 + 1.21286 w_k1_DIS_s114 + 1.1955 w_k1_FDX_s114
                                 + 1.20466 w_k1_IBM_s114
                                 + 1.09645 w_k1_INTC_s114
                                 + 1.09636 w_k1_KO_s114 + 1.57169 w_k1_MAT_s114
                                 + 1.07381 w_k1_MSFT_s114
                                 + 1.09576 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.791243 w_k1_ADM_s114
                                 + 0.763667 w_k1_AXP_s114
                                 + 0.743887 w_k1_DIS_s114
                                 + 0.842297 w_k1_FDX_s114
                                 + 0.634965 w_k1_IBM_s114
                                 + 0.667948 w_k1_INTC_s114
                                 + 0.690992 w_k1_KO_s114
                                 + 0.810991 w_k1_MAT_s114
                                 + 0.734325 w_k1_MSFT_s114
                                 + 0.744699 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.11049 w_k1_ADM_s114
                                 + 0.940469 w_k1_AXP_s114
                                 + 1.06694 w_k1_DIS_s114
                                 + 1.00572 w_k1_FDX_s114
                                 + 1.06862 w_k1_IBM_s114
                                 + 0.856058 w_k1_INTC_s114
                                 + 1.0757 w_k1_KO_s114 + 1.10512 w_k1_MAT_s114
                                 + 1.03445 w_k1_MSFT_s114
                                 + 1.11627 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.932314 w_k1_ADM_s114
                                 + 0.904297 w_k1_AXP_s114
                                 + 0.742663 w_k1_DIS_s114
                                 + 0.79577 w_k1_FDX_s114
                                 + 1.08293 w_k1_IBM_s114
                                 + 0.819294 w_k1_INTC_s114
                                 + 0.987726 w_k1_KO_s114
                                 + 1.31742 w_k1_MAT_s114
                                 + 0.84625 w_k1_MSFT_s114
                                 + 0.845788 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.04319 w_k1_ADM_s114
                                 + 1.4498 w_k1_AXP_s114 + 1.55327 w_k1_DIS_s114
                                 + 1.47882 w_k1_FDX_s114
                                 + 0.853439 w_k1_IBM_s114
                                 + 1.28568 w_k1_INTC_s114
                                 + 0.981853 w_k1_KO_s114
                                 + 1.41869 w_k1_MAT_s114
                                 + 1.16839 w_k1_MSFT_s114
                                 + 0.988839 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.578936 w_k1_ADM_s114
                                 + 0.613321 w_k1_AXP_s114
                                 + 0.856986 w_k1_DIS_s114
                                 + 0.738728 w_k1_FDX_s114
                                 + 0.796135 w_k1_IBM_s114
                                 + 0.8128 w_k1_INTC_s114
                                 + 0.829795 w_k1_KO_s114
                                 + 0.843812 w_k1_MAT_s114
                                 + 0.799948 w_k1_MSFT_s114
                                 + 0.744638 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.923753 w_k1_ADM_s115
                                 + 1.0409 w_k1_AXP_s115 + 1.03 w_k1_DIS_s115
                                 + 1.10502 w_k1_FDX_s115
                                 + 1.18799 w_k1_IBM_s115
                                 + 1.20286 w_k1_INTC_s115
                                 + 1.03482 w_k1_KO_s115 + 1.22552 w_k1_MAT_s115
                                 + 1.01628 w_k1_MSFT_s115
                                 + 1.07388 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.65183 w_k1_ADM_s115
                                 + 0.666671 w_k1_AXP_s115
                                 + 0.571577 w_k1_DIS_s115
                                 + 0.710529 w_k1_FDX_s115
                                 + 0.883547 w_k1_IBM_s115
                                 + 0.892498 w_k1_INTC_s115
                                 + 0.748687 w_k1_KO_s115
                                 + 0.814987 w_k1_MAT_s115
                                 + 0.715735 w_k1_MSFT_s115
                                 + 0.849225 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.733271 w_k1_ADM_s115
                                 + 0.832597 w_k1_AXP_s115
                                 + 0.789248 w_k1_DIS_s115
                                 + 1.01261 w_k1_FDX_s115
                                 + 0.687252 w_k1_IBM_s115
                                 + 1.15516 w_k1_INTC_s115
                                 + 0.794215 w_k1_KO_s115
                                 + 0.877891 w_k1_MAT_s115
                                 + 0.904381 w_k1_MSFT_s115
                                 + 0.882451 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.818047 w_k1_ADM_s115
                                 + 0.891647 w_k1_AXP_s115
                                 + 0.910246 w_k1_DIS_s115
                                 + 0.937426 w_k1_FDX_s115
                                 + 0.926365 w_k1_IBM_s115
                                 + 0.878919 w_k1_INTC_s115
                                 + 0.857108 w_k1_KO_s115
                                 + 0.896918 w_k1_MAT_s115
                                 + 0.833251 w_k1_MSFT_s115
                                 + 0.924613 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.913778 w_k1_ADM_s115
                                 + 0.968582 w_k1_AXP_s115
                                 + 1.28199 w_k1_DIS_s115
                                 + 1.21268 w_k1_FDX_s115
                                 + 0.804747 w_k1_IBM_s115
                                 + 1.02898 w_k1_INTC_s115
                                 + 0.946637 w_k1_KO_s115
                                 + 0.944725 w_k1_MAT_s115
                                 + 1.0458 w_k1_MSFT_s115
                                 + 0.963751 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.795411 w_k1_ADM_s115
                                 + 0.977545 w_k1_AXP_s115
                                 + 1.08731 w_k1_DIS_s115 + 1.0306 w_k1_FDX_s115
                                 + 1.62379 w_k1_IBM_s115
                                 + 0.976327 w_k1_INTC_s115
                                 + 1.14489 w_k1_KO_s115 + 1.55379 w_k1_MAT_s115
                                 + 0.962193 w_k1_MSFT_s115
                                 + 1.05948 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.14723 w_k1_ADM_s115
                                 + 1.26313 w_k1_AXP_s115
                                 + 1.65687 w_k1_DIS_s115
                                 + 1.41204 w_k1_FDX_s115
                                 + 1.06309 w_k1_IBM_s115
                                 + 1.02086 w_k1_INTC_s115
                                 + 1.17452 w_k1_KO_s115 + 1.2747 w_k1_MAT_s115
                                 + 1.13868 w_k1_MSFT_s115
                                 + 1.08199 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.618146 w_k1_ADM_s115
                                 + 0.605457 w_k1_AXP_s115
                                 + 0.890314 w_k1_DIS_s115
                                 + 0.771301 w_k1_FDX_s115
                                 + 1.17554 w_k1_IBM_s115
                                 + 0.983622 w_k1_INTC_s115
                                 + 0.975511 w_k1_KO_s115
                                 + 0.975444 w_k1_MAT_s115
                                 + 0.835936 w_k1_MSFT_s115
                                 + 0.981308 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.16819 w_k1_ADM_s116
                                 + 1.06523 w_k1_AXP_s116 + 1.2341 w_k1_DIS_s116
                                 + 1.17657 w_k1_FDX_s116
                                 + 1.11232 w_k1_IBM_s116
                                 + 1.07257 w_k1_INTC_s116
                                 + 1.05031 w_k1_KO_s116 + 1.02828 w_k1_MAT_s116
                                 + 1.18363 w_k1_MSFT_s116 + 1.0303 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.07909 w_k1_ADM_s116
                                 + 1.4955 w_k1_AXP_s116 + 1.85405 w_k1_DIS_s116
                                 + 1.37962 w_k1_FDX_s116
                                 + 1.30985 w_k1_IBM_s116
                                 + 1.55131 w_k1_INTC_s116
                                 + 1.44482 w_k1_KO_s116 + 1.25425 w_k1_MAT_s116
                                 + 1.35828 w_k1_MSFT_s116
                                 + 1.16341 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.09047 w_k1_ADM_s116
                                 + 1.13671 w_k1_AXP_s116
                                 + 0.981193 w_k1_DIS_s116
                                 + 1.10341 w_k1_FDX_s116
                                 + 0.724664 w_k1_IBM_s116
                                 + 1.1041 w_k1_INTC_s116
                                 + 0.843724 w_k1_KO_s116
                                 + 0.786474 w_k1_MAT_s116
                                 + 1.04614 w_k1_MSFT_s116
                                 + 0.882714 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.917303 w_k1_ADM_s116
                                 + 1.34769 w_k1_AXP_s116
                                 + 1.40021 w_k1_DIS_s116
                                 + 1.05826 w_k1_FDX_s116
                                 + 1.61922 w_k1_IBM_s116
                                 + 1.17014 w_k1_INTC_s116
                                 + 1.32501 w_k1_KO_s116 + 1.62254 w_k1_MAT_s116
                                 + 1.16379 w_k1_MSFT_s116
                                 + 1.15431 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.829887 w_k1_ADM_s116
                                 + 0.949044 w_k1_AXP_s116
                                 + 1.01992 w_k1_DIS_s116
                                 + 0.898777 w_k1_FDX_s116
                                 + 1.08397 w_k1_IBM_s116
                                 + 0.99205 w_k1_INTC_s116
                                 + 1.17817 w_k1_KO_s116 + 1.25979 w_k1_MAT_s116
                                 + 0.980943 w_k1_MSFT_s116
                                 + 0.998956 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.780559 w_k1_ADM_s116
                                 + 0.694123 w_k1_AXP_s116
                                 + 0.795392 w_k1_DIS_s116
                                 + 0.810496 w_k1_FDX_s116
                                 + 1.07655 w_k1_IBM_s116
                                 + 0.825465 w_k1_INTC_s116
                                 + 0.883276 w_k1_KO_s116
                                 + 0.690755 w_k1_MAT_s116
                                 + 0.827074 w_k1_MSFT_s116
                                 + 0.834385 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.946284 w_k1_ADM_s116
                                 + 0.994431 w_k1_AXP_s116
                                 + 1.17698 w_k1_DIS_s116
                                 + 0.959068 w_k1_FDX_s116
                                 + 1.40148 w_k1_IBM_s116
                                 + 1.19298 w_k1_INTC_s116
                                 + 1.35814 w_k1_KO_s116 + 1.47655 w_k1_MAT_s116
                                 + 1.10873 w_k1_MSFT_s116
                                 + 1.02089 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.810705 w_k1_ADM_s116
                                 + 0.710043 w_k1_AXP_s116
                                 + 0.84297 w_k1_DIS_s116
                                 + 0.884616 w_k1_FDX_s116
                                 + 0.714584 w_k1_IBM_s116
                                 + 0.95147 w_k1_INTC_s116
                                 + 0.940172 w_k1_KO_s116
                                 + 0.523755 w_k1_MAT_s116
                                 + 0.954973 w_k1_MSFT_s116
                                 + 0.860714 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.733744 w_k1_ADM_s117
                                 + 0.61439 w_k1_AXP_s117
                                 + 0.76787 w_k1_DIS_s117
                                 + 0.809167 w_k1_FDX_s117
                                 + 0.995448 w_k1_IBM_s117
                                 + 0.926973 w_k1_INTC_s117
                                 + 0.824699 w_k1_KO_s117
                                 + 0.767566 w_k1_MAT_s117
                                 + 0.825733 w_k1_MSFT_s117
                                 + 0.877316 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.808 w_k1_ADM_s117
                                 + 1.52261 w_k1_AXP_s117
                                 + 1.50087 w_k1_DIS_s117
                                 + 1.19233 w_k1_FDX_s117
                                 + 1.48746 w_k1_IBM_s117
                                 + 1.65002 w_k1_INTC_s117
                                 + 1.32384 w_k1_KO_s117 + 1.60895 w_k1_MAT_s117
                                 + 1.08917 w_k1_MSFT_s117
                                 + 1.02738 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.00484 w_k1_ADM_s117
                                 + 1.15221 w_k1_AXP_s117
                                 + 1.19205 w_k1_DIS_s117
                                 + 0.988006 w_k1_FDX_s117
                                 + 1.4154 w_k1_IBM_s117
                                 + 1.19059 w_k1_INTC_s117
                                 + 1.28856 w_k1_KO_s117 + 1.33557 w_k1_MAT_s117
                                 + 1.05639 w_k1_MSFT_s117
                                 + 1.00367 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.934486 w_k1_ADM_s117
                                 + 1.8514 w_k1_AXP_s117 + 1.73937 w_k1_DIS_s117
                                 + 1.61185 w_k1_FDX_s117
                                 + 1.05349 w_k1_IBM_s117
                                 + 1.58183 w_k1_INTC_s117 + 1.2277 w_k1_KO_s117
                                 + 1.54518 w_k1_MAT_s117
                                 + 1.37987 w_k1_MSFT_s117
                                 + 1.13314 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.761515 w_k1_ADM_s117
                                 + 0.904013 w_k1_AXP_s117
                                 + 1.07374 w_k1_DIS_s117
                                 + 1.00949 w_k1_FDX_s117
                                 + 1.14774 w_k1_IBM_s117
                                 + 1.17097 w_k1_INTC_s117
                                 + 1.14566 w_k1_KO_s117 + 1.23968 w_k1_MAT_s117
                                 + 1.04052 w_k1_MSFT_s117
                                 + 1.03536 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.713683 w_k1_ADM_s117
                                 + 0.705876 w_k1_AXP_s117
                                 + 0.54255 w_k1_DIS_s117
                                 + 0.755721 w_k1_FDX_s117
                                 + 0.738454 w_k1_IBM_s117
                                 + 0.797034 w_k1_INTC_s117
                                 + 0.691047 w_k1_KO_s117
                                 + 0.745634 w_k1_MAT_s117
                                 + 0.607117 w_k1_MSFT_s117
                                 + 0.765972 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.919867 w_k1_ADM_s117
                                 + 1.09849 w_k1_AXP_s117
                                 + 1.01295 w_k1_DIS_s117
                                 + 1.19974 w_k1_FDX_s117
                                 + 0.870787 w_k1_IBM_s117
                                 + 1.11995 w_k1_INTC_s117
                                 + 0.884346 w_k1_KO_s117
                                 + 0.719172 w_k1_MAT_s117
                                 + 0.991115 w_k1_MSFT_s117
                                 + 1.00806 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.04296 w_k1_ADM_s117
                                 + 1.12358 w_k1_AXP_s117
                                 + 0.76064 w_k1_DIS_s117
                                 + 0.870936 w_k1_FDX_s117
                                 + 1.06215 w_k1_IBM_s117
                                 + 0.912185 w_k1_INTC_s117
                                 + 0.868376 w_k1_KO_s117
                                 + 1.11405 w_k1_MAT_s117
                                 + 0.769726 w_k1_MSFT_s117
                                 + 0.916292 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0136361891438586 w_k1_ADM_s000
                                 + 0.0148391124151576 w_k1_ADM_s001
                                 + 0.0143871919417445 w_k1_ADM_s002
                                 + 0.011626660598419 w_k1_ADM_s003
                                 + 0.0152854218585419 w_k1_ADM_s004
                                 + 0.00416886634418927 w_k1_ADM_s005
                                 + 0.00384914188103486 w_k1_ADM_s006
                                 + 0.0142412007224664 w_k1_ADM_s007
                                 + 0.0301022123248075 w_k1_ADM_s010
                                 + 0.026348694727537 w_k1_ADM_s011
                                 + 0.0179637876474383 w_k1_ADM_s012
                                 + 0.0162332191800167 w_k1_ADM_s013
                                 + 0.0185327569349869 w_k1_ADM_s014
                                 + 0.0263384838317297 w_k1_ADM_s015
                                 + 0.0183660330771479 w_k1_ADM_s016
                                 + 0.0699229833125129 w_k1_ADM_s017
                                 + 0.0298769042879137 w_k1_ADM_s100
                                 + 0.137019276383685 w_k1_ADM_s101
                                 + 0.0626806418218794 w_k1_ADM_s102
                                 + 0.0455171120499583 w_k1_ADM_s103
                                 + 0.0625702872446352 w_k1_ADM_s104
                                 + 0.0646571962652088 w_k1_ADM_s105
                                 + 0.0471290292014369 w_k1_ADM_s106
                                 + 0.0984101732178577 w_k1_ADM_s107
                                 + 0.0155781699148611 w_k1_ADM_s110
                                 + 0.00876440949522407 w_k1_ADM_s111
                                 + 0.014639994450846 w_k1_ADM_s112
                                 + 0.0154052025731376 w_k1_ADM_s113
                                 + 0.0105629726944475 w_k1_ADM_s114
                                 + 0.0235681860019445 w_k1_ADM_s115
                                 + 0.017926814649158 w_k1_ADM_s116
                                 + 0.0170148976664158 w_k1_ADM_s117
                                 + 0.0151597225456129 w_k1_AXP_s000
                                 + 0.013906062085349 w_k1_AXP_s001
                                 + 0.0127292455860045 w_k1_AXP_s002
                                 + 0.0112105647704477 w_k1_AXP_s003
                                 + 0.0156527731427693 w_k1_AXP_s004
                                 + 0.00471846690078046 w_k1_AXP_s005
                                 + 0.00420427008964169 w_k1_AXP_s006
                                 + 0.0143093975223971 w_k1_AXP_s007
                                 + 0.0326608150893846 w_k1_AXP_s010
                                 + 0.0281589986837054 w_k1_AXP_s011
                                 + 0.0186998135910691 w_k1_AXP_s012
                                 + 0.0165580052361241 w_k1_AXP_s013
                                 + 0.0215274411347059 w_k1_AXP_s014
                                 + 0.0273044739637026 w_k1_AXP_s015
                                 + 0.0211824048426805 w_k1_AXP_s016
                                 + 0.0754672589095316 w_k1_AXP_s017
                                 + 0.0333807475909116 w_k1_AXP_s100
                                 + 0.151814990374617 w_k1_AXP_s101
                                 + 0.0623177863844076 w_k1_AXP_s102
                                 + 0.0535391358520654 w_k1_AXP_s103
                                 + 0.0598731300874257 w_k1_AXP_s104
                                 + 0.061816519527634 w_k1_AXP_s105
                                 + 0.0493678870821188 w_k1_AXP_s106
                                 + 0.104701896888716 w_k1_AXP_s107
                                 + 0.0159840067208029 w_k1_AXP_s110
                                 + 0.00842649236061482 w_k1_AXP_s111
                                 + 0.0157260231018849 w_k1_AXP_s112
                                 + 0.0176391734637817 w_k1_AXP_s113
                                 + 0.0108644225322246 w_k1_AXP_s114
                                 + 0.0262063550565723 w_k1_AXP_s115
                                 + 0.0184142855492508 w_k1_AXP_s116
                                 + 0.021402222167422 w_k1_AXP_s117
                                 + 0.0158994840348469 w_k1_DIS_s000
                                 + 0.0135278293331923 w_k1_DIS_s001
                                 + 0.0150524289137496 w_k1_DIS_s002
                                 + 0.0122034185924499 w_k1_DIS_s003
                                 + 0.0160811808910623 w_k1_DIS_s004
                                 + 0.0044006134050115 w_k1_DIS_s005
                                 + 0.00412497606306337 w_k1_DIS_s006
                                 + 0.0131916439294042 w_k1_DIS_s007
                                 + 0.0314974411361586 w_k1_DIS_s010
                                 + 0.0269834444574469 w_k1_DIS_s011
                                 + 0.0198269327557934 w_k1_DIS_s012
                                 + 0.0168715970602077 w_k1_DIS_s013
                                 + 0.0202660426030591 w_k1_DIS_s014
                                 + 0.0265518300262624 w_k1_DIS_s015
                                 + 0.0213961153795092 w_k1_DIS_s016
                                 + 0.0717913773861981 w_k1_DIS_s017
                                 + 0.0342793567705764 w_k1_DIS_s100
                                 + 0.15379715130051 w_k1_DIS_s101
                                 + 0.0616248151455711 w_k1_DIS_s102
                                 + 0.0504942235179436 w_k1_DIS_s103
                                 + 0.0592844437692061 w_k1_DIS_s104
                                 + 0.0606669736440907 w_k1_DIS_s105
                                 + 0.0493376402451201 w_k1_DIS_s106
                                 + 0.0976414817363883 w_k1_DIS_s107
                                 + 0.0185261849013592 w_k1_DIS_s110
                                 + 0.00915474394734422 w_k1_DIS_s111
                                 + 0.0149482964647583 w_k1_DIS_s112
                                 + 0.0162268597242589 w_k1_DIS_s113
                                 + 0.0111977450825986 w_k1_DIS_s114
                                 + 0.028747869472218 w_k1_DIS_s115
                                 + 0.0208752061927204 w_k1_DIS_s116
                                 + 0.0206773339258155 w_k1_DIS_s117
                                 + 0.0149164410907024 w_k1_FDX_s000
                                 + 0.013027351820136 w_k1_FDX_s001
                                 + 0.0134529071073846 w_k1_FDX_s002
                                 + 0.0111679086488838 w_k1_FDX_s003
                                 + 0.0154542209835647 w_k1_FDX_s004
                                 + 0.0045547891401123 w_k1_FDX_s005
                                 + 0.00407602971431166 w_k1_FDX_s006
                                 + 0.0130119105880035 w_k1_FDX_s007
                                 + 0.031901694083729 w_k1_FDX_s010
                                 + 0.0269022847834925 w_k1_FDX_s011
                                 + 0.018668287832028 w_k1_FDX_s012
                                 + 0.0160000764408107 w_k1_FDX_s013
                                 + 0.0213042879013928 w_k1_FDX_s014
                                 + 0.0267259594025753 w_k1_FDX_s015
                                 + 0.0203250407030175 w_k1_FDX_s016
                                 + 0.0685956580204298 w_k1_FDX_s017
                                 + 0.0344642024611048 w_k1_FDX_s100
                                 + 0.140970979388008 w_k1_FDX_s101
                                 + 0.059573621296738 w_k1_FDX_s102
                                 + 0.0528377671746384 w_k1_FDX_s103
                                 + 0.0639926458925869 w_k1_FDX_s104
                                 + 0.0618527183496361 w_k1_FDX_s105
                                 + 0.0481879807308905 w_k1_FDX_s106
                                 + 0.0970580962280324 w_k1_FDX_s107
                                 + 0.0183635353047586 w_k1_FDX_s110
                                 + 0.00837329306944222 w_k1_FDX_s111
                                 + 0.0153407709288433 w_k1_FDX_s112
                                 + 0.0159475520388385 w_k1_FDX_s113
                                 + 0.0107446164974443 w_k1_FDX_s114
                                 + 0.0292589619292861 w_k1_FDX_s115
                                 + 0.0191154713375654 w_k1_FDX_s116
                                 + 0.0208678831208079 w_k1_FDX_s117
                                 + 0.014628869035956 w_k1_IBM_s000
                                 + 0.0132649121467253 w_k1_IBM_s001
                                 + 0.0130751209061851 w_k1_IBM_s002
                                 + 0.0114254908080882 w_k1_IBM_s003
                                 + 0.0155285081051616 w_k1_IBM_s004
                                 + 0.00467763431096983 w_k1_IBM_s005
                                 + 0.00427560149154266 w_k1_IBM_s006
                                 + 0.014208923109965 w_k1_IBM_s007
                                 + 0.0325742976006021 w_k1_IBM_s010
                                 + 0.0282281111676429 w_k1_IBM_s011
                                 + 0.0191923847072548 w_k1_IBM_s012
                                 + 0.0163076802925526 w_k1_IBM_s013
                                 + 0.0207972068132567 w_k1_IBM_s014
                                 + 0.028036711774088 w_k1_IBM_s015
                                 + 0.0198737392696816 w_k1_IBM_s016
                                 + 0.0702016833159321 w_k1_IBM_s017
                                 + 0.0345373240100299 w_k1_IBM_s100
                                 + 0.150018845834443 w_k1_IBM_s101
                                 + 0.0602508637937301 w_k1_IBM_s102
                                 + 0.054781708761227 w_k1_IBM_s103
                                 + 0.0634920810225593 w_k1_IBM_s104
                                 + 0.0634540511251781 w_k1_IBM_s105
                                 + 0.0495085487869379 w_k1_IBM_s106
                                 + 0.10132868940575 w_k1_IBM_s107
                                 + 0.0177608461907036 w_k1_IBM_s110
                                 + 0.00892140345981337 w_k1_IBM_s111
                                 + 0.0154719907350883 w_k1_IBM_s112
                                 + 0.0160903436826253 w_k1_IBM_s113
                                 + 0.0109623639162356 w_k1_IBM_s114
                                 + 0.0288384492296968 w_k1_IBM_s115
                                 + 0.0210466658248153 w_k1_IBM_s116
                                 + 0.0212233855322149 w_k1_IBM_s117
                                 + 0.0137834048031286 w_k1_INTC_s000
                                 + 0.0128584081127507 w_k1_INTC_s001
                                 + 0.0139947912913644 w_k1_INTC_s002
                                 + 0.0114478314630768 w_k1_INTC_s003
                                 + 0.0160632326155133 w_k1_INTC_s004
                                 + 0.00502292919327324 w_k1_INTC_s005
                                 + 0.00402670968534045 w_k1_INTC_s006
                                 + 0.0108693233470038 w_k1_INTC_s007
                                 + 0.0336173833521978 w_k1_INTC_s010
                                 + 0.0307492792095546 w_k1_INTC_s011
                                 + 0.0191853182124472 w_k1_INTC_s012
                                 + 0.0156198388744793 w_k1_INTC_s013
                                 + 0.0228208185836897 w_k1_INTC_s014
                                 + 0.0258322393382753 w_k1_INTC_s015
                                 + 0.019143543690755 w_k1_INTC_s016
                                 + 0.0679004662661645 w_k1_INTC_s017
                                 + 0.0299676567275414 w_k1_INTC_s100
                                 + 0.145547192995189 w_k1_INTC_s101
                                 + 0.0499410089206299 w_k1_INTC_s102
                                 + 0.0670635667545584 w_k1_INTC_s103
                                 + 0.0642511215259465 w_k1_INTC_s104
                                 + 0.0629271291747508 w_k1_INTC_s105
                                 + 0.0435581874334627 w_k1_INTC_s106
                                 + 0.104403164023052 w_k1_INTC_s107
                                 + 0.0181739856581978 w_k1_INTC_s110
                                 + 0.00946241380309444 w_k1_INTC_s111
                                 + 0.0138464409232601 w_k1_INTC_s112
                                 + 0.0159187348648242 w_k1_INTC_s113
                                 + 0.0101268518508024 w_k1_INTC_s114
                                 + 0.0290445620002336 w_k1_INTC_s115
                                 + 0.0199793917064799 w_k1_INTC_s116
                                 + 0.0229114576106526 w_k1_INTC_s117
                                 + 0.0165374559652296 w_k1_KO_s000
                                 + 0.0141179923212845 w_k1_KO_s001
                                 + 0.0132602959986894 w_k1_KO_s002
                                 + 0.0118157336098998 w_k1_KO_s003
                                 + 0.0160052074062834 w_k1_KO_s004
                                 + 0.00469968873776913 w_k1_KO_s005
                                 + 0.00376000890867702 w_k1_KO_s006
                                 + 0.0132795688571489 w_k1_KO_s007
                                 + 0.0338338794358171 w_k1_KO_s010
                                 + 0.0274198713405039 w_k1_KO_s011
                                 + 0.0213857329285756 w_k1_KO_s012
                                 + 0.0147815962469013 w_k1_KO_s013
                                 + 0.0184389182309943 w_k1_KO_s014
                                 + 0.0304979807268375 w_k1_KO_s015
                                 + 0.0205612716733416 w_k1_KO_s016
                                 + 0.0746330411983757 w_k1_KO_s017
                                 + 0.0333380345299213 w_k1_KO_s100
                                 + 0.149555586364945 w_k1_KO_s101
                                 + 0.0558933670726566 w_k1_KO_s102
                                 + 0.055842937734554 w_k1_KO_s103
                                 + 0.0569192655813647 w_k1_KO_s104
                                 + 0.0603921130999315 w_k1_KO_s105
                                 + 0.0523707274102065 w_k1_KO_s106
                                 + 0.0986213926527416 w_k1_KO_s107
                                 + 0.0173250694357961 w_k1_KO_s110
                                 + 0.00889410762099841 w_k1_KO_s111
                                 + 0.0128837589561628 w_k1_KO_s112
                                 + 0.0147432765212954 w_k1_KO_s113
                                 + 0.0108504925497087 w_k1_KO_s114
                                 + 0.0267569166670934 w_k1_KO_s115
                                 + 0.0207025482605179 w_k1_KO_s116
                                 + 0.0199028635733913 w_k1_KO_s117
                                 + 0.0166370942726554 w_k1_MAT_s000
                                 + 0.010514691654262 w_k1_MAT_s001
                                 + 0.0153617758422811 w_k1_MAT_s002
                                 + 0.011480645088143 w_k1_MAT_s003
                                 + 0.0142414239758324 w_k1_MAT_s004
                                 + 0.00378902538844983 w_k1_MAT_s005
                                 + 0.00322146492232944 w_k1_MAT_s006
                                 + 0.0120629993312509 w_k1_MAT_s007
                                 + 0.0313475536204123 w_k1_MAT_s010
                                 + 0.0344625178059144 w_k1_MAT_s011
                                 + 0.0274596253048724 w_k1_MAT_s012
                                 + 0.0143876371447447 w_k1_MAT_s013
                                 + 0.0184647195208372 w_k1_MAT_s014
                                 + 0.0337406866910675 w_k1_MAT_s015
                                 + 0.0222408431187574 w_k1_MAT_s016
                                 + 0.0808877930308623 w_k1_MAT_s017
                                 + 0.0309807931866799 w_k1_MAT_s100
                                 + 0.154981502363643 w_k1_MAT_s101
                                 + 0.0560435193882475 w_k1_MAT_s102
                                 + 0.0581879254853703 w_k1_MAT_s103
                                 + 0.0578853244241353 w_k1_MAT_s104
                                 + 0.0567402384550012 w_k1_MAT_s105
                                 + 0.0539404953780642 w_k1_MAT_s106
                                 + 0.0919399329932386 w_k1_MAT_s107
                                 + 0.0159143287430814 w_k1_MAT_s110
                                 + 0.00818875126391447 w_k1_MAT_s111
                                 + 0.0162449907469577 w_k1_MAT_s112
                                 + 0.0145768943923524 w_k1_MAT_s113
                                 + 0.0128856556009602 w_k1_MAT_s114
                                 + 0.0304328949223991 w_k1_MAT_s115
                                 + 0.0190975200119102 w_k1_MAT_s116
                                 + 0.0208840096842755 w_k1_MAT_s117
                                 + 0.0138090595118199 w_k1_MSFT_s000
                                 + 0.0127323659358262 w_k1_MSFT_s001
                                 + 0.0151579334166499 w_k1_MSFT_s002
                                 + 0.0120399033939302 w_k1_MSFT_s003
                                 + 0.0143934519049904 w_k1_MSFT_s004
                                 + 0.00457033497619843 w_k1_MSFT_s005
                                 + 0.00340744555157683 w_k1_MSFT_s006
                                 + 0.012781159951354 w_k1_MSFT_s007
                                 + 0.0308393426407144 w_k1_MSFT_s010
                                 + 0.0281306082212914 w_k1_MSFT_s011
                                 + 0.0191916617565139 w_k1_MSFT_s012
                                 + 0.0159415952481397 w_k1_MSFT_s013
                                 + 0.021535180608287 w_k1_MSFT_s014
                                 + 0.0238190595965468 w_k1_MSFT_s015
                                 + 0.0214724174435508 w_k1_MSFT_s016
                                 + 0.0658972864784874 w_k1_MSFT_s017
                                 + 0.037273010902749 w_k1_MSFT_s100
                                 + 0.141682063914607 w_k1_MSFT_s101
                                 + 0.05734785267624 w_k1_MSFT_s102
                                 + 0.054036064233572 w_k1_MSFT_s103
                                 + 0.057617921820658 w_k1_MSFT_s104
                                 + 0.0642915356122006 w_k1_MSFT_s105
                                 + 0.0460996243434583 w_k1_MSFT_s106
                                 + 0.0909296910644888 w_k1_MSFT_s107
                                 + 0.01664645962359 w_k1_MSFT_s110
                                 + 0.00884601268120173 w_k1_MSFT_s111
                                 + 0.0159600018129463 w_k1_MSFT_s112
                                 + 0.0149269145361147 w_k1_MSFT_s113
                                 + 0.0104809402295994 w_k1_MSFT_s114
                                 + 0.0262826257836763 w_k1_MSFT_s115
                                 + 0.0199426096761437 w_k1_MSFT_s116
                                 + 0.0189749826845564 w_k1_MSFT_s117
                                 + 0.0145913295040426 w_k1_PG_s000
                                 + 0.0134510910694754 w_k1_PG_s001
                                 + 0.0136402223083492 w_k1_PG_s002
                                 + 0.0115768532404012 w_k1_PG_s003
                                 + 0.0165415243394195 w_k1_PG_s004
                                 + 0.00453180820013941 w_k1_PG_s005
                                 + 0.00381092811858761 w_k1_PG_s006
                                 + 0.0135134453933752 w_k1_PG_s007
                                 + 0.0329834800620429 w_k1_PG_s010
                                 + 0.0266887405455129 w_k1_PG_s011
                                 + 0.019428002490963 w_k1_PG_s012
                                 + 0.0144217547782583 w_k1_PG_s013
                                 + 0.0193126274764021 w_k1_PG_s014
                                 + 0.02844941424062 w_k1_PG_s015
                                 + 0.019773509120188 w_k1_PG_s016
                                 + 0.0745426331861306 w_k1_PG_s017
                                 + 0.0341516802979268 w_k1_PG_s100
                                 + 0.146384876979324 w_k1_PG_s101
                                 + 0.0524694693920642 w_k1_PG_s102
                                 + 0.0524604412151054 w_k1_PG_s103
                                 + 0.0635969449757285 w_k1_PG_s104
                                 + 0.0613984043928595 w_k1_PG_s105
                                 + 0.0442261523613482 w_k1_PG_s106
                                 + 0.0951354035007295 w_k1_PG_s107
                                 + 0.0175299843254052 w_k1_PG_s110
                                 + 0.00886517348232957 w_k1_PG_s111
                                 + 0.0153839485613199 w_k1_PG_s112
                                 + 0.0148920442849076 w_k1_PG_s113
                                 + 0.0104316959868591 w_k1_PG_s114
                                 + 0.0272631883636119 w_k1_PG_s115
                                 + 0.0185364705597378 w_k1_PG_s116
                                 + 0.0192153088210775 w_k1_PG_s117 >= 1.05
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
