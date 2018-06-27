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
 w_k1_ADM_s0_return_constraint:  1.16554 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.950647 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.789865 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.18976 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.864416 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.14687 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.921687 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.44617 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.44414 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.939147 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.07201 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.19898 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.943486 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.38207 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.08243 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.12146 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.39646 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.822507 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.761969 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.03916 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.08777 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.954209 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.01257 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.38873 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.936998 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.784164 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.15754 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.51252 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.30642 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.29872 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.20509 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.876181 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.0208 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.692539 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.02819 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.00773 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.22269 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.06909 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24429 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.872076 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.809556 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.27303 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.960614 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.42471 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.699143 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.72806 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.34425 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.942223 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.2062 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.15539 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.900887 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.27733 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.1025 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.41406 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.61114 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.972278 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.792441 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.14518 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.915533 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.740515 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.944617 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.18505 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.730149 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.67865 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.28481 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.68465 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.17013 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.01624 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.40367 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.808703 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.22185 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.681903 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.01973 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.96751 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.04942 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.769833 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.3255 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.885626 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.824078 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.29528 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.903138 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.41596 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.863939 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.53883 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.66406 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.84505 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.07388 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.18033 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.776403 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.18544 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.994167 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.39537 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.79987 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.923294 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.736014 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.2336 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.23732 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.837007 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.986539 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.23815 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.801762 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.733346 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.969453 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.85518 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.05611 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.62072 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.53314 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.676618 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.26477 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.554261 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.1473 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.992164 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.16737 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.782287 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20773 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.918274 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.867189 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.15738 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.928878 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.23293 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.85588 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.40913 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.1489 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.825371 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.938968 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.12669 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.918968 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.20798 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.00652 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.29952 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.41575 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.822138 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.807103 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.27508 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.16116 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.02207 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.920403 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.23356 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.9233 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.80866 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.07247 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.49005 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.30326 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.32097 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.42813 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.699523 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.01575 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.698107 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.951819 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.998148 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.22908 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.894547 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.19806 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.937794 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.961797 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.1654 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.891264 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.33741 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.854717 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.23554 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.51639 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.1944 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.76676 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.14109 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.646329 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.725402 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.1123 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.04254 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.64604 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.5902 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.791102 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.649809 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.849138 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.537296 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.50794 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.04112 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.61009 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.92042 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.16261 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.48985 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.656508 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.897004 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.27099 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.27753 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.89309 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.707921 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.32455 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.03614 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.677335 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.737994 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11459 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.951588 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.915134 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.20802 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.94886 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.09292 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.889025 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.15843 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.11777 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.857299 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.10865 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.05356 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.945622 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.21535 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.07442 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.09374 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.29383 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.82001 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.865043 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.13025 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.1053 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.905739 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.10082 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.07296 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.973037 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.864719 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.935151 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.24072 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.18237 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.1453 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.2424 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.776933 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.00443 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.84107 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.01995 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.15032 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.27727 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.997512 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.2008 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.890603 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.878508 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.19382 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.972748 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.19964 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.981144 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.23383 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.57442 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.04832 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.33527 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.22838 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.81392 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.950468 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.00087 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.08696 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.65281 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.11657 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.874983 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.93657 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.00359 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.676695 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.06973 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.18427 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.843138 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.807937 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.02471 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.3629 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.885253 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.01295 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.51743 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.03385 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.46588 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.764328 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.40677 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.17387 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.10571 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.947809 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.31958 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.910334 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.797739 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.22147 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.922832 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.34453 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.873028 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.86462 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.586 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.15913 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 2.17786 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.34939 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.729144 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.793753 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.881075 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.40273 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.67125 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.3215 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.744426 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.631517 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.665904 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.474521 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.54701 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.882516 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.644011 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.777092 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.20535 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.63608 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.661924 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.720178 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.45145 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.30964 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 2.10085 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.967046 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.40801 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.03837 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.730031 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.01322 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19099 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.917202 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.855073 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.08892 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.985372 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.21472 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.992512 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.1769 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.19122 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.905876 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.07007 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.12636 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.86881 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.15724 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.00746 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.17968 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.45461 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.84417 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.841731 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.24091 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.30717 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.998882 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.937137 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.13662 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.88316 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.838079 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.919049 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.2861 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.04965 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.18572 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.18066 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.701168 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.12682 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.699424 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.0207 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.05191 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.22709 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.958417 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12222 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.924737 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.954118 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.05284 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.946546 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.21939 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.887347 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.28128 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.2005 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.882937 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.06613 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.05064 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.86434 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.01074 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.03482 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.1074 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.30599 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.03695 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.933646 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.02743 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.10288 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.901041 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.03597 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.19028 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.916646 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.867121 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.05335 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.23862 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.02415 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.12141 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.19548 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.86446 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.0239 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.795638 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.06448 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.03561 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.966188 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.914276 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.166999944207698 z_k1_s0000
                                 - 0.157000165582162 z_k1_s0001
                                 - 0.253000020667985 z_k1_s0002
                                 - 0.777001452170533 z_k1_s0003
                                 - 0.578998936342558 z_k1_s0004
                                 - 0.234000185759379 z_k1_s0005
                                 - 0.243999964384915 z_k1_s0006
                                 - 0.782999786225329 z_k1_s0007
                                 - 0.0869999450638213 z_k1_s0010
                                 - 0.07099997683943 z_k1_s0011
                                 - 0.227999578622561 z_k1_s0012
                                 - 0.0629999298772645 z_k1_s0013
                                 - 0.304999590683616 z_k1_s0014
                                 - 0.12699967707489 z_k1_s0015
                                 - 0.121999961973386 z_k1_s0016
                                 - 0.25399935414978 z_k1_s0017
                                 - 0.202000004794828 z_k1_s0020
                                 - 0.074999931116945 z_k1_s0021
                                 - 0.0799998899247691 z_k1_s0022
                                 - 0.226000070592183 z_k1_s0023
                                 - 0.0859999063741081 z_k1_s0024
                                 - 0.195000260103724 z_k1_s0025
                                 - 0.123000062711657 z_k1_s0026
                                 - 0.111000029812979 z_k1_s0027
                                 - 0.301999563633655 z_k1_s0030
                                 - 0.13499979764022 z_k1_s0031
                                 - 0.143999857589469 z_k1_s0032
                                 - 0.0879999129713424 z_k1_s0033
                                 - 0.235999124005824 z_k1_s0034
                                 - 0.193000061579992 z_k1_s0035
                                 - 0.33499978344757 z_k1_s0036
                                 - 0.402999624398276 z_k1_s0037
                                 - 0.127000172547346 z_k1_s0040
                                 - 0.388999715699016 z_k1_s0041
                                 - 0.457001048694585 z_k1_s0042
                                 - 0.124000246900667 z_k1_s0043
                                 - 0.123000211318357 z_k1_s0044
                                 - 0.338000074204216 z_k1_s0045
                                 - 0.117000178924748 z_k1_s0046
                                 - 0.136000130587634 z_k1_s0047
                                 - 0.395999415790952 z_k1_s0050
                                 - 0.289999902512695 z_k1_s0051
                                 - 0.205999902557369 z_k1_s0052
                                 - 0.243999805204085 z_k1_s0053
                                 - 0.302000194505534 z_k1_s0054
                                 - 0.331999902490358 z_k1_s0055
                                 - 0.171999484043234 z_k1_s0056
                                 - 0.10199982474622 z_k1_s0057
                                 - 0.386999227684339 z_k1_s0060
                                 - 0.636999013657526 z_k1_s0061
                                 - 0.329999699032635 z_k1_s0062
                                 - 0.643000517892279 z_k1_s0063
                                 - 1.2869974423293 z_k1_s0064
                                 - 0.442999581995345 z_k1_s0065
                                 - 0.422999488405652 z_k1_s0066
                                 - 0.463001520782344 z_k1_s0067
                                 - 0.221000562332749 z_k1_s0070
                                 - 0.169000235283505 z_k1_s0071
                                 - 0.429001870529723 z_k1_s0072
                                 - 0.400001836685229 z_k1_s0073
                                 - 0.169000235283505 z_k1_s0074
                                 - 0.444000670378063 z_k1_s0075
                                 - 0.112000161151208 z_k1_s0076
                                 - 0.262999850313018 z_k1_s0077
                                 - 0.821998166635822 z_k1_s0100
                                 - 0.287999379601564 z_k1_s0101
                                 - 0.298999603000153 z_k1_s0102
                                 - 0.419000764423339 z_k1_s0103
                                 - 0.347997955716638 z_k1_s0104
                                 - 0.419999721688693 z_k1_s0105
                                 - 0.420998678954047 z_k1_s0106
                                 - 0.323999594388751 z_k1_s0107
                                 - 0.236000174535731 z_k1_s0110
                                 - 0.361999442143859 z_k1_s0111
                                 - 0.168999860061626 z_k1_s0112
                                 - 0.162000304638919 z_k1_s0113
                                 - 0.112000011362726 z_k1_s0114
                                 - 0.105999935457586 z_k1_s0115
                                 - 0.162000304638919 z_k1_s0116
                                 - 0.144999992641031 z_k1_s0117
                                 - 0.172999851010418 z_k1_s0120
                                 - 0.162999738801745 z_k1_s0121
                                 - 0.157999682697408 z_k1_s0122
                                 - 0.160000302339036 z_k1_s0123
                                 - 0.17099923136879 z_k1_s0124
                                 - 0.103999972570413 z_k1_s0125
                                 - 0.455999443316507 z_k1_s0126
                                 - 0.108000017453883 z_k1_s0127
                                 - 0.315000155514666 z_k1_s0130
                                 - 0.13600036870546 z_k1_s0131
                                 - 0.232000698815361 z_k1_s0132
                                 - 0.200000192977789 z_k1_s0133
                                 - 0.241001212145558 z_k1_s0134
                                 - 0.255000691322737 z_k1_s0135
                                 - 0.10300026759896 z_k1_s0136
                                 - 0.497001300836753 z_k1_s0137
                                 - 0.401000448881974 z_k1_s0140
                                 - 0.391998523915194 z_k1_s0141
                                 - 0.237999789405422 z_k1_s0142
                                 - 0.247999947635846 z_k1_s0143
                                 - 0.493001081882149 z_k1_s0144
                                 - 0.3089998158818 z_k1_s0145
                                 - 0.357999631371199 z_k1_s0146
                                 - 0.988998234754764 z_k1_s0147
                                 - 0.0589999839347836 z_k1_s0150
                                 - 0.283000574204401 z_k1_s0151
                                 - 0.214000466628076 z_k1_s0152
                                 - 0.116999783455342 z_k1_s0153
                                 - 0.0919999905017028 z_k1_s0154
                                 - 0.158000462435731 z_k1_s0155
                                 - 0.12500011176867 z_k1_s0156
                                 - 0.0990000197007579 z_k1_s0157
                                 - 0.287999809451845 z_k1_s0160
                                 - 0.158999989359867 z_k1_s0161
                                 - 0.168000429180208 z_k1_s0162
                                 - 0.0870000369969054 z_k1_s0163
                                 - 0.0940000008460843 z_k1_s0164
                                 - 0.201999280990568 z_k1_s0165
                                 - 0.215999857088884 z_k1_s0166
                                 - 0.407000653787791 z_k1_s0167
                                 - 0.103999995834809 z_k1_s0170
                                 - 0.127000084551181 z_k1_s0171
                                 - 0.13600057025334 z_k1_s0172
                                 - 0.395000381867873 z_k1_s0173
                                 - 0.177999510596008 z_k1_s0174
                                 - 0.169000206594112 z_k1_s0175
                                 - 0.0940000543103212 z_k1_s0176
                                 - 0.110000013269514 z_k1_s0177
                                 - 1.39800243744781 z_k1_s1000
                                 - 1.09299933869661 z_k1_s1001
                                 - 1.29699929945629 z_k1_s1002
                                 - 0.900002473099042 z_k1_s1003
                                 - 0.78000207519234 z_k1_s1004
                                 - 1.41300108138278 z_k1_s1005
                                 - 1.16700082799539 z_k1_s1006
                                 - 2.17600980546107 z_k1_s1007
                                 - 0.78400068553191 z_k1_s1010
                                 - 0.861996214070687 z_k1_s1011
                                 - 0.307999238689538 z_k1_s1012
                                 - 0.547000089768554 z_k1_s1013
                                 - 0.321999422702274 z_k1_s1014
                                 - 0.36399944034152 z_k1_s1015
                                 - 0.744000923208324 z_k1_s1016
                                 - 1.41299895206641 z_k1_s1017
                                 - 0.298999774446197 z_k1_s1020
                                 - 1.38399769847754 z_k1_s1021
                                 - 0.923995879025584 z_k1_s1022
                                 - 0.473999303335358 z_k1_s1023
                                 - 0.542998872154364 z_k1_s1024
                                 - 1.14699804680986 z_k1_s1025
                                 - 1.0490015776503 z_k1_s1026
                                 - 1.22099882931216 z_k1_s1027
                                 - 0.577000003635127 z_k1_s1030
                                 - 1.46599619376787 z_k1_s1031
                                 - 0.899999336119452 z_k1_s1032
                                 - 1.33399427549863 z_k1_s1033
                                 - 2.80199490966098 z_k1_s1034
                                 - 1.26399563356607 z_k1_s1035
                                 - 1.42699359100813 z_k1_s1036
                                 - 1.42800141020089 z_k1_s1037
                                 - 0.977001154663418 z_k1_s1040
                                 - 0.724001523021293 z_k1_s1041
                                 - 0.309000713302326 z_k1_s1042
                                 - 0.638998674945019 z_k1_s1043
                                 - 0.702003342795625 z_k1_s1044
                                 - 0.932000337918733 z_k1_s1045
                                 - 0.332001000215978 z_k1_s1046
                                 - 1.25900079051759 z_k1_s1047
                                 - 0.344000949199732 z_k1_s1050
                                 - 0.261000250119059 z_k1_s1051
                                 - 0.241000202453967 z_k1_s1052
                                 - 0.545000732923166 z_k1_s1053
                                 - 0.46500118906349 z_k1_s1054
                                 - 0.329001641101693 z_k1_s1055
                                 - 0.58500212185474 z_k1_s1056
                                 - 0.464001881990981 z_k1_s1057
                                 - 0.523000764301294 z_k1_s1060
                                 - 0.612999601976488 z_k1_s1061
                                 - 0.135999878098047 z_k1_s1062
                                 - 0.58099936144754 z_k1_s1063
                                 - 0.301000035469202 z_k1_s1064
                                 - 0.295000112957522 z_k1_s1065
                                 - 0.390001501259077 z_k1_s1066
                                 - 0.430000984670275 z_k1_s1067
                                 - 0.446000705548139 z_k1_s1070
                                 - 0.344000878225789 z_k1_s1071
                                 - 0.215000119140708 z_k1_s1072
                                 - 0.375000207803561 z_k1_s1073
                                 - 0.351999965858057 z_k1_s1074
                                 - 0.18200015815405 z_k1_s1075
                                 - 0.709999247440315 z_k1_s1076
                                 - 0.241000305448586 z_k1_s1077
                                 - 0.317999793362611 z_k1_s1100
                                 - 0.417997768651404 z_k1_s1101
                                 - 0.170999604256086 z_k1_s1102
                                 - 0.190999446613219 z_k1_s1103
                                 - 0.347999680547999 z_k1_s1104
                                 - 0.189999615239957 z_k1_s1105
                                 - 0.50199794736924 z_k1_s1106
                                 - 0.334999152402456 z_k1_s1107
                                 - 0.209999844256842 z_k1_s1110
                                 - 0.0779999421525412 z_k1_s1111
                                 - 0.0909998936446784 z_k1_s1112
                                 - 0.129999514921372 z_k1_s1113
                                 - 0.0719999466023458 z_k1_s1114
                                 - 0.192999584798283 z_k1_s1115
                                 - 0.0889998562613269 z_k1_s1116
                                 - 0.302999891884731 z_k1_s1117
                                 - 0.209999664167965 z_k1_s1120
                                 - 0.0839998656671858 z_k1_s1121
                                 - 0.0899997891007054 z_k1_s1122
                                 - 0.102999818539563 z_k1_s1123
                                 - 0.0999997396230641 z_k1_s1124
                                 - 0.108999741973082 z_k1_s1125
                                 - 0.280999265996815 z_k1_s1126
                                 - 0.194999269585471 z_k1_s1127
                                 - 0.0639999492674285 z_k1_s1130
                                 - 0.145999625888473 z_k1_s1131
                                 - 0.135999880580798 z_k1_s1132
                                 - 0.170999918156666 z_k1_s1133
                                 - 0.064999923798196 z_k1_s1134
                                 - 0.0449999686833435 z_k1_s1135
                                 - 0.0939998354897568 z_k1_s1136
                                 - 0.208000090593869 z_k1_s1137
                                 - 0.291999058559172 z_k1_s1140
                                 - 0.310999279013194 z_k1_s1141
                                 - 0.140999952317709 z_k1_s1142
                                 - 0.300000209697065 z_k1_s1143
                                 - 0.110999931348003 z_k1_s1144
                                 - 0.403000635106175 z_k1_s1145
                                 - 0.178999844826086 z_k1_s1146
                                 - 0.0909999783014948 z_k1_s1147
                                 - 0.694999347686175 z_k1_s1150
                                 - 0.603996337268732 z_k1_s1151
                                 - 0.36099836392198 z_k1_s1152
                                 - 0.488998848477952 z_k1_s1153
                                 - 0.14199953853268 z_k1_s1154
                                 - 0.507996871848698 z_k1_s1155
                                 - 0.289999133453807 z_k1_s1156
                                 - 0.502996824858326 z_k1_s1157
                                 - 0.833000388840603 z_k1_s1160
                                 - 0.308999956765627 z_k1_s1161
                                 - 0.351000313160906 z_k1_s1162
                                 - 0.238000061740179 z_k1_s1163
                                 - 0.539000658073277 z_k1_s1164
                                 - 0.230999887174314 z_k1_s1165
                                 - 0.475998741480542 z_k1_s1166
                                 - 0.477999186213594 z_k1_s1167
                                 - 0.227000106895362 z_k1_s1170
                                 - 0.29700013985869 z_k1_s1171
                                 - 0.269000126673359 z_k1_s1172
                                 - 0.315000148334974 z_k1_s1173
                                 - 0.310000145980451 z_k1_s1174
                                 - 0.181000085233747 z_k1_s1175
                                 - 0.525000247224957 z_k1_s1176
                                 - 0.376000177060159 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.41654 w_k1_ADM_s000
                                 + 1.67116 w_k1_AXP_s000
                                 + 1.84632 w_k1_DIS_s000
                                 + 1.31816 w_k1_FDX_s000
                                 + 1.59443 w_k1_IBM_s000
                                 + 1.46301 w_k1_INTC_s000
                                 + 1.76648 w_k1_KO_s000 + 1.70476 w_k1_MAT_s000
                                 + 1.40831 w_k1_MSFT_s000
                                 + 1.36245 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.24439 w_k1_ADM_s000
                                 + 1.24681 w_k1_AXP_s000
                                 + 1.41322 w_k1_DIS_s000 + 1.3305 w_k1_FDX_s000
                                 + 0.797179 w_k1_IBM_s000
                                 + 1.30315 w_k1_INTC_s000
                                 + 0.995166 w_k1_KO_s000
                                 + 0.862242 w_k1_MAT_s000
                                 + 1.28232 w_k1_MSFT_s000
                                 + 1.07374 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.701369 w_k1_ADM_s000
                                 + 0.640298 w_k1_AXP_s000
                                 + 0.460212 w_k1_DIS_s000
                                 + 0.620438 w_k1_FDX_s000
                                 + 0.710671 w_k1_IBM_s000
                                 + 0.823667 w_k1_INTC_s000
                                 + 0.592794 w_k1_KO_s000
                                 + 0.51305 w_k1_MAT_s000
                                 + 0.693297 w_k1_MSFT_s000
                                 + 0.702517 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.870023 w_k1_ADM_s000
                                 + 0.945768 w_k1_AXP_s000
                                 + 0.717709 w_k1_DIS_s000
                                 + 0.772634 w_k1_FDX_s000
                                 + 0.990499 w_k1_IBM_s000
                                 + 0.717668 w_k1_INTC_s000
                                 + 0.843973 w_k1_KO_s000
                                 + 0.957655 w_k1_MAT_s000
                                 + 0.757462 w_k1_MSFT_s000
                                 + 0.894726 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.3142 w_k1_ADM_s000
                                 + 1.20394 w_k1_AXP_s000
                                 + 1.01408 w_k1_DIS_s000 + 1.0056 w_k1_FDX_s000
                                 + 0.958887 w_k1_IBM_s000
                                 + 0.981043 w_k1_INTC_s000
                                 + 0.928839 w_k1_KO_s000
                                 + 1.12114 w_k1_MAT_s000
                                 + 0.896082 w_k1_MSFT_s000
                                 + 1.04935 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.819318 w_k1_ADM_s000
                                 + 0.857577 w_k1_AXP_s000
                                 + 0.89578 w_k1_DIS_s000
                                 + 0.936142 w_k1_FDX_s000
                                 + 0.632373 w_k1_IBM_s000
                                 + 0.857545 w_k1_INTC_s000
                                 + 0.736655 w_k1_KO_s000
                                 + 0.692536 w_k1_MAT_s000
                                 + 0.933033 w_k1_MSFT_s000
                                 + 0.917983 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.20825 w_k1_ADM_s000
                                 + 1.21579 w_k1_AXP_s000
                                 + 1.39731 w_k1_DIS_s000
                                 + 1.17082 w_k1_FDX_s000
                                 + 0.796593 w_k1_IBM_s000
                                 + 1.10859 w_k1_INTC_s000
                                 + 1.14309 w_k1_KO_s000
                                 + 0.888189 w_k1_MAT_s000
                                 + 1.14231 w_k1_MSFT_s000
                                 + 1.11869 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.42877 w_k1_ADM_s000
                                 + 1.24221 w_k1_AXP_s000
                                 + 1.30983 w_k1_DIS_s000
                                 + 1.19709 w_k1_FDX_s000
                                 + 1.23974 w_k1_IBM_s000
                                 + 1.17018 w_k1_INTC_s000
                                 + 1.28768 w_k1_KO_s000 + 1.10272 w_k1_MAT_s000
                                 + 1.13766 w_k1_MSFT_s000
                                 + 1.23764 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.944648 w_k1_ADM_s001
                                 + 0.811984 w_k1_AXP_s001
                                 + 0.780345 w_k1_DIS_s001
                                 + 0.753984 w_k1_FDX_s001
                                 + 0.997882 w_k1_IBM_s001
                                 + 0.84122 w_k1_INTC_s001
                                 + 0.98519 w_k1_KO_s001
                                 + 0.835599 w_k1_MAT_s001
                                 + 0.874016 w_k1_MSFT_s001
                                 + 0.911043 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.879994 w_k1_ADM_s001
                                 + 0.618214 w_k1_AXP_s001
                                 + 0.739604 w_k1_DIS_s001
                                 + 0.76318 w_k1_FDX_s001
                                 + 0.796774 w_k1_IBM_s001
                                 + 0.91392 w_k1_INTC_s001
                                 + 0.910174 w_k1_KO_s001
                                 + 0.528131 w_k1_MAT_s001
                                 + 0.934395 w_k1_MSFT_s001
                                 + 0.815183 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.886176 w_k1_ADM_s001
                                 + 1.21351 w_k1_AXP_s001
                                 + 1.03173 w_k1_DIS_s001
                                 + 0.977337 w_k1_FDX_s001
                                 + 1.48303 w_k1_IBM_s001
                                 + 1.20202 w_k1_INTC_s001 + 1.236 w_k1_KO_s001
                                 + 1.13598 w_k1_MAT_s001
                                 + 1.03742 w_k1_MSFT_s001
                                 + 0.954268 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.88217 w_k1_ADM_s001
                                 + 0.914952 w_k1_AXP_s001
                                 + 0.636915 w_k1_DIS_s001
                                 + 0.726257 w_k1_FDX_s001
                                 + 1.09035 w_k1_IBM_s001
                                 + 0.797739 w_k1_INTC_s001
                                 + 0.892457 w_k1_KO_s001
                                 + 1.04299 w_k1_MAT_s001
                                 + 0.810188 w_k1_MSFT_s001
                                 + 0.833133 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.887347 w_k1_ADM_s001
                                 + 0.863349 w_k1_AXP_s001
                                 + 0.768311 w_k1_DIS_s001
                                 + 0.960342 w_k1_FDX_s001
                                 + 0.787325 w_k1_IBM_s001
                                 + 1.18373 w_k1_INTC_s001
                                 + 0.926945 w_k1_KO_s001
                                 + 0.531781 w_k1_MAT_s001
                                 + 1.07747 w_k1_MSFT_s001
                                 + 0.833411 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.02803 w_k1_ADM_s001
                                 + 0.848231 w_k1_AXP_s001
                                 + 1.00946 w_k1_DIS_s001
                                 + 0.970673 w_k1_FDX_s001
                                 + 1.24424 w_k1_IBM_s001
                                 + 1.09891 w_k1_INTC_s001
                                 + 1.19735 w_k1_KO_s001
                                 + 0.863276 w_k1_MAT_s001
                                 + 1.06102 w_k1_MSFT_s001
                                 + 1.09173 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.49642 w_k1_ADM_s001
                                 + 1.43276 w_k1_AXP_s001
                                 + 1.78265 w_k1_DIS_s001
                                 + 1.43773 w_k1_FDX_s001
                                 + 1.07955 w_k1_IBM_s001
                                 + 1.62166 w_k1_INTC_s001
                                 + 1.42545 w_k1_KO_s001 + 1.12682 w_k1_MAT_s001
                                 + 1.5722 w_k1_MSFT_s001 + 1.41425 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.19938 w_k1_ADM_s001
                                 + 0.865839 w_k1_AXP_s001
                                 + 1.18945 w_k1_DIS_s001
                                 + 1.14324 w_k1_FDX_s001
                                 + 0.84839 w_k1_IBM_s001
                                 + 1.25082 w_k1_INTC_s001
                                 + 1.13652 w_k1_KO_s001
                                 + 0.803946 w_k1_MAT_s001
                                 + 1.4123 w_k1_MSFT_s001
                                 + 0.952698 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.910753 w_k1_ADM_s002
                                 + 1.20193 w_k1_AXP_s002
                                 + 1.02414 w_k1_DIS_s002
                                 + 0.97585 w_k1_FDX_s002
                                 + 1.19434 w_k1_IBM_s002
                                 + 1.17433 w_k1_INTC_s002
                                 + 1.04029 w_k1_KO_s002 + 1.5888 w_k1_MAT_s002
                                 + 1.14806 w_k1_MSFT_s002
                                 + 0.947132 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.800634 w_k1_ADM_s002
                                 + 0.975089 w_k1_AXP_s002
                                 + 0.741169 w_k1_DIS_s002
                                 + 1.05074 w_k1_FDX_s002
                                 + 0.596381 w_k1_IBM_s002
                                 + 1.03533 w_k1_INTC_s002
                                 + 0.67286 w_k1_KO_s002
                                 + 0.754316 w_k1_MAT_s002
                                 + 0.915862 w_k1_MSFT_s002
                                 + 0.856513 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.673162 w_k1_ADM_s002
                                 + 0.887699 w_k1_AXP_s002
                                 + 0.669612 w_k1_DIS_s002
                                 + 0.733562 w_k1_FDX_s002
                                 + 0.920002 w_k1_IBM_s002
                                 + 0.93723 w_k1_INTC_s002
                                 + 0.733451 w_k1_KO_s002
                                 + 0.716345 w_k1_MAT_s002
                                 + 0.771161 w_k1_MSFT_s002
                                 + 0.810402 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.10672 w_k1_ADM_s002
                                 + 1.58822 w_k1_AXP_s002
                                 + 1.42576 w_k1_DIS_s002
                                 + 1.30533 w_k1_FDX_s002
                                 + 1.30801 w_k1_IBM_s002
                                 + 1.22222 w_k1_INTC_s002
                                 + 1.36007 w_k1_KO_s002 + 1.45706 w_k1_MAT_s002
                                 + 1.25783 w_k1_MSFT_s002
                                 + 1.21366 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.662303 w_k1_ADM_s002
                                 + 0.58403 w_k1_AXP_s002
                                 + 0.598453 w_k1_DIS_s002
                                 + 0.740331 w_k1_FDX_s002
                                 + 0.703102 w_k1_IBM_s002
                                 + 1.00497 w_k1_INTC_s002
                                 + 0.702177 w_k1_KO_s002
                                 + 0.643178 w_k1_MAT_s002
                                 + 0.878685 w_k1_MSFT_s002
                                 + 0.78616 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.817037 w_k1_ADM_s002
                                 + 1.00403 w_k1_AXP_s002
                                 + 1.03576 w_k1_DIS_s002
                                 + 1.18082 w_k1_FDX_s002
                                 + 0.856599 w_k1_IBM_s002
                                 + 1.46328 w_k1_INTC_s002
                                 + 1.01416 w_k1_KO_s002 + 0.80384 w_k1_MAT_s002
                                 + 1.23778 w_k1_MSFT_s002
                                 + 0.970427 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.74781 w_k1_ADM_s002
                                 + 0.835808 w_k1_AXP_s002
                                 + 0.829662 w_k1_DIS_s002
                                 + 0.91196 w_k1_FDX_s002
                                 + 0.998352 w_k1_IBM_s002
                                 + 1.1276 w_k1_INTC_s002
                                 + 0.854705 w_k1_KO_s002
                                 + 0.902318 w_k1_MAT_s002
                                 + 0.959667 w_k1_MSFT_s002
                                 + 0.776381 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.941003 w_k1_ADM_s002
                                 + 0.99962 w_k1_AXP_s002
                                 + 0.742322 w_k1_DIS_s002
                                 + 0.98917 w_k1_FDX_s002
                                 + 0.902472 w_k1_IBM_s002
                                 + 1.15272 w_k1_INTC_s002
                                 + 0.867422 w_k1_KO_s002
                                 + 0.700541 w_k1_MAT_s002
                                 + 0.971792 w_k1_MSFT_s002 + 1.015 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.14272 w_k1_ADM_s003
                                 + 1.1822 w_k1_AXP_s003 + 1.23642 w_k1_DIS_s003
                                 + 1.07121 w_k1_FDX_s003
                                 + 1.38303 w_k1_IBM_s003
                                 + 1.04096 w_k1_INTC_s003
                                 + 1.20374 w_k1_KO_s003 + 1.44814 w_k1_MAT_s003
                                 + 0.90491 w_k1_MSFT_s003
                                 + 1.28122 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.910322 w_k1_ADM_s003
                                 + 0.92934 w_k1_AXP_s003
                                 + 0.930105 w_k1_DIS_s003
                                 + 0.987643 w_k1_FDX_s003
                                 + 0.720507 w_k1_IBM_s003
                                 + 0.948873 w_k1_INTC_s003
                                 + 0.848418 w_k1_KO_s003
                                 + 0.86454 w_k1_MAT_s003
                                 + 1.06996 w_k1_MSFT_s003
                                 + 0.957715 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.62629 w_k1_ADM_s003
                                 + 1.46554 w_k1_AXP_s003
                                 + 1.56659 w_k1_DIS_s003
                                 + 1.39673 w_k1_FDX_s003
                                 + 1.21754 w_k1_IBM_s003
                                 + 1.26213 w_k1_INTC_s003 + 1.2794 w_k1_KO_s003
                                 + 1.27388 w_k1_MAT_s003
                                 + 1.24542 w_k1_MSFT_s003
                                 + 1.31426 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.902867 w_k1_ADM_s003
                                 + 0.842164 w_k1_AXP_s003
                                 + 0.779292 w_k1_DIS_s003
                                 + 0.755159 w_k1_FDX_s003
                                 + 1.43742 w_k1_IBM_s003
                                 + 0.928979 w_k1_INTC_s003
                                 + 1.05189 w_k1_KO_s003 + 1.27925 w_k1_MAT_s003
                                 + 0.857027 w_k1_MSFT_s003
                                 + 0.988929 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.826927 w_k1_ADM_s003
                                 + 0.734706 w_k1_AXP_s003
                                 + 0.596327 w_k1_DIS_s003
                                 + 0.746819 w_k1_FDX_s003
                                 + 0.733305 w_k1_IBM_s003
                                 + 0.839299 w_k1_INTC_s003
                                 + 0.66189 w_k1_KO_s003
                                 + 0.491212 w_k1_MAT_s003
                                 + 0.741301 w_k1_MSFT_s003
                                 + 0.808103 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.12107 w_k1_ADM_s003
                                 + 0.750816 w_k1_AXP_s003
                                 + 1.07337 w_k1_DIS_s003
                                 + 0.912191 w_k1_FDX_s003
                                 + 0.922192 w_k1_IBM_s003
                                 + 0.900711 w_k1_INTC_s003 + 1.035 w_k1_KO_s003
                                 + 0.915897 w_k1_MAT_s003
                                 + 1.10549 w_k1_MSFT_s003
                                 + 1.08626 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.04247 w_k1_ADM_s003
                                 + 1.08829 w_k1_AXP_s003
                                 + 1.27451 w_k1_DIS_s003
                                 + 1.09144 w_k1_FDX_s003
                                 + 0.927219 w_k1_IBM_s003
                                 + 1.01345 w_k1_INTC_s003
                                 + 1.06651 w_k1_KO_s003
                                 + 0.946272 w_k1_MAT_s003
                                 + 1.11135 w_k1_MSFT_s003
                                 + 1.10288 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.818707 w_k1_ADM_s003
                                 + 0.829036 w_k1_AXP_s003
                                 + 0.84754 w_k1_DIS_s003
                                 + 0.81277 w_k1_FDX_s003
                                 + 1.04479 w_k1_IBM_s003
                                 + 0.81844 w_k1_INTC_s003
                                 + 0.953456 w_k1_KO_s003 + 1.0826 w_k1_MAT_s003
                                 + 0.864205 w_k1_MSFT_s003
                                 + 0.996967 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.630467 w_k1_ADM_s004
                                 + 0.763635 w_k1_AXP_s004
                                 + 0.783125 w_k1_DIS_s004
                                 + 0.766783 w_k1_FDX_s004
                                 + 0.68732 w_k1_IBM_s004
                                 + 0.656868 w_k1_INTC_s004
                                 + 0.743138 w_k1_KO_s004
                                 + 0.524976 w_k1_MAT_s004
                                 + 0.691387 w_k1_MSFT_s004
                                 + 0.776944 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.23619 w_k1_ADM_s004
                                 + 1.26604 w_k1_AXP_s004
                                 + 1.38019 w_k1_DIS_s004
                                 + 1.20182 w_k1_FDX_s004
                                 + 1.60307 w_k1_IBM_s004
                                 + 0.967581 w_k1_INTC_s004
                                 + 1.44077 w_k1_KO_s004 + 1.40258 w_k1_MAT_s004
                                 + 1.02428 w_k1_MSFT_s004
                                 + 1.30771 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.938637 w_k1_ADM_s004
                                 + 0.936014 w_k1_AXP_s004
                                 + 0.971949 w_k1_DIS_s004
                                 + 0.885109 w_k1_FDX_s004
                                 + 1.1336 w_k1_IBM_s004
                                 + 0.948969 w_k1_INTC_s004
                                 + 1.17039 w_k1_KO_s004 + 0.94204 w_k1_MAT_s004
                                 + 0.886431 w_k1_MSFT_s004
                                 + 1.03595 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.76155 w_k1_ADM_s004
                                 + 0.711714 w_k1_AXP_s004
                                 + 0.567678 w_k1_DIS_s004
                                 + 0.718491 w_k1_FDX_s004
                                 + 0.883986 w_k1_IBM_s004
                                 + 0.581039 w_k1_INTC_s004
                                 + 0.765052 w_k1_KO_s004
                                 + 0.63497 w_k1_MAT_s004
                                 + 0.696528 w_k1_MSFT_s004
                                 + 0.747261 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.42575 w_k1_ADM_s004
                                 + 0.997873 w_k1_AXP_s004
                                 + 1.01952 w_k1_DIS_s004
                                 + 1.16062 w_k1_FDX_s004
                                 + 0.758134 w_k1_IBM_s004
                                 + 1.02662 w_k1_INTC_s004
                                 + 1.01493 w_k1_KO_s004
                                 + 0.974728 w_k1_MAT_s004
                                 + 1.07946 w_k1_MSFT_s004
                                 + 0.996374 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.32505 w_k1_ADM_s004
                                 + 1.08277 w_k1_AXP_s004 + 1.274 w_k1_DIS_s004
                                 + 1.18962 w_k1_FDX_s004
                                 + 0.986222 w_k1_IBM_s004
                                 + 1.06255 w_k1_INTC_s004
                                 + 1.13829 w_k1_KO_s004 + 1.11686 w_k1_MAT_s004
                                 + 1.06857 w_k1_MSFT_s004
                                 + 1.09187 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.09555 w_k1_ADM_s004
                                 + 0.780161 w_k1_AXP_s004
                                 + 0.781197 w_k1_DIS_s004
                                 + 0.968665 w_k1_FDX_s004
                                 + 0.612441 w_k1_IBM_s004
                                 + 0.83161 w_k1_INTC_s004
                                 + 0.806908 w_k1_KO_s004
                                 + 0.462282 w_k1_MAT_s004
                                 + 0.870105 w_k1_MSFT_s004
                                 + 0.948051 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.961166 w_k1_ADM_s004
                                 + 0.867709 w_k1_AXP_s004
                                 + 0.82595 w_k1_DIS_s004
                                 + 0.867074 w_k1_FDX_s004
                                 + 0.823071 w_k1_IBM_s004
                                 + 0.788907 w_k1_INTC_s004
                                 + 0.918185 w_k1_KO_s004
                                 + 0.695001 w_k1_MAT_s004
                                 + 0.774583 w_k1_MSFT_s004
                                 + 0.769411 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.13573 w_k1_ADM_s005
                                 + 1.13393 w_k1_AXP_s005
                                 + 1.05367 w_k1_DIS_s005
                                 + 1.01495 w_k1_FDX_s005
                                 + 1.25828 w_k1_IBM_s005
                                 + 0.989703 w_k1_INTC_s005
                                 + 1.22996 w_k1_KO_s005 + 1.24352 w_k1_MAT_s005
                                 + 1.04784 w_k1_MSFT_s005
                                 + 1.12073 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.913358 w_k1_ADM_s005
                                 + 0.745602 w_k1_AXP_s005
                                 + 0.644229 w_k1_DIS_s005
                                 + 0.754034 w_k1_FDX_s005
                                 + 0.779189 w_k1_IBM_s005
                                 + 0.781838 w_k1_INTC_s005
                                 + 0.821197 w_k1_KO_s005
                                 + 0.763973 w_k1_MAT_s005
                                 + 0.727804 w_k1_MSFT_s005
                                 + 0.937483 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.0726 w_k1_ADM_s005
                                 + 0.993154 w_k1_AXP_s005
                                 + 0.681298 w_k1_DIS_s005
                                 + 0.755677 w_k1_FDX_s005
                                 + 1.08191 w_k1_IBM_s005
                                 + 0.799763 w_k1_INTC_s005
                                 + 0.879531 w_k1_KO_s005
                                 + 1.20183 w_k1_MAT_s005
                                 + 0.763314 w_k1_MSFT_s005
                                 + 0.873496 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.34696 w_k1_ADM_s005
                                 + 1.6003 w_k1_AXP_s005 + 1.49241 w_k1_DIS_s005
                                 + 1.28344 w_k1_FDX_s005
                                 + 1.26184 w_k1_IBM_s005
                                 + 1.24861 w_k1_INTC_s005
                                 + 1.34762 w_k1_KO_s005 + 1.5487 w_k1_MAT_s005
                                 + 1.3779 w_k1_MSFT_s005 + 1.35194 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.21471 w_k1_ADM_s005
                                 + 1.07513 w_k1_AXP_s005
                                 + 1.29599 w_k1_DIS_s005
                                 + 1.20398 w_k1_FDX_s005
                                 + 0.869227 w_k1_IBM_s005
                                 + 1.22018 w_k1_INTC_s005
                                 + 1.07282 w_k1_KO_s005
                                 + 0.879231 w_k1_MAT_s005
                                 + 1.2082 w_k1_MSFT_s005 + 1.05816 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.11555 w_k1_ADM_s005
                                 + 0.996034 w_k1_AXP_s005
                                 + 0.910427 w_k1_DIS_s005
                                 + 0.901134 w_k1_FDX_s005
                                 + 0.874851 w_k1_IBM_s005
                                 + 1.13429 w_k1_INTC_s005
                                 + 0.924256 w_k1_KO_s005
                                 + 0.729766 w_k1_MAT_s005
                                 + 0.96022 w_k1_MSFT_s005
                                 + 0.935211 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.07283 w_k1_ADM_s005
                                 + 0.99413 w_k1_AXP_s005
                                 + 0.889954 w_k1_DIS_s005
                                 + 0.953997 w_k1_FDX_s005
                                 + 0.690514 w_k1_IBM_s005
                                 + 1.09027 w_k1_INTC_s005
                                 + 0.796531 w_k1_KO_s005
                                 + 0.504644 w_k1_MAT_s005
                                 + 1.00063 w_k1_MSFT_s005
                                 + 0.919906 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.883508 w_k1_ADM_s005
                                 + 1.02373 w_k1_AXP_s005
                                 + 0.734851 w_k1_DIS_s005
                                 + 0.785292 w_k1_FDX_s005
                                 + 1.42014 w_k1_IBM_s005
                                 + 0.768683 w_k1_INTC_s005
                                 + 0.887955 w_k1_KO_s005
                                 + 1.19361 w_k1_MAT_s005
                                 + 0.804764 w_k1_MSFT_s005
                                 + 0.868508 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.30769 w_k1_ADM_s006
                                 + 1.52259 w_k1_AXP_s006
                                 + 1.60286 w_k1_DIS_s006
                                 + 1.28664 w_k1_FDX_s006
                                 + 1.67982 w_k1_IBM_s006
                                 + 1.16239 w_k1_INTC_s006
                                 + 1.45306 w_k1_KO_s006 + 1.57094 w_k1_MAT_s006
                                 + 1.32035 w_k1_MSFT_s006 + 1.3308 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.1532 w_k1_ADM_s006
                                 + 1.29679 w_k1_AXP_s006 + 1.1498 w_k1_DIS_s006
                                 + 1.03514 w_k1_FDX_s006
                                 + 1.27913 w_k1_IBM_s006
                                 + 0.997137 w_k1_INTC_s006
                                 + 1.10533 w_k1_KO_s006 + 1.42805 w_k1_MAT_s006
                                 + 1.11561 w_k1_MSFT_s006
                                 + 1.09046 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.983152 w_k1_ADM_s006
                                 + 1.14739 w_k1_AXP_s006
                                 + 0.924957 w_k1_DIS_s006
                                 + 0.817672 w_k1_FDX_s006
                                 + 1.77556 w_k1_IBM_s006
                                 + 0.815996 w_k1_INTC_s006
                                 + 1.2147 w_k1_KO_s006 + 2.22063 w_k1_MAT_s006
                                 + 0.908545 w_k1_MSFT_s006
                                 + 0.949171 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.24921 w_k1_ADM_s006
                                 + 1.13199 w_k1_AXP_s006
                                 + 1.10642 w_k1_DIS_s006
                                 + 1.22892 w_k1_FDX_s006
                                 + 0.802097 w_k1_IBM_s006
                                 + 0.989075 w_k1_INTC_s006
                                 + 1.07246 w_k1_KO_s006 + 1.01503 w_k1_MAT_s006
                                 + 1.14282 w_k1_MSFT_s006
                                 + 1.08817 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.949116 w_k1_ADM_s006
                                 + 0.710675 w_k1_AXP_s006
                                 + 0.716781 w_k1_DIS_s006
                                 + 0.769527 w_k1_FDX_s006
                                 + 0.869399 w_k1_IBM_s006
                                 + 0.790181 w_k1_INTC_s006
                                 + 0.851086 w_k1_KO_s006
                                 + 0.667282 w_k1_MAT_s006
                                 + 0.880554 w_k1_MSFT_s006
                                 + 0.804696 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.0041 w_k1_ADM_s006
                                 + 0.905835 w_k1_AXP_s006
                                 + 0.845044 w_k1_DIS_s006
                                 + 0.987018 w_k1_FDX_s006
                                 + 0.568279 w_k1_IBM_s006
                                 + 0.951744 w_k1_INTC_s006
                                 + 0.825136 w_k1_KO_s006
                                 + 0.633527 w_k1_MAT_s006
                                 + 1.04793 w_k1_MSFT_s006
                                 + 0.931413 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.62547 w_k1_ADM_s006
                                 + 1.46121 w_k1_AXP_s006
                                 + 1.42969 w_k1_DIS_s006
                                 + 1.40405 w_k1_FDX_s006
                                 + 1.11931 w_k1_IBM_s006
                                 + 1.20602 w_k1_INTC_s006
                                 + 1.04291 w_k1_KO_s006 + 1.10768 w_k1_MAT_s006
                                 + 1.28165 w_k1_MSFT_s006
                                 + 1.13675 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.998625 w_k1_ADM_s006
                                 + 1.22715 w_k1_AXP_s006
                                 + 0.859332 w_k1_DIS_s006
                                 + 0.908931 w_k1_FDX_s006
                                 + 1.01293 w_k1_IBM_s006
                                 + 0.984808 w_k1_INTC_s006
                                 + 0.9385 w_k1_KO_s006 + 1.13733 w_k1_MAT_s006
                                 + 0.92999 w_k1_MSFT_s006
                                 + 0.970479 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.869561 w_k1_ADM_s007
                                 + 0.920292 w_k1_AXP_s007
                                 + 0.963937 w_k1_DIS_s007
                                 + 0.877256 w_k1_FDX_s007
                                 + 1.14616 w_k1_IBM_s007
                                 + 0.844624 w_k1_INTC_s007
                                 + 0.99041 w_k1_KO_s007 + 1.03713 w_k1_MAT_s007
                                 + 0.932037 w_k1_MSFT_s007
                                 + 0.831483 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.879884 w_k1_ADM_s007
                                 + 1.4823 w_k1_AXP_s007 + 1.346 w_k1_DIS_s007
                                 + 1.38743 w_k1_FDX_s007
                                 + 1.12512 w_k1_IBM_s007
                                 + 1.41309 w_k1_INTC_s007
                                 + 1.18787 w_k1_KO_s007 + 1.54749 w_k1_MAT_s007
                                 + 1.30457 w_k1_MSFT_s007
                                 + 1.10865 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.21836 w_k1_ADM_s007
                                 + 1.08889 w_k1_AXP_s007
                                 + 1.09742 w_k1_DIS_s007 + 1.1181 w_k1_FDX_s007
                                 + 0.826416 w_k1_IBM_s007
                                 + 0.916977 w_k1_INTC_s007
                                 + 0.982138 w_k1_KO_s007
                                 + 0.833308 w_k1_MAT_s007
                                 + 1.03279 w_k1_MSFT_s007
                                 + 0.946314 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.1897 w_k1_ADM_s007
                                 + 0.995106 w_k1_AXP_s007
                                 + 1.38302 w_k1_DIS_s007
                                 + 1.16708 w_k1_FDX_s007
                                 + 1.08147 w_k1_IBM_s007
                                 + 1.00659 w_k1_INTC_s007
                                 + 1.28997 w_k1_KO_s007 + 1.35093 w_k1_MAT_s007
                                 + 1.06969 w_k1_MSFT_s007
                                 + 1.11124 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.16931 w_k1_ADM_s007
                                 + 1.36872 w_k1_AXP_s007
                                 + 1.73651 w_k1_DIS_s007
                                 + 1.38844 w_k1_FDX_s007
                                 + 1.52872 w_k1_IBM_s007
                                 + 1.14565 w_k1_INTC_s007
                                 + 1.52445 w_k1_KO_s007 + 1.61433 w_k1_MAT_s007
                                 + 1.23528 w_k1_MSFT_s007
                                 + 1.18882 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.889248 w_k1_ADM_s007
                                 + 0.820554 w_k1_AXP_s007
                                 + 0.869881 w_k1_DIS_s007
                                 + 0.883862 w_k1_FDX_s007
                                 + 0.90142 w_k1_IBM_s007
                                 + 0.947277 w_k1_INTC_s007
                                 + 0.926361 w_k1_KO_s007
                                 + 1.26027 w_k1_MAT_s007
                                 + 0.925299 w_k1_MSFT_s007
                                 + 0.919274 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.619376 w_k1_ADM_s007
                                 + 0.708931 w_k1_AXP_s007
                                 + 0.621074 w_k1_DIS_s007
                                 + 0.729805 w_k1_FDX_s007
                                 + 0.695397 w_k1_IBM_s007
                                 + 0.866233 w_k1_INTC_s007
                                 + 0.678003 w_k1_KO_s007
                                 + 0.693954 w_k1_MAT_s007
                                 + 0.757602 w_k1_MSFT_s007
                                 + 0.757029 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.916116 w_k1_ADM_s007
                                 + 1.12047 w_k1_AXP_s007
                                 + 0.757092 w_k1_DIS_s007
                                 + 0.81155 w_k1_FDX_s007
                                 + 1.28941 w_k1_IBM_s007
                                 + 0.784718 w_k1_INTC_s007
                                 + 0.939799 w_k1_KO_s007
                                 + 1.40723 w_k1_MAT_s007
                                 + 0.714895 w_k1_MSFT_s007
                                 + 0.993078 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.972299 w_k1_ADM_s010
                                 + 0.959764 w_k1_AXP_s010
                                 + 1.09442 w_k1_DIS_s010
                                 + 1.04135 w_k1_FDX_s010
                                 + 0.941694 w_k1_IBM_s010
                                 + 0.998671 w_k1_INTC_s010
                                 + 1.0249 w_k1_KO_s010 + 0.909431 w_k1_MAT_s010
                                 + 1.05984 w_k1_MSFT_s010
                                 + 1.02833 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.656113 w_k1_ADM_s010
                                 + 0.668005 w_k1_AXP_s010
                                 + 0.659678 w_k1_DIS_s010
                                 + 0.762265 w_k1_FDX_s010
                                 + 0.701518 w_k1_IBM_s010
                                 + 0.755284 w_k1_INTC_s010
                                 + 0.739268 w_k1_KO_s010
                                 + 0.738447 w_k1_MAT_s010
                                 + 0.706251 w_k1_MSFT_s010
                                 + 0.856725 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.898269 w_k1_ADM_s010
                                 + 1.1124 w_k1_AXP_s010 + 1.13958 w_k1_DIS_s010
                                 + 1.12512 w_k1_FDX_s010
                                 + 1.19926 w_k1_IBM_s010
                                 + 1.12165 w_k1_INTC_s010
                                 + 1.11147 w_k1_KO_s010 + 1.50955 w_k1_MAT_s010
                                 + 1.19684 w_k1_MSFT_s010
                                 + 1.02617 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.696459 w_k1_ADM_s010
                                 + 0.790925 w_k1_AXP_s010
                                 + 0.763717 w_k1_DIS_s010
                                 + 0.791641 w_k1_FDX_s010
                                 + 0.91669 w_k1_IBM_s010
                                 + 0.728403 w_k1_INTC_s010
                                 + 0.827012 w_k1_KO_s010
                                 + 0.943589 w_k1_MAT_s010
                                 + 0.799298 w_k1_MSFT_s010
                                 + 0.918249 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.79597 w_k1_ADM_s010
                                 + 1.12502 w_k1_AXP_s010
                                 + 0.985411 w_k1_DIS_s010
                                 + 1.08466 w_k1_FDX_s010
                                 + 0.721178 w_k1_IBM_s010
                                 + 0.88803 w_k1_INTC_s010
                                 + 0.894858 w_k1_KO_s010
                                 + 0.982438 w_k1_MAT_s010
                                 + 0.946002 w_k1_MSFT_s010
                                 + 1.03133 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.952198 w_k1_ADM_s010
                                 + 1.15294 w_k1_AXP_s010
                                 + 1.44338 w_k1_DIS_s010
                                 + 1.06358 w_k1_FDX_s010
                                 + 1.43495 w_k1_IBM_s010
                                 + 1.02366 w_k1_INTC_s010
                                 + 1.39145 w_k1_KO_s010 + 1.98457 w_k1_MAT_s010
                                 + 1.2192 w_k1_MSFT_s010 + 1.14456 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.770576 w_k1_ADM_s010
                                 + 0.733629 w_k1_AXP_s010
                                 + 0.890873 w_k1_DIS_s010
                                 + 0.893358 w_k1_FDX_s010
                                 + 0.860502 w_k1_IBM_s010
                                 + 0.774245 w_k1_INTC_s010
                                 + 0.874265 w_k1_KO_s010
                                 + 0.765165 w_k1_MAT_s010
                                 + 0.94765 w_k1_MSFT_s010
                                 + 0.86927 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.13703 w_k1_ADM_s010
                                 + 1.46792 w_k1_AXP_s010
                                 + 1.44313 w_k1_DIS_s010
                                 + 1.39925 w_k1_FDX_s010
                                 + 0.938987 w_k1_IBM_s010
                                 + 0.89973 w_k1_INTC_s010
                                 + 1.18476 w_k1_KO_s010 + 1.33279 w_k1_MAT_s010
                                 + 1.16809 w_k1_MSFT_s010
                                 + 1.23509 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.32974 w_k1_ADM_s011
                                 + 1.4112 w_k1_AXP_s011 + 1.60666 w_k1_DIS_s011
                                 + 1.31774 w_k1_FDX_s011
                                 + 1.27162 w_k1_IBM_s011
                                 + 1.19867 w_k1_INTC_s011
                                 + 1.30079 w_k1_KO_s011
                                 + 0.940947 w_k1_MAT_s011
                                 + 1.25671 w_k1_MSFT_s011
                                 + 1.15237 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.00698 w_k1_ADM_s011
                                 + 0.823326 w_k1_AXP_s011
                                 + 0.939783 w_k1_DIS_s011
                                 + 0.953013 w_k1_FDX_s011
                                 + 0.871634 w_k1_IBM_s011
                                 + 1.10436 w_k1_INTC_s011
                                 + 1.03083 w_k1_KO_s011
                                 + 0.623528 w_k1_MAT_s011
                                 + 1.06366 w_k1_MSFT_s011
                                 + 1.00835 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.853876 w_k1_ADM_s011
                                 + 0.944215 w_k1_AXP_s011
                                 + 0.603544 w_k1_DIS_s011
                                 + 0.777855 w_k1_FDX_s011
                                 + 0.929763 w_k1_IBM_s011
                                 + 0.792097 w_k1_INTC_s011
                                 + 0.801763 w_k1_KO_s011
                                 + 0.976571 w_k1_MAT_s011
                                 + 0.819898 w_k1_MSFT_s011
                                 + 1.03527 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.32437 w_k1_ADM_s011
                                 + 1.25986 w_k1_AXP_s011
                                 + 1.06834 w_k1_DIS_s011
                                 + 1.03452 w_k1_FDX_s011 + 1.5624 w_k1_IBM_s011
                                 + 1.24352 w_k1_INTC_s011
                                 + 1.24198 w_k1_KO_s011 + 1.15591 w_k1_MAT_s011
                                 + 1.10421 w_k1_MSFT_s011
                                 + 1.27591 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.938394 w_k1_ADM_s011
                                 + 0.993172 w_k1_AXP_s011
                                 + 1.39702 w_k1_DIS_s011
                                 + 1.29495 w_k1_FDX_s011
                                 + 0.806364 w_k1_IBM_s011
                                 + 1.23416 w_k1_INTC_s011 + 1.1636 w_k1_KO_s011
                                 + 0.676848 w_k1_MAT_s011
                                 + 1.40692 w_k1_MSFT_s011
                                 + 1.14809 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.937225 w_k1_ADM_s011
                                 + 0.668716 w_k1_AXP_s011
                                 + 0.833009 w_k1_DIS_s011
                                 + 0.738816 w_k1_FDX_s011
                                 + 0.975839 w_k1_IBM_s011
                                 + 0.948563 w_k1_INTC_s011
                                 + 1.01414 w_k1_KO_s011
                                 + 0.784349 w_k1_MAT_s011
                                 + 1.01716 w_k1_MSFT_s011
                                 + 1.00446 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.77275 w_k1_ADM_s011
                                 + 0.71734 w_k1_AXP_s011
                                 + 0.857767 w_k1_DIS_s011
                                 + 0.801906 w_k1_FDX_s011
                                 + 1.32968 w_k1_IBM_s011
                                 + 0.863374 w_k1_INTC_s011
                                 + 1.10953 w_k1_KO_s011 + 1.19595 w_k1_MAT_s011
                                 + 0.974754 w_k1_MSFT_s011
                                 + 1.04837 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.16095 w_k1_ADM_s011
                                 + 1.2097 w_k1_AXP_s011 + 1.02262 w_k1_DIS_s011
                                 + 1.16962 w_k1_FDX_s011
                                 + 0.922627 w_k1_IBM_s011
                                 + 1.1815 w_k1_INTC_s011 + 1.00122 w_k1_KO_s011
                                 + 0.877642 w_k1_MAT_s011
                                 + 1.1085 w_k1_MSFT_s011 + 1.11967 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.07842 w_k1_ADM_s012
                                 + 1.08502 w_k1_AXP_s012
                                 + 1.19342 w_k1_DIS_s012
                                 + 1.11423 w_k1_FDX_s012
                                 + 1.17064 w_k1_IBM_s012
                                 + 1.21215 w_k1_INTC_s012
                                 + 1.03238 w_k1_KO_s012
                                 + 0.975612 w_k1_MAT_s012
                                 + 1.1076 w_k1_MSFT_s012 + 1.31187 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.723798 w_k1_ADM_s012
                                 + 0.816448 w_k1_AXP_s012
                                 + 1.15396 w_k1_DIS_s012
                                 + 0.949538 w_k1_FDX_s012
                                 + 0.98357 w_k1_IBM_s012
                                 + 0.907592 w_k1_INTC_s012
                                 + 1.00394 w_k1_KO_s012 + 1.04531 w_k1_MAT_s012
                                 + 1.04872 w_k1_MSFT_s012
                                 + 1.09781 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.31605 w_k1_ADM_s012
                                 + 1.03031 w_k1_AXP_s012
                                 + 1.16994 w_k1_DIS_s012
                                 + 1.22853 w_k1_FDX_s012
                                 + 0.561334 w_k1_IBM_s012
                                 + 0.904392 w_k1_INTC_s012
                                 + 0.815386 w_k1_KO_s012
                                 + 0.750412 w_k1_MAT_s012
                                 + 1.12607 w_k1_MSFT_s012
                                 + 1.08212 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.81565 w_k1_ADM_s012
                                 + 1.06649 w_k1_AXP_s012
                                 + 1.19122 w_k1_DIS_s012
                                 + 0.848667 w_k1_FDX_s012
                                 + 1.95264 w_k1_IBM_s012
                                 + 0.85358 w_k1_INTC_s012
                                 + 1.15765 w_k1_KO_s012 + 1.95591 w_k1_MAT_s012
                                 + 0.812552 w_k1_MSFT_s012
                                 + 1.12587 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.86413 w_k1_ADM_s012
                                 + 1.12433 w_k1_AXP_s012
                                 + 0.793834 w_k1_DIS_s012
                                 + 0.837978 w_k1_FDX_s012
                                 + 1.24656 w_k1_IBM_s012
                                 + 0.660266 w_k1_INTC_s012
                                 + 0.885993 w_k1_KO_s012
                                 + 1.54991 w_k1_MAT_s012
                                 + 0.657216 w_k1_MSFT_s012
                                 + 0.829201 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.42625 w_k1_ADM_s012
                                 + 1.55331 w_k1_AXP_s012
                                 + 2.05235 w_k1_DIS_s012
                                 + 1.45629 w_k1_FDX_s012
                                 + 1.27331 w_k1_IBM_s012
                                 + 1.15375 w_k1_INTC_s012
                                 + 1.29413 w_k1_KO_s012 + 1.5712 w_k1_MAT_s012
                                 + 1.27349 w_k1_MSFT_s012
                                 + 1.33239 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.827364 w_k1_ADM_s012
                                 + 0.730963 w_k1_AXP_s012
                                 + 0.705009 w_k1_DIS_s012
                                 + 0.752366 w_k1_FDX_s012
                                 + 0.778647 w_k1_IBM_s012
                                 + 0.792848 w_k1_INTC_s012
                                 + 0.709013 w_k1_KO_s012
                                 + 0.66154 w_k1_MAT_s012
                                 + 0.844497 w_k1_MSFT_s012
                                 + 0.918947 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.867345 w_k1_ADM_s012
                                 + 1.57027 w_k1_AXP_s012
                                 + 1.22607 w_k1_DIS_s012
                                 + 1.36536 w_k1_FDX_s012
                                 + 0.93644 w_k1_IBM_s012
                                 + 1.51453 w_k1_INTC_s012
                                 + 1.04283 w_k1_KO_s012 + 1.23387 w_k1_MAT_s012
                                 + 1.15006 w_k1_MSFT_s012
                                 + 1.41649 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.942756 w_k1_ADM_s013
                                 + 0.805669 w_k1_AXP_s013
                                 + 0.844972 w_k1_DIS_s013
                                 + 0.932645 w_k1_FDX_s013
                                 + 0.819187 w_k1_IBM_s013
                                 + 1.19839 w_k1_INTC_s013
                                 + 0.941342 w_k1_KO_s013
                                 + 0.826399 w_k1_MAT_s013
                                 + 1.07671 w_k1_MSFT_s013
                                 + 0.952608 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.844552 w_k1_ADM_s013
                                 + 1.03296 w_k1_AXP_s013
                                 + 0.944589 w_k1_DIS_s013
                                 + 0.982397 w_k1_FDX_s013
                                 + 1.2529 w_k1_IBM_s013
                                 + 0.99617 w_k1_INTC_s013
                                 + 1.19257 w_k1_KO_s013 + 1.56023 w_k1_MAT_s013
                                 + 1.03448 w_k1_MSFT_s013
                                 + 1.00519 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.917049 w_k1_ADM_s013
                                 + 0.716549 w_k1_AXP_s013
                                 + 0.574922 w_k1_DIS_s013
                                 + 0.791647 w_k1_FDX_s013
                                 + 0.605717 w_k1_IBM_s013
                                 + 0.739841 w_k1_INTC_s013
                                 + 0.649627 w_k1_KO_s013
                                 + 0.773092 w_k1_MAT_s013
                                 + 0.807807 w_k1_MSFT_s013
                                 + 0.817255 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.938815 w_k1_ADM_s013
                                 + 0.787258 w_k1_AXP_s013
                                 + 0.659456 w_k1_DIS_s013
                                 + 0.808933 w_k1_FDX_s013
                                 + 1.00055 w_k1_IBM_s013
                                 + 0.965933 w_k1_INTC_s013
                                 + 0.848234 w_k1_KO_s013
                                 + 0.923949 w_k1_MAT_s013
                                 + 0.895365 w_k1_MSFT_s013
                                 + 0.847885 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.42385 w_k1_ADM_s013
                                 + 1.59706 w_k1_AXP_s013
                                 + 1.66571 w_k1_DIS_s013
                                 + 1.40446 w_k1_FDX_s013
                                 + 1.31241 w_k1_IBM_s013
                                 + 1.12371 w_k1_INTC_s013
                                 + 1.46262 w_k1_KO_s013 + 1.9792 w_k1_MAT_s013
                                 + 1.33706 w_k1_MSFT_s013
                                 + 1.47476 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.39001 w_k1_ADM_s013
                                 + 1.19697 w_k1_AXP_s013
                                 + 1.30046 w_k1_DIS_s013
                                 + 0.981404 w_k1_FDX_s013
                                 + 1.57327 w_k1_IBM_s013
                                 + 1.12878 w_k1_INTC_s013
                                 + 1.53974 w_k1_KO_s013 + 1.60405 w_k1_MAT_s013
                                 + 1.1397 w_k1_MSFT_s013 + 1.26898 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.14757 w_k1_ADM_s013
                                 + 1.17822 w_k1_AXP_s013
                                 + 1.12295 w_k1_DIS_s013
                                 + 1.21103 w_k1_FDX_s013
                                 + 0.652154 w_k1_IBM_s013
                                 + 1.21763 w_k1_INTC_s013
                                 + 0.949977 w_k1_KO_s013
                                 + 1.02575 w_k1_MAT_s013
                                 + 1.25523 w_k1_MSFT_s013
                                 + 1.09924 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.23104 w_k1_ADM_s013
                                 + 1.29515 w_k1_AXP_s013
                                 + 1.27817 w_k1_DIS_s013
                                 + 1.24565 w_k1_FDX_s013
                                 + 0.954865 w_k1_IBM_s013
                                 + 1.29262 w_k1_INTC_s013 + 1.13 w_k1_KO_s013
                                 + 1.2484 w_k1_MAT_s013
                                 + 1.20279 w_k1_MSFT_s013
                                 + 1.07172 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.910423 w_k1_ADM_s014
                                 + 0.756233 w_k1_AXP_s014
                                 + 0.788079 w_k1_DIS_s014
                                 + 0.83796 w_k1_FDX_s014
                                 + 0.956814 w_k1_IBM_s014
                                 + 0.843971 w_k1_INTC_s014
                                 + 0.759171 w_k1_KO_s014
                                 + 0.635118 w_k1_MAT_s014
                                 + 0.933607 w_k1_MSFT_s014
                                 + 0.856584 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.35242 w_k1_ADM_s014
                                 + 1.51141 w_k1_AXP_s014
                                 + 1.47741 w_k1_DIS_s014
                                 + 1.30949 w_k1_FDX_s014
                                 + 1.31512 w_k1_IBM_s014
                                 + 1.05577 w_k1_INTC_s014
                                 + 1.26416 w_k1_KO_s014 + 1.72387 w_k1_MAT_s014
                                 + 1.27945 w_k1_MSFT_s014
                                 + 1.24733 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.9541 w_k1_ADM_s014
                                 + 1.17337 w_k1_AXP_s014
                                 + 0.924111 w_k1_DIS_s014
                                 + 1.08588 w_k1_FDX_s014
                                 + 0.756018 w_k1_IBM_s014
                                 + 0.90158 w_k1_INTC_s014
                                 + 0.729353 w_k1_KO_s014
                                 + 1.28551 w_k1_MAT_s014
                                 + 0.926625 w_k1_MSFT_s014
                                 + 0.969318 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.12483 w_k1_ADM_s014
                                 + 1.13208 w_k1_AXP_s014
                                 + 1.23952 w_k1_DIS_s014 + 1.2897 w_k1_FDX_s014
                                 + 0.898578 w_k1_IBM_s014
                                 + 1.16496 w_k1_INTC_s014
                                 + 0.922765 w_k1_KO_s014
                                 + 0.787199 w_k1_MAT_s014
                                 + 1.19081 w_k1_MSFT_s014
                                 + 1.07666 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.880574 w_k1_ADM_s014
                                 + 0.773498 w_k1_AXP_s014
                                 + 0.72937 w_k1_DIS_s014
                                 + 0.839589 w_k1_FDX_s014
                                 + 0.786994 w_k1_IBM_s014
                                 + 0.82586 w_k1_INTC_s014
                                 + 0.744333 w_k1_KO_s014
                                 + 0.79702 w_k1_MAT_s014
                                 + 0.974668 w_k1_MSFT_s014
                                 + 0.814486 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.938043 w_k1_ADM_s014
                                 + 0.95624 w_k1_AXP_s014
                                 + 1.08144 w_k1_DIS_s014
                                 + 1.09516 w_k1_FDX_s014
                                 + 0.932601 w_k1_IBM_s014
                                 + 0.979334 w_k1_INTC_s014
                                 + 0.935731 w_k1_KO_s014
                                 + 1.22905 w_k1_MAT_s014
                                 + 1.11739 w_k1_MSFT_s014
                                 + 1.04577 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.875265 w_k1_ADM_s014
                                 + 0.732813 w_k1_AXP_s014
                                 + 0.685067 w_k1_DIS_s014
                                 + 0.87138 w_k1_FDX_s014
                                 + 0.567684 w_k1_IBM_s014
                                 + 0.759654 w_k1_INTC_s014
                                 + 0.605595 w_k1_KO_s014
                                 + 0.425664 w_k1_MAT_s014
                                 + 0.86596 w_k1_MSFT_s014
                                 + 0.808946 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.983289 w_k1_ADM_s014
                                 + 1.15736 w_k1_AXP_s014
                                 + 0.989833 w_k1_DIS_s014
                                 + 0.932532 w_k1_FDX_s014
                                 + 1.3756 w_k1_IBM_s014
                                 + 0.911524 w_k1_INTC_s014
                                 + 0.912851 w_k1_KO_s014 + 1.3881 w_k1_MAT_s014
                                 + 0.876387 w_k1_MSFT_s014
                                 + 0.906134 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.06393 w_k1_ADM_s015
                                 + 0.843873 w_k1_AXP_s015
                                 + 0.628748 w_k1_DIS_s015
                                 + 0.803464 w_k1_FDX_s015
                                 + 0.722248 w_k1_IBM_s015
                                 + 0.91811 w_k1_INTC_s015
                                 + 0.768501 w_k1_KO_s015
                                 + 0.661856 w_k1_MAT_s015
                                 + 0.769516 w_k1_MSFT_s015
                                 + 0.929655 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.11823 w_k1_ADM_s015
                                 + 0.888823 w_k1_AXP_s015
                                 + 0.917607 w_k1_DIS_s015
                                 + 0.940027 w_k1_FDX_s015
                                 + 0.900704 w_k1_IBM_s015
                                 + 0.791782 w_k1_INTC_s015
                                 + 0.817462 w_k1_KO_s015 + 1.1031 w_k1_MAT_s015
                                 + 0.817073 w_k1_MSFT_s015
                                 + 0.957782 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.954349 w_k1_ADM_s015
                                 + 1.05206 w_k1_AXP_s015
                                 + 1.15993 w_k1_DIS_s015
                                 + 1.14421 w_k1_FDX_s015
                                 + 0.875193 w_k1_IBM_s015
                                 + 0.967451 w_k1_INTC_s015
                                 + 1.04089 w_k1_KO_s015 + 1.31641 w_k1_MAT_s015
                                 + 1.16492 w_k1_MSFT_s015
                                 + 1.03027 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.848615 w_k1_ADM_s015
                                 + 0.851952 w_k1_AXP_s015
                                 + 0.615263 w_k1_DIS_s015
                                 + 0.740587 w_k1_FDX_s015
                                 + 0.82204 w_k1_IBM_s015
                                 + 0.824394 w_k1_INTC_s015
                                 + 0.698397 w_k1_KO_s015
                                 + 0.872144 w_k1_MAT_s015
                                 + 0.701206 w_k1_MSFT_s015
                                 + 0.886511 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.24317 w_k1_ADM_s015
                                 + 1.36003 w_k1_AXP_s015
                                 + 0.880839 w_k1_DIS_s015
                                 + 1.02142 w_k1_FDX_s015
                                 + 1.16034 w_k1_IBM_s015
                                 + 0.781404 w_k1_INTC_s015
                                 + 0.938173 w_k1_KO_s015
                                 + 1.86708 w_k1_MAT_s015
                                 + 0.853426 w_k1_MSFT_s015
                                 + 1.01258 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.22496 w_k1_ADM_s015
                                 + 1.67395 w_k1_AXP_s015
                                 + 1.56234 w_k1_DIS_s015
                                 + 1.40116 w_k1_FDX_s015
                                 + 1.26655 w_k1_IBM_s015
                                 + 1.1225 w_k1_INTC_s015 + 1.06841 w_k1_KO_s015
                                 + 1.63581 w_k1_MAT_s015
                                 + 1.19399 w_k1_MSFT_s015
                                 + 1.17821 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.04142 w_k1_ADM_s015
                                 + 0.987793 w_k1_AXP_s015
                                 + 1.22893 w_k1_DIS_s015
                                 + 1.14497 w_k1_FDX_s015
                                 + 1.17926 w_k1_IBM_s015
                                 + 1.24242 w_k1_INTC_s015
                                 + 1.08879 w_k1_KO_s015
                                 + 0.936938 w_k1_MAT_s015
                                 + 1.1725 w_k1_MSFT_s015 + 1.05598 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.803244 w_k1_ADM_s015
                                 + 0.75845 w_k1_AXP_s015
                                 + 1.01792 w_k1_DIS_s015
                                 + 0.878736 w_k1_FDX_s015
                                 + 1.07812 w_k1_IBM_s015
                                 + 1.01617 w_k1_INTC_s015
                                 + 1.03991 w_k1_KO_s015 + 1.09009 w_k1_MAT_s015
                                 + 0.997744 w_k1_MSFT_s015
                                 + 0.947876 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.23399 w_k1_ADM_s016
                                 + 1.31451 w_k1_AXP_s016 + 1.3838 w_k1_DIS_s016
                                 + 1.19925 w_k1_FDX_s016
                                 + 1.37034 w_k1_IBM_s016
                                 + 0.969899 w_k1_INTC_s016
                                 + 1.25777 w_k1_KO_s016 + 1.29942 w_k1_MAT_s016
                                 + 0.969863 w_k1_MSFT_s016
                                 + 1.22138 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.7388 w_k1_ADM_s016
                                 + 0.704959 w_k1_AXP_s016
                                 + 0.52073 w_k1_DIS_s016
                                 + 0.718086 w_k1_FDX_s016
                                 + 0.687612 w_k1_IBM_s016
                                 + 0.766569 w_k1_INTC_s016
                                 + 0.682815 w_k1_KO_s016
                                 + 1.02502 w_k1_MAT_s016
                                 + 0.684679 w_k1_MSFT_s016
                                 + 0.768607 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.01041 w_k1_ADM_s016
                                 + 0.769549 w_k1_AXP_s016
                                 + 0.946028 w_k1_DIS_s016
                                 + 0.938197 w_k1_FDX_s016
                                 + 0.791908 w_k1_IBM_s016
                                 + 1.04296 w_k1_INTC_s016
                                 + 0.976342 w_k1_KO_s016
                                 + 0.784975 w_k1_MAT_s016
                                 + 0.937351 w_k1_MSFT_s016
                                 + 0.922215 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.22509 w_k1_ADM_s016
                                 + 1.75684 w_k1_AXP_s016
                                 + 1.22585 w_k1_DIS_s016
                                 + 1.36629 w_k1_FDX_s016
                                 + 0.951455 w_k1_IBM_s016
                                 + 1.26292 w_k1_INTC_s016
                                 + 0.985714 w_k1_KO_s016
                                 + 0.974047 w_k1_MAT_s016
                                 + 1.08805 w_k1_MSFT_s016
                                 + 0.993457 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.843323 w_k1_ADM_s016
                                 + 0.932192 w_k1_AXP_s016
                                 + 0.618999 w_k1_DIS_s016
                                 + 0.982392 w_k1_FDX_s016
                                 + 0.639502 w_k1_IBM_s016
                                 + 0.784915 w_k1_INTC_s016
                                 + 0.687644 w_k1_KO_s016
                                 + 0.828074 w_k1_MAT_s016
                                 + 0.797256 w_k1_MSFT_s016
                                 + 0.952105 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.810294 w_k1_ADM_s016
                                 + 1.01174 w_k1_AXP_s016
                                 + 0.667695 w_k1_DIS_s016
                                 + 0.864193 w_k1_FDX_s016
                                 + 0.890014 w_k1_IBM_s016
                                 + 0.880111 w_k1_INTC_s016
                                 + 0.781108 w_k1_KO_s016
                                 + 1.18606 w_k1_MAT_s016
                                 + 0.736536 w_k1_MSFT_s016
                                 + 0.78649 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.923962 w_k1_ADM_s016
                                 + 0.957582 w_k1_AXP_s016
                                 + 0.882982 w_k1_DIS_s016
                                 + 0.849015 w_k1_FDX_s016
                                 + 1.17211 w_k1_IBM_s016
                                 + 0.820249 w_k1_INTC_s016
                                 + 0.930294 w_k1_KO_s016
                                 + 1.14459 w_k1_MAT_s016
                                 + 0.856826 w_k1_MSFT_s016
                                 + 0.929475 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.03297 w_k1_ADM_s016
                                 + 1.24751 w_k1_AXP_s016
                                 + 1.01368 w_k1_DIS_s016
                                 + 1.13691 w_k1_FDX_s016
                                 + 1.00019 w_k1_IBM_s016
                                 + 0.995381 w_k1_INTC_s016
                                 + 0.984923 w_k1_KO_s016
                                 + 1.41729 w_k1_MAT_s016
                                 + 0.903964 w_k1_MSFT_s016
                                 + 1.03981 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.800876 w_k1_ADM_s017
                                 + 1.07195 w_k1_AXP_s017
                                 + 0.868932 w_k1_DIS_s017
                                 + 0.950716 w_k1_FDX_s017
                                 + 0.821922 w_k1_IBM_s017
                                 + 0.904513 w_k1_INTC_s017
                                 + 1.02167 w_k1_KO_s017 + 1.23533 w_k1_MAT_s017
                                 + 1.01809 w_k1_MSFT_s017
                                 + 1.07905 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.943287 w_k1_ADM_s017
                                 + 0.675769 w_k1_AXP_s017
                                 + 0.971417 w_k1_DIS_s017
                                 + 0.903409 w_k1_FDX_s017
                                 + 0.856472 w_k1_IBM_s017
                                 + 0.907981 w_k1_INTC_s017
                                 + 1.06258 w_k1_KO_s017
                                 + 0.719758 w_k1_MAT_s017
                                 + 1.04345 w_k1_MSFT_s017
                                 + 1.06997 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.876904 w_k1_ADM_s017
                                 + 0.776767 w_k1_AXP_s017
                                 + 0.635329 w_k1_DIS_s017
                                 + 0.805396 w_k1_FDX_s017
                                 + 0.900918 w_k1_IBM_s017
                                 + 0.882508 w_k1_INTC_s017
                                 + 0.865463 w_k1_KO_s017
                                 + 0.866983 w_k1_MAT_s017
                                 + 0.818929 w_k1_MSFT_s017
                                 + 1.02044 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.05078 w_k1_ADM_s017
                                 + 1.11273 w_k1_AXP_s017
                                 + 1.37659 w_k1_DIS_s017
                                 + 1.10801 w_k1_FDX_s017
                                 + 1.29415 w_k1_IBM_s017
                                 + 1.12063 w_k1_INTC_s017
                                 + 1.22448 w_k1_KO_s017 + 1.2996 w_k1_MAT_s017
                                 + 1.16891 w_k1_MSFT_s017
                                 + 1.11727 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.07414 w_k1_ADM_s017
                                 + 1.28164 w_k1_AXP_s017
                                 + 1.27505 w_k1_DIS_s017
                                 + 1.20672 w_k1_FDX_s017
                                 + 0.935163 w_k1_IBM_s017
                                 + 1.0771 w_k1_INTC_s017 + 1.15913 w_k1_KO_s017
                                 + 1.07597 w_k1_MAT_s017
                                 + 1.21352 w_k1_MSFT_s017
                                 + 1.15016 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.844324 w_k1_ADM_s017
                                 + 0.949041 w_k1_AXP_s017
                                 + 0.852142 w_k1_DIS_s017
                                 + 0.886221 w_k1_FDX_s017
                                 + 1.16353 w_k1_IBM_s017
                                 + 0.831925 w_k1_INTC_s017
                                 + 1.00339 w_k1_KO_s017 + 1.25141 w_k1_MAT_s017
                                 + 0.849073 w_k1_MSFT_s017
                                 + 1.00601 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.820655 w_k1_ADM_s017
                                 + 0.584725 w_k1_AXP_s017
                                 + 0.558426 w_k1_DIS_s017
                                 + 0.760035 w_k1_FDX_s017
                                 + 0.631776 w_k1_IBM_s017
                                 + 0.858393 w_k1_INTC_s017
                                 + 0.781913 w_k1_KO_s017
                                 + 0.447525 w_k1_MAT_s017
                                 + 0.805854 w_k1_MSFT_s017
                                 + 0.845493 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.992996 w_k1_ADM_s017
                                 + 1.35906 w_k1_AXP_s017
                                 + 0.89749 w_k1_DIS_s017
                                 + 1.05463 w_k1_FDX_s017
                                 + 0.795173 w_k1_IBM_s017
                                 + 0.913226 w_k1_INTC_s017
                                 + 0.860076 w_k1_KO_s017
                                 + 1.04582 w_k1_MAT_s017
                                 + 0.921178 w_k1_MSFT_s017
                                 + 0.850398 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.0822 w_k1_ADM_s100
                                 + 0.973112 w_k1_AXP_s100
                                 + 1.06474 w_k1_DIS_s100
                                 + 1.02294 w_k1_FDX_s100
                                 + 0.797977 w_k1_IBM_s100
                                 + 0.95579 w_k1_INTC_s100
                                 + 1.00143 w_k1_KO_s100
                                 + 0.662819 w_k1_MAT_s100
                                 + 1.19736 w_k1_MSFT_s100
                                 + 0.99574 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.917169 w_k1_ADM_s100
                                 + 0.909539 w_k1_AXP_s100
                                 + 0.847494 w_k1_DIS_s100
                                 + 0.764602 w_k1_FDX_s100
                                 + 1.45226 w_k1_IBM_s100
                                 + 0.890317 w_k1_INTC_s100
                                 + 1.27852 w_k1_KO_s100 + 1.06622 w_k1_MAT_s100
                                 + 0.828272 w_k1_MSFT_s100
                                 + 0.901831 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.776419 w_k1_ADM_s100
                                 + 0.702416 w_k1_AXP_s100
                                 + 0.623694 w_k1_DIS_s100
                                 + 0.721747 w_k1_FDX_s100
                                 + 0.691995 w_k1_IBM_s100
                                 + 0.601241 w_k1_INTC_s100
                                 + 0.649137 w_k1_KO_s100
                                 + 0.458777 w_k1_MAT_s100
                                 + 0.745016 w_k1_MSFT_s100
                                 + 0.777733 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.815066 w_k1_ADM_s100
                                 + 0.840197 w_k1_AXP_s100
                                 + 0.645387 w_k1_DIS_s100
                                 + 0.740227 w_k1_FDX_s100
                                 + 0.992387 w_k1_IBM_s100
                                 + 0.570428 w_k1_INTC_s100
                                 + 0.883455 w_k1_KO_s100
                                 + 0.831111 w_k1_MAT_s100
                                 + 0.651841 w_k1_MSFT_s100
                                 + 0.914561 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.27651 w_k1_ADM_s100
                                 + 1.56921 w_k1_AXP_s100
                                 + 1.66393 w_k1_DIS_s100 + 1.1968 w_k1_FDX_s100
                                 + 1.83328 w_k1_IBM_s100
                                 + 0.828868 w_k1_INTC_s100
                                 + 1.48031 w_k1_KO_s100 + 1.7096 w_k1_MAT_s100
                                 + 1.04231 w_k1_MSFT_s100
                                 + 1.18321 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.22094 w_k1_ADM_s100
                                 + 1.49095 w_k1_AXP_s100
                                 + 1.58886 w_k1_DIS_s100
                                 + 1.33649 w_k1_FDX_s100
                                 + 1.14846 w_k1_IBM_s100
                                 + 1.19672 w_k1_INTC_s100
                                 + 1.28026 w_k1_KO_s100 + 1.01115 w_k1_MAT_s100
                                 + 1.32989 w_k1_MSFT_s100 + 1.226 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.944137 w_k1_ADM_s100
                                 + 0.890478 w_k1_AXP_s100
                                 + 0.95364 w_k1_DIS_s100
                                 + 0.930399 w_k1_FDX_s100
                                 + 0.972619 w_k1_IBM_s100
                                 + 0.845825 w_k1_INTC_s100
                                 + 1.01306 w_k1_KO_s100
                                 + 0.580586 w_k1_MAT_s100
                                 + 0.984022 w_k1_MSFT_s100
                                 + 0.975231 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.14552 w_k1_ADM_s100
                                 + 1.33278 w_k1_AXP_s100
                                 + 0.929333 w_k1_DIS_s100
                                 + 1.08486 w_k1_FDX_s100
                                 + 1.06862 w_k1_IBM_s100
                                 + 0.880525 w_k1_INTC_s100
                                 + 1.08611 w_k1_KO_s100 + 1.01552 w_k1_MAT_s100
                                 + 0.920342 w_k1_MSFT_s100
                                 + 1.02585 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.06718 w_k1_ADM_s101
                                 + 1.33717 w_k1_AXP_s101
                                 + 1.48342 w_k1_DIS_s101
                                 + 1.22521 w_k1_FDX_s101
                                 + 1.49215 w_k1_IBM_s101
                                 + 1.21242 w_k1_INTC_s101
                                 + 1.33102 w_k1_KO_s101 + 2.07391 w_k1_MAT_s101
                                 + 1.25276 w_k1_MSFT_s101
                                 + 1.18698 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.826179 w_k1_ADM_s101
                                 + 0.853601 w_k1_AXP_s101
                                 + 0.93627 w_k1_DIS_s101
                                 + 0.942112 w_k1_FDX_s101
                                 + 0.94085 w_k1_IBM_s101
                                 + 1.04765 w_k1_INTC_s101
                                 + 0.944419 w_k1_KO_s101
                                 + 0.794581 w_k1_MAT_s101
                                 + 1.05618 w_k1_MSFT_s101
                                 + 0.870815 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.03151 w_k1_ADM_s101
                                 + 0.825855 w_k1_AXP_s101
                                 + 0.744567 w_k1_DIS_s101
                                 + 0.904713 w_k1_FDX_s101
                                 + 0.800773 w_k1_IBM_s101
                                 + 0.791827 w_k1_INTC_s101
                                 + 0.784682 w_k1_KO_s101
                                 + 0.939006 w_k1_MAT_s101
                                 + 0.871333 w_k1_MSFT_s101
                                 + 0.80322 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.70609 w_k1_ADM_s101
                                 + 0.677017 w_k1_AXP_s101
                                 + 0.666482 w_k1_DIS_s101
                                 + 0.729362 w_k1_FDX_s101
                                 + 0.753997 w_k1_IBM_s101
                                 + 0.81949 w_k1_INTC_s101
                                 + 0.685543 w_k1_KO_s101
                                 + 0.714763 w_k1_MAT_s101
                                 + 0.844283 w_k1_MSFT_s101
                                 + 0.731053 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.89273 w_k1_ADM_s101
                                 + 0.840181 w_k1_AXP_s101
                                 + 0.75014 w_k1_DIS_s101
                                 + 0.961966 w_k1_FDX_s101
                                 + 0.652783 w_k1_IBM_s101
                                 + 0.922125 w_k1_INTC_s101
                                 + 0.668005 w_k1_KO_s101
                                 + 0.732389 w_k1_MAT_s101
                                 + 0.905976 w_k1_MSFT_s101
                                 + 0.765244 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.896318 w_k1_ADM_s101
                                 + 0.722019 w_k1_AXP_s101
                                 + 0.781465 w_k1_DIS_s101
                                 + 0.822302 w_k1_FDX_s101
                                 + 1.2052 w_k1_IBM_s101
                                 + 1.07829 w_k1_INTC_s101
                                 + 1.00069 w_k1_KO_s101 + 1.11495 w_k1_MAT_s101
                                 + 0.978535 w_k1_MSFT_s101
                                 + 0.964918 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.902269 w_k1_ADM_s101
                                 + 1.20223 w_k1_AXP_s101
                                 + 1.17565 w_k1_DIS_s101
                                 + 1.07718 w_k1_FDX_s101
                                 + 1.23235 w_k1_IBM_s101
                                 + 1.13651 w_k1_INTC_s101
                                 + 1.05934 w_k1_KO_s101 + 1.4287 w_k1_MAT_s101
                                 + 0.967846 w_k1_MSFT_s101
                                 + 0.974661 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.03291 w_k1_ADM_s101
                                 + 1.16916 w_k1_AXP_s101 + 1.193 w_k1_DIS_s101
                                 + 1.2165 w_k1_FDX_s101
                                 + 0.846858 w_k1_IBM_s101
                                 + 1.25737 w_k1_INTC_s101
                                 + 0.977467 w_k1_KO_s101
                                 + 1.09394 w_k1_MAT_s101
                                 + 1.28488 w_k1_MSFT_s101
                                 + 0.951319 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.30969 w_k1_ADM_s102
                                 + 1.05142 w_k1_AXP_s102
                                 + 1.71902 w_k1_DIS_s102
                                 + 1.43125 w_k1_FDX_s102 + 1.0813 w_k1_IBM_s102
                                 + 1.01943 w_k1_INTC_s102
                                 + 1.20001 w_k1_KO_s102 + 1.07764 w_k1_MAT_s102
                                 + 1.50057 w_k1_MSFT_s102
                                 + 1.19436 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.35863 w_k1_ADM_s102
                                 + 1.34067 w_k1_AXP_s102
                                 + 1.31105 w_k1_DIS_s102 + 1.1683 w_k1_FDX_s102
                                 + 1.23898 w_k1_IBM_s102
                                 + 0.926243 w_k1_INTC_s102
                                 + 1.02862 w_k1_KO_s102 + 1.17651 w_k1_MAT_s102
                                 + 1.10305 w_k1_MSFT_s102
                                 + 1.16089 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.73355 w_k1_ADM_s102
                                 + 0.730962 w_k1_AXP_s102
                                 + 0.575951 w_k1_DIS_s102
                                 + 0.678613 w_k1_FDX_s102
                                 + 0.845168 w_k1_IBM_s102
                                 + 0.577729 w_k1_INTC_s102
                                 + 0.670247 w_k1_KO_s102
                                 + 0.679705 w_k1_MAT_s102
                                 + 0.701981 w_k1_MSFT_s102
                                 + 0.78471 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.969535 w_k1_ADM_s102
                                 + 0.980444 w_k1_AXP_s102
                                 + 0.979851 w_k1_DIS_s102
                                 + 0.891958 w_k1_FDX_s102
                                 + 1.25879 w_k1_IBM_s102
                                 + 0.795834 w_k1_INTC_s102
                                 + 0.966514 w_k1_KO_s102
                                 + 1.24569 w_k1_MAT_s102
                                 + 0.983219 w_k1_MSFT_s102
                                 + 1.01521 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.67645 w_k1_ADM_s102
                                 + 1.56908 w_k1_AXP_s102 + 1.6719 w_k1_DIS_s102
                                 + 1.45392 w_k1_FDX_s102
                                 + 1.25881 w_k1_IBM_s102
                                 + 1.22196 w_k1_INTC_s102
                                 + 1.24833 w_k1_KO_s102 + 1.05296 w_k1_MAT_s102
                                 + 1.4187 w_k1_MSFT_s102 + 1.24904 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.14855 w_k1_ADM_s102
                                 + 1.11646 w_k1_AXP_s102
                                 + 0.928917 w_k1_DIS_s102
                                 + 1.00689 w_k1_FDX_s102
                                 + 1.10104 w_k1_IBM_s102
                                 + 0.827001 w_k1_INTC_s102
                                 + 0.948084 w_k1_KO_s102
                                 + 1.34762 w_k1_MAT_s102
                                 + 0.997468 w_k1_MSFT_s102
                                 + 0.990719 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.00105 w_k1_ADM_s102
                                 + 1.03288 w_k1_AXP_s102
                                 + 1.03978 w_k1_DIS_s102 + 1.0978 w_k1_FDX_s102
                                 + 0.806295 w_k1_IBM_s102
                                 + 0.899881 w_k1_INTC_s102
                                 + 0.687594 w_k1_KO_s102
                                 + 0.744371 w_k1_MAT_s102
                                 + 1.06876 w_k1_MSFT_s102
                                 + 0.924867 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.901293 w_k1_ADM_s102
                                 + 0.823684 w_k1_AXP_s102
                                 + 0.730771 w_k1_DIS_s102
                                 + 0.843361 w_k1_FDX_s102
                                 + 0.77578 w_k1_IBM_s102
                                 + 0.70688 w_k1_INTC_s102
                                 + 0.606448 w_k1_KO_s102
                                 + 0.505688 w_k1_MAT_s102
                                 + 0.792884 w_k1_MSFT_s102
                                 + 0.765902 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.3272 w_k1_ADM_s103
                                 + 1.02745 w_k1_AXP_s103 + 1.333 w_k1_DIS_s103
                                 + 1.23022 w_k1_FDX_s103
                                 + 0.723364 w_k1_IBM_s103
                                 + 1.15082 w_k1_INTC_s103
                                 + 1.10358 w_k1_KO_s103
                                 + 0.781948 w_k1_MAT_s103
                                 + 1.28972 w_k1_MSFT_s103
                                 + 1.06765 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.50926 w_k1_ADM_s103
                                 + 1.46056 w_k1_AXP_s103
                                 + 1.02007 w_k1_DIS_s103
                                 + 1.20286 w_k1_FDX_s103
                                 + 0.912013 w_k1_IBM_s103
                                 + 1.00786 w_k1_INTC_s103
                                 + 0.934357 w_k1_KO_s103
                                 + 1.20681 w_k1_MAT_s103
                                 + 1.03216 w_k1_MSFT_s103
                                 + 1.05821 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.06149 w_k1_ADM_s103
                                 + 0.837867 w_k1_AXP_s103
                                 + 1.08479 w_k1_DIS_s103
                                 + 1.03364 w_k1_FDX_s103
                                 + 0.867852 w_k1_IBM_s103
                                 + 0.979077 w_k1_INTC_s103 + 1.034 w_k1_KO_s103
                                 + 0.723232 w_k1_MAT_s103
                                 + 1.01296 w_k1_MSFT_s103
                                 + 0.989322 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.00301 w_k1_ADM_s103
                                 + 1.13021 w_k1_AXP_s103
                                 + 1.00174 w_k1_DIS_s103
                                 + 0.945702 w_k1_FDX_s103
                                 + 1.11066 w_k1_IBM_s103
                                 + 1.0381 w_k1_INTC_s103
                                 + 0.999367 w_k1_KO_s103
                                 + 1.03334 w_k1_MAT_s103
                                 + 1.00798 w_k1_MSFT_s103
                                 + 1.05185 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.989456 w_k1_ADM_s103
                                 + 0.712217 w_k1_AXP_s103
                                 + 0.629884 w_k1_DIS_s103
                                 + 0.78251 w_k1_FDX_s103
                                 + 0.675346 w_k1_IBM_s103
                                 + 0.756935 w_k1_INTC_s103
                                 + 0.729397 w_k1_KO_s103
                                 + 0.70724 w_k1_MAT_s103
                                 + 0.77682 w_k1_MSFT_s103
                                 + 0.871551 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.50343 w_k1_ADM_s103
                                 + 1.62346 w_k1_AXP_s103
                                 + 1.45452 w_k1_DIS_s103
                                 + 1.29352 w_k1_FDX_s103
                                 + 1.43301 w_k1_IBM_s103
                                 + 1.32105 w_k1_INTC_s103
                                 + 1.33856 w_k1_KO_s103 + 1.78451 w_k1_MAT_s103
                                 + 1.20836 w_k1_MSFT_s103
                                 + 1.27898 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.880556 w_k1_ADM_s103
                                 + 0.922493 w_k1_AXP_s103
                                 + 0.716377 w_k1_DIS_s103
                                 + 0.736481 w_k1_FDX_s103
                                 + 1.10745 w_k1_IBM_s103
                                 + 0.918401 w_k1_INTC_s103
                                 + 0.875969 w_k1_KO_s103
                                 + 1.12644 w_k1_MAT_s103
                                 + 0.853392 w_k1_MSFT_s103
                                 + 0.883955 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.22171 w_k1_ADM_s103
                                 + 1.40223 w_k1_AXP_s103
                                 + 1.63427 w_k1_DIS_s103 + 1.2389 w_k1_FDX_s103
                                 + 1.24112 w_k1_IBM_s103
                                 + 1.24115 w_k1_INTC_s103
                                 + 1.29201 w_k1_KO_s103 + 1.20568 w_k1_MAT_s103
                                 + 1.26603 w_k1_MSFT_s103
                                 + 1.27697 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.839883 w_k1_ADM_s104
                                 + 0.767302 w_k1_AXP_s104
                                 + 0.686852 w_k1_DIS_s104
                                 + 0.766548 w_k1_FDX_s104
                                 + 0.726515 w_k1_IBM_s104
                                 + 0.961217 w_k1_INTC_s104
                                 + 0.797026 w_k1_KO_s104
                                 + 0.663002 w_k1_MAT_s104
                                 + 0.837396 w_k1_MSFT_s104
                                 + 0.811379 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.901679 w_k1_ADM_s104
                                 + 0.874864 w_k1_AXP_s104
                                 + 0.908435 w_k1_DIS_s104
                                 + 0.871343 w_k1_FDX_s104
                                 + 0.989542 w_k1_IBM_s104
                                 + 0.997918 w_k1_INTC_s104
                                 + 0.910822 w_k1_KO_s104
                                 + 1.02205 w_k1_MAT_s104
                                 + 0.925515 w_k1_MSFT_s104
                                 + 0.97126 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.82667 w_k1_ADM_s104
                                 + 0.857146 w_k1_AXP_s104
                                 + 0.750555 w_k1_DIS_s104
                                 + 0.910466 w_k1_FDX_s104
                                 + 0.632179 w_k1_IBM_s104
                                 + 1.05206 w_k1_INTC_s104
                                 + 0.767103 w_k1_KO_s104
                                 + 0.753804 w_k1_MAT_s104
                                 + 1.07162 w_k1_MSFT_s104
                                 + 0.897674 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.16886 w_k1_ADM_s104
                                 + 1.3762 w_k1_AXP_s104 + 1.02256 w_k1_DIS_s104
                                 + 1.04085 w_k1_FDX_s104
                                 + 0.937618 w_k1_IBM_s104
                                 + 0.97245 w_k1_INTC_s104 + 1.0641 w_k1_KO_s104
                                 + 1.32957 w_k1_MAT_s104
                                 + 1.06533 w_k1_MSFT_s104
                                 + 1.03268 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.47893 w_k1_ADM_s104
                                 + 1.54068 w_k1_AXP_s104
                                 + 1.69584 w_k1_DIS_s104
                                 + 1.43215 w_k1_FDX_s104
                                 + 1.30404 w_k1_IBM_s104
                                 + 1.14029 w_k1_INTC_s104
                                 + 1.19614 w_k1_KO_s104 + 1.86589 w_k1_MAT_s104
                                 + 1.46816 w_k1_MSFT_s104
                                 + 1.20797 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.03018 w_k1_ADM_s104
                                 + 1.12291 w_k1_AXP_s104
                                 + 1.19864 w_k1_DIS_s104
                                 + 0.985615 w_k1_FDX_s104
                                 + 1.34229 w_k1_IBM_s104
                                 + 0.965078 w_k1_INTC_s104
                                 + 1.09959 w_k1_KO_s104 + 1.56575 w_k1_MAT_s104
                                 + 1.12404 w_k1_MSFT_s104
                                 + 1.20235 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.946913 w_k1_ADM_s104
                                 + 0.853589 w_k1_AXP_s104
                                 + 0.923829 w_k1_DIS_s104
                                 + 0.947364 w_k1_FDX_s104
                                 + 0.723098 w_k1_IBM_s104
                                 + 1.17336 w_k1_INTC_s104
                                 + 1.04184 w_k1_KO_s104
                                 + 0.736168 w_k1_MAT_s104
                                 + 1.17311 w_k1_MSFT_s104
                                 + 1.01035 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.13867 w_k1_ADM_s104
                                 + 1.02254 w_k1_AXP_s104 + 1.2187 w_k1_DIS_s104
                                 + 1.17464 w_k1_FDX_s104
                                 + 0.940298 w_k1_IBM_s104
                                 + 1.15142 w_k1_INTC_s104
                                 + 1.00931 w_k1_KO_s104 + 1.03901 w_k1_MAT_s104
                                 + 1.26648 w_k1_MSFT_s104
                                 + 1.14514 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.800807 w_k1_ADM_s105
                                 + 1.18363 w_k1_AXP_s105
                                 + 0.869042 w_k1_DIS_s105
                                 + 1.0263 w_k1_FDX_s105 + 1.05689 w_k1_IBM_s105
                                 + 1.14241 w_k1_INTC_s105
                                 + 0.951144 w_k1_KO_s105
                                 + 1.12956 w_k1_MAT_s105
                                 + 0.92366 w_k1_MSFT_s105
                                 + 1.08658 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.944967 w_k1_ADM_s105
                                 + 1.05198 w_k1_AXP_s105
                                 + 1.22495 w_k1_DIS_s105
                                 + 1.14823 w_k1_FDX_s105
                                 + 0.944978 w_k1_IBM_s105
                                 + 1.26087 w_k1_INTC_s105
                                 + 0.998322 w_k1_KO_s105
                                 + 0.795431 w_k1_MAT_s105
                                 + 1.11185 w_k1_MSFT_s105
                                 + 1.12084 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.1453 w_k1_ADM_s105
                                 + 1.00073 w_k1_AXP_s105
                                 + 0.956261 w_k1_DIS_s105
                                 + 1.27467 w_k1_FDX_s105
                                 + 0.609186 w_k1_IBM_s105
                                 + 1.17766 w_k1_INTC_s105
                                 + 0.852848 w_k1_KO_s105
                                 + 0.70393 w_k1_MAT_s105
                                 + 1.0938 w_k1_MSFT_s105 + 0.97405 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.10584 w_k1_ADM_s105
                                 + 1.04379 w_k1_AXP_s105
                                 + 1.34085 w_k1_DIS_s105 + 1.1238 w_k1_FDX_s105
                                 + 1.34813 w_k1_IBM_s105
                                 + 1.00882 w_k1_INTC_s105
                                 + 1.33316 w_k1_KO_s105 + 1.31326 w_k1_MAT_s105
                                 + 1.11122 w_k1_MSFT_s105
                                 + 1.23854 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.73363 w_k1_ADM_s105
                                 + 0.667036 w_k1_AXP_s105
                                 + 0.613049 w_k1_DIS_s105
                                 + 0.782676 w_k1_FDX_s105
                                 + 0.746302 w_k1_IBM_s105
                                 + 0.967102 w_k1_INTC_s105
                                 + 0.824933 w_k1_KO_s105
                                 + 0.571688 w_k1_MAT_s105
                                 + 0.698679 w_k1_MSFT_s105
                                 + 0.894591 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.332 w_k1_ADM_s105
                                 + 1.34985 w_k1_AXP_s105
                                 + 1.69407 w_k1_DIS_s105
                                 + 1.39671 w_k1_FDX_s105 + 1.0121 w_k1_IBM_s105
                                 + 1.07991 w_k1_INTC_s105
                                 + 1.36591 w_k1_KO_s105 + 1.17564 w_k1_MAT_s105
                                 + 1.23677 w_k1_MSFT_s105
                                 + 1.38177 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.04033 w_k1_ADM_s105
                                 + 0.818785 w_k1_AXP_s105
                                 + 0.930189 w_k1_DIS_s105
                                 + 0.978377 w_k1_FDX_s105
                                 + 0.875048 w_k1_IBM_s105
                                 + 0.986746 w_k1_INTC_s105
                                 + 0.990944 w_k1_KO_s105
                                 + 0.760824 w_k1_MAT_s105
                                 + 1.0824 w_k1_MSFT_s105 + 1.14043 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.85378 w_k1_ADM_s105
                                 + 0.975586 w_k1_AXP_s105
                                 + 0.912496 w_k1_DIS_s105
                                 + 0.848445 w_k1_FDX_s105
                                 + 1.56716 w_k1_IBM_s105
                                 + 1.05294 w_k1_INTC_s105
                                 + 1.19902 w_k1_KO_s105 + 1.43042 w_k1_MAT_s105
                                 + 0.97892 w_k1_MSFT_s105
                                 + 1.02968 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.782688 w_k1_ADM_s106
                                 + 0.810438 w_k1_AXP_s106
                                 + 0.745531 w_k1_DIS_s106
                                 + 0.917686 w_k1_FDX_s106
                                 + 0.730791 w_k1_IBM_s106
                                 + 0.805467 w_k1_INTC_s106
                                 + 0.889839 w_k1_KO_s106
                                 + 1.14724 w_k1_MAT_s106
                                 + 0.950632 w_k1_MSFT_s106
                                 + 0.960804 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.714565 w_k1_ADM_s106
                                 + 0.757962 w_k1_AXP_s106
                                 + 0.672799 w_k1_DIS_s106
                                 + 0.709864 w_k1_FDX_s106
                                 + 1.07473 w_k1_IBM_s106
                                 + 0.819009 w_k1_INTC_s106
                                 + 0.787786 w_k1_KO_s106
                                 + 0.953199 w_k1_MAT_s106
                                 + 0.781863 w_k1_MSFT_s106
                                 + 0.877977 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.15171 w_k1_ADM_s106
                                 + 1.62614 w_k1_AXP_s106 + 1.0606 w_k1_DIS_s106
                                 + 1.0761 w_k1_FDX_s106 + 1.14208 w_k1_IBM_s106
                                 + 1.03711 w_k1_INTC_s106
                                 + 1.11503 w_k1_KO_s106 + 1.49518 w_k1_MAT_s106
                                 + 1.0434 w_k1_MSFT_s106 + 1.1669 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.29751 w_k1_ADM_s106
                                 + 1.37271 w_k1_AXP_s106
                                 + 1.53101 w_k1_DIS_s106
                                 + 1.34029 w_k1_FDX_s106
                                 + 1.19705 w_k1_IBM_s106
                                 + 1.07928 w_k1_INTC_s106
                                 + 1.08259 w_k1_KO_s106 + 1.79152 w_k1_MAT_s106
                                 + 1.2834 w_k1_MSFT_s106 + 1.12183 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.20273 w_k1_ADM_s106
                                 + 1.23779 w_k1_AXP_s106
                                 + 1.38418 w_k1_DIS_s106
                                 + 1.20638 w_k1_FDX_s106
                                 + 0.916741 w_k1_IBM_s106
                                 + 1.09325 w_k1_INTC_s106
                                 + 1.16902 w_k1_KO_s106 + 1.45745 w_k1_MAT_s106
                                 + 1.29941 w_k1_MSFT_s106
                                 + 1.18603 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.1848 w_k1_ADM_s106
                                 + 1.20522 w_k1_AXP_s106
                                 + 1.25161 w_k1_DIS_s106
                                 + 1.01615 w_k1_FDX_s106
                                 + 1.70818 w_k1_IBM_s106
                                 + 0.875598 w_k1_INTC_s106
                                 + 1.04653 w_k1_KO_s106 + 1.82178 w_k1_MAT_s106
                                 + 1.0324 w_k1_MSFT_s106 + 1.04177 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.992787 w_k1_ADM_s106
                                 + 0.903078 w_k1_AXP_s106
                                 + 1.0584 w_k1_DIS_s106
                                 + 0.904863 w_k1_FDX_s106
                                 + 1.12145 w_k1_IBM_s106
                                 + 0.981917 w_k1_INTC_s106
                                 + 0.955011 w_k1_KO_s106
                                 + 0.943231 w_k1_MAT_s106
                                 + 0.970074 w_k1_MSFT_s106
                                 + 0.961318 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.20285 w_k1_ADM_s106
                                 + 1.09535 w_k1_AXP_s106
                                 + 0.892246 w_k1_DIS_s106
                                 + 0.958781 w_k1_FDX_s106
                                 + 1.19647 w_k1_IBM_s106
                                 + 0.814266 w_k1_INTC_s106
                                 + 1.11271 w_k1_KO_s106 + 1.52687 w_k1_MAT_s106
                                 + 1.01131 w_k1_MSFT_s106 + 1.113 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.08416 w_k1_ADM_s107
                                 + 1.20517 w_k1_AXP_s107
                                 + 1.26039 w_k1_DIS_s107
                                 + 1.19273 w_k1_FDX_s107 + 1.025 w_k1_IBM_s107
                                 + 1.23123 w_k1_INTC_s107 + 1.1675 w_k1_KO_s107
                                 + 1.38437 w_k1_MAT_s107
                                 + 1.28652 w_k1_MSFT_s107
                                 + 1.07975 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.850551 w_k1_ADM_s107
                                 + 1.10209 w_k1_AXP_s107
                                 + 0.952534 w_k1_DIS_s107
                                 + 1.01528 w_k1_FDX_s107
                                 + 1.05542 w_k1_IBM_s107
                                 + 1.19694 w_k1_INTC_s107 + 1.2086 w_k1_KO_s107
                                 + 1.099 w_k1_MAT_s107 + 1.03565 w_k1_MSFT_s107
                                 + 0.97048 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.0699 w_k1_ADM_s107
                                 + 1.04542 w_k1_AXP_s107
                                 + 1.00509 w_k1_DIS_s107
                                 + 0.928339 w_k1_FDX_s107
                                 + 1.25992 w_k1_IBM_s107
                                 + 1.01232 w_k1_INTC_s107
                                 + 1.30579 w_k1_KO_s107 + 1.63693 w_k1_MAT_s107
                                 + 1.07746 w_k1_MSFT_s107
                                 + 1.01641 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.771874 w_k1_ADM_s107
                                 + 0.844104 w_k1_AXP_s107
                                 + 0.820669 w_k1_DIS_s107
                                 + 0.77256 w_k1_FDX_s107
                                 + 1.12121 w_k1_IBM_s107
                                 + 1.05099 w_k1_INTC_s107
                                 + 1.29534 w_k1_KO_s107 + 1.15255 w_k1_MAT_s107
                                 + 0.990773 w_k1_MSFT_s107
                                 + 0.897061 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.0561 w_k1_ADM_s107
                                 + 1.24169 w_k1_AXP_s107
                                 + 0.948754 w_k1_DIS_s107
                                 + 1.17416 w_k1_FDX_s107
                                 + 0.795969 w_k1_IBM_s107
                                 + 1.31822 w_k1_INTC_s107
                                 + 1.15853 w_k1_KO_s107 + 0.80943 w_k1_MAT_s107
                                 + 1.03016 w_k1_MSFT_s107
                                 + 1.04525 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.2816 w_k1_ADM_s107
                                 + 1.55096 w_k1_AXP_s107
                                 + 1.67447 w_k1_DIS_s107
                                 + 1.49642 w_k1_FDX_s107
                                 + 1.29653 w_k1_IBM_s107
                                 + 1.51103 w_k1_INTC_s107
                                 + 1.56006 w_k1_KO_s107 + 1.28274 w_k1_MAT_s107
                                 + 1.54468 w_k1_MSFT_s107 + 1.2841 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.905153 w_k1_ADM_s107
                                 + 0.774276 w_k1_AXP_s107
                                 + 0.707291 w_k1_DIS_s107
                                 + 0.867278 w_k1_FDX_s107
                                 + 0.763242 w_k1_IBM_s107
                                 + 0.929478 w_k1_INTC_s107
                                 + 0.85169 w_k1_KO_s107
                                 + 0.793267 w_k1_MAT_s107
                                 + 0.862892 w_k1_MSFT_s107
                                 + 0.796309 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.915643 w_k1_ADM_s107
                                 + 1.16115 w_k1_AXP_s107
                                 + 1.31748 w_k1_DIS_s107 + 1.059 w_k1_FDX_s107
                                 + 1.59654 w_k1_IBM_s107
                                 + 1.14746 w_k1_INTC_s107 + 1.414 w_k1_KO_s107
                                 + 1.66102 w_k1_MAT_s107
                                 + 1.20857 w_k1_MSFT_s107
                                 + 1.07289 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.15246 w_k1_ADM_s110
                                 + 1.2775 w_k1_AXP_s110
                                 + 0.897396 w_k1_DIS_s110
                                 + 0.896579 w_k1_FDX_s110
                                 + 1.33271 w_k1_IBM_s110
                                 + 0.953893 w_k1_INTC_s110
                                 + 1.01614 w_k1_KO_s110 + 1.3856 w_k1_MAT_s110
                                 + 1.04944 w_k1_MSFT_s110
                                 + 1.00255 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.13425 w_k1_ADM_s110
                                 + 1.00054 w_k1_AXP_s110
                                 + 0.977223 w_k1_DIS_s110
                                 + 1.02123 w_k1_FDX_s110
                                 + 0.79948 w_k1_IBM_s110
                                 + 1.12169 w_k1_INTC_s110
                                 + 0.972179 w_k1_KO_s110
                                 + 0.815847 w_k1_MAT_s110
                                 + 1.14605 w_k1_MSFT_s110
                                 + 1.08932 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.01848 w_k1_ADM_s110
                                 + 1.3287 w_k1_AXP_s110 + 1.3467 w_k1_DIS_s110
                                 + 1.11413 w_k1_FDX_s110
                                 + 0.99944 w_k1_IBM_s110
                                 + 1.2106 w_k1_INTC_s110
                                 + 0.981554 w_k1_KO_s110
                                 + 1.00247 w_k1_MAT_s110
                                 + 1.15727 w_k1_MSFT_s110
                                 + 0.954045 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.928274 w_k1_ADM_s110
                                 + 0.804555 w_k1_AXP_s110
                                 + 0.922708 w_k1_DIS_s110
                                 + 0.794553 w_k1_FDX_s110
                                 + 1.00599 w_k1_IBM_s110
                                 + 0.945004 w_k1_INTC_s110
                                 + 0.930328 w_k1_KO_s110
                                 + 0.970239 w_k1_MAT_s110
                                 + 1.13902 w_k1_MSFT_s110
                                 + 0.90526 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.79458 w_k1_ADM_s110
                                 + 1.36865 w_k1_AXP_s110 + 1.5707 w_k1_DIS_s110
                                 + 1.31272 w_k1_FDX_s110
                                 + 0.968276 w_k1_IBM_s110
                                 + 1.40429 w_k1_INTC_s110
                                 + 1.33723 w_k1_KO_s110
                                 + 0.845831 w_k1_MAT_s110
                                 + 1.24731 w_k1_MSFT_s110
                                 + 1.25143 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.03978 w_k1_ADM_s110
                                 + 0.704738 w_k1_AXP_s110
                                 + 1.10912 w_k1_DIS_s110
                                 + 0.896349 w_k1_FDX_s110
                                 + 0.651518 w_k1_IBM_s110
                                 + 1.02689 w_k1_INTC_s110
                                 + 0.889538 w_k1_KO_s110
                                 + 0.506707 w_k1_MAT_s110
                                 + 1.14507 w_k1_MSFT_s110
                                 + 0.953752 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.9828 w_k1_ADM_s110
                                 + 0.962905 w_k1_AXP_s110
                                 + 0.663724 w_k1_DIS_s110
                                 + 0.789991 w_k1_FDX_s110
                                 + 0.83963 w_k1_IBM_s110
                                 + 0.878508 w_k1_INTC_s110
                                 + 0.822482 w_k1_KO_s110
                                 + 0.829016 w_k1_MAT_s110
                                 + 0.869331 w_k1_MSFT_s110
                                 + 0.826327 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.08418 w_k1_ADM_s110
                                 + 1.19262 w_k1_AXP_s110
                                 + 1.46576 w_k1_DIS_s110
                                 + 1.09089 w_k1_FDX_s110 + 1.5703 w_k1_IBM_s110
                                 + 1.14663 w_k1_INTC_s110
                                 + 1.26934 w_k1_KO_s110 + 1.48549 w_k1_MAT_s110
                                 + 1.37514 w_k1_MSFT_s110
                                 + 1.12999 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.758022 w_k1_ADM_s111
                                 + 0.734367 w_k1_AXP_s111
                                 + 0.742661 w_k1_DIS_s111
                                 + 0.796514 w_k1_FDX_s111
                                 + 0.962901 w_k1_IBM_s111
                                 + 0.862917 w_k1_INTC_s111
                                 + 0.763639 w_k1_KO_s111
                                 + 0.817649 w_k1_MAT_s111
                                 + 0.88247 w_k1_MSFT_s111
                                 + 0.955475 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.751623 w_k1_ADM_s111
                                 + 0.512381 w_k1_AXP_s111
                                 + 0.657909 w_k1_DIS_s111
                                 + 0.696287 w_k1_FDX_s111
                                 + 0.635218 w_k1_IBM_s111
                                 + 1.03343 w_k1_INTC_s111
                                 + 0.739687 w_k1_KO_s111
                                 + 0.45583 w_k1_MAT_s111
                                 + 0.875908 w_k1_MSFT_s111
                                 + 0.846115 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.06369 w_k1_ADM_s111
                                 + 0.769414 w_k1_AXP_s111
                                 + 0.709011 w_k1_DIS_s111
                                 + 0.859883 w_k1_FDX_s111
                                 + 0.762762 w_k1_IBM_s111
                                 + 0.784393 w_k1_INTC_s111
                                 + 0.759706 w_k1_KO_s111
                                 + 0.744603 w_k1_MAT_s111
                                 + 0.898709 w_k1_MSFT_s111
                                 + 0.747149 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.15387 w_k1_ADM_s111
                                 + 1.29921 w_k1_AXP_s111
                                 + 1.36465 w_k1_DIS_s111
                                 + 1.12516 w_k1_FDX_s111
                                 + 1.37035 w_k1_IBM_s111
                                 + 1.36158 w_k1_INTC_s111
                                 + 1.50342 w_k1_KO_s111 + 1.33992 w_k1_MAT_s111
                                 + 1.3575 w_k1_MSFT_s111 + 1.24659 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.02961 w_k1_ADM_s111
                                 + 1.20116 w_k1_AXP_s111
                                 + 0.998538 w_k1_DIS_s111
                                 + 1.02123 w_k1_FDX_s111
                                 + 1.79275 w_k1_IBM_s111
                                 + 1.25738 w_k1_INTC_s111
                                 + 1.34239 w_k1_KO_s111 + 1.8606 w_k1_MAT_s111
                                 + 1.184 w_k1_MSFT_s111 + 1.19732 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.974271 w_k1_ADM_s111
                                 + 0.885081 w_k1_AXP_s111
                                 + 0.944861 w_k1_DIS_s111
                                 + 1.11161 w_k1_FDX_s111
                                 + 0.725556 w_k1_IBM_s111
                                 + 0.923721 w_k1_INTC_s111
                                 + 0.865118 w_k1_KO_s111
                                 + 0.78024 w_k1_MAT_s111
                                 + 1.03029 w_k1_MSFT_s111
                                 + 0.866601 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.875963 w_k1_ADM_s111
                                 + 0.723287 w_k1_AXP_s111
                                 + 1.06369 w_k1_DIS_s111
                                 + 0.904111 w_k1_FDX_s111
                                 + 1.32171 w_k1_IBM_s111
                                 + 0.944741 w_k1_INTC_s111
                                 + 1.12518 w_k1_KO_s111 + 1.06161 w_k1_MAT_s111
                                 + 1.0754 w_k1_MSFT_s111 + 1.0518 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.878106 w_k1_ADM_s111
                                 + 1.21529 w_k1_AXP_s111
                                 + 1.09272 w_k1_DIS_s111
                                 + 1.05411 w_k1_FDX_s111
                                 + 1.03938 w_k1_IBM_s111
                                 + 0.955335 w_k1_INTC_s111
                                 + 1.07782 w_k1_KO_s111 + 1.19654 w_k1_MAT_s111
                                 + 1.12582 w_k1_MSFT_s111
                                 + 1.17524 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.903301 w_k1_ADM_s112
                                 + 0.643199 w_k1_AXP_s112
                                 + 0.723982 w_k1_DIS_s112
                                 + 0.770386 w_k1_FDX_s112
                                 + 0.738547 w_k1_IBM_s112
                                 + 1.10464 w_k1_INTC_s112
                                 + 0.876967 w_k1_KO_s112
                                 + 0.504635 w_k1_MAT_s112
                                 + 0.877998 w_k1_MSFT_s112
                                 + 0.908881 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.30308 w_k1_ADM_s112
                                 + 1.36605 w_k1_AXP_s112
                                 + 1.10453 w_k1_DIS_s112
                                 + 1.14925 w_k1_FDX_s112
                                 + 0.779456 w_k1_IBM_s112
                                 + 1.32066 w_k1_INTC_s112
                                 + 1.06576 w_k1_KO_s112 + 0.83001 w_k1_MAT_s112
                                 + 1.00232 w_k1_MSFT_s112
                                 + 1.11462 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.28218 w_k1_ADM_s112
                                 + 1.10113 w_k1_AXP_s112
                                 + 1.54306 w_k1_DIS_s112
                                 + 1.28994 w_k1_FDX_s112
                                 + 1.05609 w_k1_IBM_s112
                                 + 1.40293 w_k1_INTC_s112
                                 + 1.46613 w_k1_KO_s112
                                 + 0.818528 w_k1_MAT_s112
                                 + 1.15963 w_k1_MSFT_s112
                                 + 1.29281 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.0853 w_k1_ADM_s112
                                 + 1.02216 w_k1_AXP_s112 + 1.0111 w_k1_DIS_s112
                                 + 1.02903 w_k1_FDX_s112
                                 + 1.38298 w_k1_IBM_s112
                                 + 1.20813 w_k1_INTC_s112
                                 + 1.38416 w_k1_KO_s112 + 1.73302 w_k1_MAT_s112
                                 + 1.17051 w_k1_MSFT_s112
                                 + 1.26658 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.956205 w_k1_ADM_s112
                                 + 1.57306 w_k1_AXP_s112 + 1.8237 w_k1_DIS_s112
                                 + 1.42931 w_k1_FDX_s112
                                 + 1.55427 w_k1_IBM_s112
                                 + 1.53726 w_k1_INTC_s112
                                 + 1.55188 w_k1_KO_s112 + 1.48202 w_k1_MAT_s112
                                 + 1.17076 w_k1_MSFT_s112
                                 + 1.32013 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.10019 w_k1_ADM_s112
                                 + 1.0229 w_k1_AXP_s112 + 1.10064 w_k1_DIS_s112
                                 + 1.05268 w_k1_FDX_s112
                                 + 1.08344 w_k1_IBM_s112
                                 + 1.09791 w_k1_INTC_s112
                                 + 1.11704 w_k1_KO_s112 + 1.30146 w_k1_MAT_s112
                                 + 0.911887 w_k1_MSFT_s112
                                 + 1.20982 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.860862 w_k1_ADM_s112
                                 + 0.883088 w_k1_AXP_s112
                                 + 0.884433 w_k1_DIS_s112
                                 + 0.850897 w_k1_FDX_s112
                                 + 0.884396 w_k1_IBM_s112
                                 + 0.969944 w_k1_INTC_s112
                                 + 0.9578 w_k1_KO_s112 + 1.00124 w_k1_MAT_s112
                                 + 0.868666 w_k1_MSFT_s112
                                 + 1.11014 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.572637 w_k1_ADM_s112
                                 + 0.791384 w_k1_AXP_s112
                                 + 0.744514 w_k1_DIS_s112
                                 + 0.759562 w_k1_FDX_s112
                                 + 1.24078 w_k1_IBM_s112
                                 + 0.930872 w_k1_INTC_s112
                                 + 1.0586 w_k1_KO_s112 + 0.977084 w_k1_MAT_s112
                                 + 0.661058 w_k1_MSFT_s112
                                 + 1.00245 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.942896 w_k1_ADM_s113
                                 + 0.580375 w_k1_AXP_s113
                                 + 0.674538 w_k1_DIS_s113
                                 + 0.78883 w_k1_FDX_s113
                                 + 0.743043 w_k1_IBM_s113
                                 + 0.812435 w_k1_INTC_s113
                                 + 0.721676 w_k1_KO_s113
                                 + 0.593237 w_k1_MAT_s113
                                 + 0.819659 w_k1_MSFT_s113
                                 + 0.792958 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.05105 w_k1_ADM_s113
                                 + 0.792015 w_k1_AXP_s113
                                 + 0.780624 w_k1_DIS_s113
                                 + 0.802057 w_k1_FDX_s113
                                 + 1.01858 w_k1_IBM_s113
                                 + 0.909782 w_k1_INTC_s113
                                 + 0.923397 w_k1_KO_s113
                                 + 0.601397 w_k1_MAT_s113
                                 + 1.0019 w_k1_MSFT_s113
                                 + 0.840667 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.24545 w_k1_ADM_s113
                                 + 1.31765 w_k1_AXP_s113
                                 + 1.37592 w_k1_DIS_s113
                                 + 1.09684 w_k1_FDX_s113
                                 + 1.65369 w_k1_IBM_s113
                                 + 1.04839 w_k1_INTC_s113
                                 + 1.30896 w_k1_KO_s113 + 1.1902 w_k1_MAT_s113
                                 + 1.24898 w_k1_MSFT_s113
                                 + 1.13311 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.44527 w_k1_ADM_s113
                                 + 1.37574 w_k1_AXP_s113
                                 + 1.23437 w_k1_DIS_s113 + 1.293 w_k1_FDX_s113
                                 + 0.900759 w_k1_IBM_s113
                                 + 0.971977 w_k1_INTC_s113
                                 + 1.12504 w_k1_KO_s113 + 1.05578 w_k1_MAT_s113
                                 + 1.25268 w_k1_MSFT_s113
                                 + 1.11952 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.26866 w_k1_ADM_s113
                                 + 1.01063 w_k1_AXP_s113 + 1.1707 w_k1_DIS_s113
                                 + 0.902564 w_k1_FDX_s113
                                 + 1.42706 w_k1_IBM_s113
                                 + 1.15241 w_k1_INTC_s113
                                 + 1.19806 w_k1_KO_s113
                                 + 0.864161 w_k1_MAT_s113
                                 + 1.10928 w_k1_MSFT_s113
                                 + 0.988403 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.767985 w_k1_ADM_s113
                                 + 0.788879 w_k1_AXP_s113
                                 + 0.679208 w_k1_DIS_s113
                                 + 0.775384 w_k1_FDX_s113
                                 + 1.35887 w_k1_IBM_s113
                                 + 1.05549 w_k1_INTC_s113
                                 + 0.959565 w_k1_KO_s113 + 1.0295 w_k1_MAT_s113
                                 + 0.881773 w_k1_MSFT_s113
                                 + 0.844591 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.25799 w_k1_ADM_s113
                                 + 0.921174 w_k1_AXP_s113
                                 + 1.14699 w_k1_DIS_s113
                                 + 0.991804 w_k1_FDX_s113
                                 + 1.0622 w_k1_IBM_s113
                                 + 0.809197 w_k1_INTC_s113
                                 + 1.10175 w_k1_KO_s113
                                 + 0.754124 w_k1_MAT_s113
                                 + 1.13879 w_k1_MSFT_s113
                                 + 0.971689 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.32073 w_k1_ADM_s113
                                 + 0.887792 w_k1_AXP_s113
                                 + 0.889132 w_k1_DIS_s113
                                 + 1.04673 w_k1_FDX_s113
                                 + 0.68747 w_k1_IBM_s113
                                 + 1.05874 w_k1_INTC_s113
                                 + 0.738211 w_k1_KO_s113
                                 + 0.508561 w_k1_MAT_s113
                                 + 1.15299 w_k1_MSFT_s113
                                 + 0.810828 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.866692 w_k1_ADM_s114
                                 + 1.02226 w_k1_AXP_s114
                                 + 0.861068 w_k1_DIS_s114
                                 + 1.05718 w_k1_FDX_s114
                                 + 0.789141 w_k1_IBM_s114
                                 + 1.30571 w_k1_INTC_s114
                                 + 0.911689 w_k1_KO_s114
                                 + 0.910055 w_k1_MAT_s114
                                 + 0.976329 w_k1_MSFT_s114
                                 + 1.07372 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.04136 w_k1_ADM_s114
                                 + 1.08925 w_k1_AXP_s114
                                 + 1.18536 w_k1_DIS_s114
                                 + 1.14714 w_k1_FDX_s114
                                 + 0.951791 w_k1_IBM_s114
                                 + 1.06187 w_k1_INTC_s114
                                 + 0.953224 w_k1_KO_s114
                                 + 0.914831 w_k1_MAT_s114
                                 + 1.20157 w_k1_MSFT_s114
                                 + 0.963644 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.50874 w_k1_ADM_s114
                                 + 1.47509 w_k1_AXP_s114
                                 + 1.44636 w_k1_DIS_s114
                                 + 1.46186 w_k1_FDX_s114
                                 + 1.14902 w_k1_IBM_s114
                                 + 1.47851 w_k1_INTC_s114
                                 + 1.09677 w_k1_KO_s114 + 1.17375 w_k1_MAT_s114
                                 + 1.41046 w_k1_MSFT_s114 + 1.1233 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.04646 w_k1_ADM_s114
                                 + 1.12712 w_k1_AXP_s114
                                 + 0.883318 w_k1_DIS_s114
                                 + 0.954475 w_k1_FDX_s114
                                 + 1.41388 w_k1_IBM_s114
                                 + 1.02241 w_k1_INTC_s114 + 1.1885 w_k1_KO_s114
                                 + 1.50892 w_k1_MAT_s114
                                 + 0.879825 w_k1_MSFT_s114
                                 + 1.14752 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.05661 w_k1_ADM_s114
                                 + 1.39009 w_k1_AXP_s114
                                 + 1.69465 w_k1_DIS_s114
                                 + 1.34444 w_k1_FDX_s114
                                 + 1.55244 w_k1_IBM_s114
                                 + 1.26773 w_k1_INTC_s114 + 1.5456 w_k1_KO_s114
                                 + 1.39569 w_k1_MAT_s114
                                 + 1.29497 w_k1_MSFT_s114 + 1.2783 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.894608 w_k1_ADM_s114
                                 + 0.77905 w_k1_AXP_s114
                                 + 0.86525 w_k1_DIS_s114
                                 + 0.922712 w_k1_FDX_s114
                                 + 0.970103 w_k1_IBM_s114
                                 + 0.877681 w_k1_INTC_s114
                                 + 0.974931 w_k1_KO_s114
                                 + 0.761527 w_k1_MAT_s114
                                 + 0.91763 w_k1_MSFT_s114
                                 + 0.950576 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.838097 w_k1_ADM_s114
                                 + 0.757837 w_k1_AXP_s114
                                 + 0.582268 w_k1_DIS_s114
                                 + 0.779118 w_k1_FDX_s114
                                 + 0.932733 w_k1_IBM_s114
                                 + 0.869704 w_k1_INTC_s114
                                 + 0.694103 w_k1_KO_s114
                                 + 0.821944 w_k1_MAT_s114
                                 + 0.733243 w_k1_MSFT_s114
                                 + 0.802943 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.919005 w_k1_ADM_s114
                                 + 0.85899 w_k1_AXP_s114
                                 + 0.887192 w_k1_DIS_s114
                                 + 0.98068 w_k1_FDX_s114
                                 + 0.638715 w_k1_IBM_s114
                                 + 1.09087 w_k1_INTC_s114 + 0.8407 w_k1_KO_s114
                                 + 0.735268 w_k1_MAT_s114
                                 + 1.07686 w_k1_MSFT_s114
                                 + 0.867335 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.03329 w_k1_ADM_s115
                                 + 0.752401 w_k1_AXP_s115
                                 + 0.684435 w_k1_DIS_s115
                                 + 0.791239 w_k1_FDX_s115
                                 + 0.83714 w_k1_IBM_s115
                                 + 0.699696 w_k1_INTC_s115
                                 + 0.762295 w_k1_KO_s115
                                 + 0.883293 w_k1_MAT_s115
                                 + 0.737486 w_k1_MSFT_s115
                                 + 0.863413 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.898127 w_k1_ADM_s115
                                 + 0.744023 w_k1_AXP_s115
                                 + 0.877955 w_k1_DIS_s115
                                 + 0.802521 w_k1_FDX_s115
                                 + 1.17167 w_k1_IBM_s115
                                 + 0.815926 w_k1_INTC_s115
                                 + 0.951752 w_k1_KO_s115
                                 + 1.20531 w_k1_MAT_s115
                                 + 0.793144 w_k1_MSFT_s115
                                 + 0.801097 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.951514 w_k1_ADM_s115
                                 + 1.08332 w_k1_AXP_s115
                                 + 0.932914 w_k1_DIS_s115
                                 + 0.9541 w_k1_FDX_s115 + 1.19188 w_k1_IBM_s115
                                 + 0.825503 w_k1_INTC_s115
                                 + 0.973631 w_k1_KO_s115
                                 + 1.34189 w_k1_MAT_s115
                                 + 0.951429 w_k1_MSFT_s115
                                 + 0.915831 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.21752 w_k1_ADM_s115
                                 + 1.40755 w_k1_AXP_s115
                                 + 1.47959 w_k1_DIS_s115
                                 + 1.20209 w_k1_FDX_s115
                                 + 1.44949 w_k1_IBM_s115
                                 + 0.941206 w_k1_INTC_s115
                                 + 1.20578 w_k1_KO_s115 + 1.47222 w_k1_MAT_s115
                                 + 1.12231 w_k1_MSFT_s115
                                 + 1.07932 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.17945 w_k1_ADM_s115
                                 + 1.54742 w_k1_AXP_s115
                                 + 1.05513 w_k1_DIS_s115
                                 + 0.995294 w_k1_FDX_s115
                                 + 1.15741 w_k1_IBM_s115
                                 + 0.790677 w_k1_INTC_s115
                                 + 1.1245 w_k1_KO_s115 + 1.59525 w_k1_MAT_s115
                                 + 0.798959 w_k1_MSFT_s115
                                 + 1.04735 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.914761 w_k1_ADM_s115
                                 + 1.18652 w_k1_AXP_s115
                                 + 0.97251 w_k1_DIS_s115
                                 + 1.08016 w_k1_FDX_s115
                                 + 0.914723 w_k1_IBM_s115
                                 + 0.953531 w_k1_INTC_s115
                                 + 0.797951 w_k1_KO_s115
                                 + 1.18848 w_k1_MAT_s115
                                 + 0.947194 w_k1_MSFT_s115
                                 + 0.981453 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.865113 w_k1_ADM_s115
                                 + 0.853053 w_k1_AXP_s115
                                 + 0.842228 w_k1_DIS_s115
                                 + 0.947837 w_k1_FDX_s115
                                 + 0.671534 w_k1_IBM_s115
                                 + 0.893032 w_k1_INTC_s115
                                 + 0.745139 w_k1_KO_s115
                                 + 0.758278 w_k1_MAT_s115
                                 + 0.926424 w_k1_MSFT_s115
                                 + 0.87728 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.10527 w_k1_ADM_s115
                                 + 1.20637 w_k1_AXP_s115
                                 + 1.34475 w_k1_DIS_s115
                                 + 1.19654 w_k1_FDX_s115
                                 + 0.930476 w_k1_IBM_s115
                                 + 1.13355 w_k1_INTC_s115
                                 + 1.03508 w_k1_KO_s115 + 1.05031 w_k1_MAT_s115
                                 + 1.08107 w_k1_MSFT_s115
                                 + 1.12203 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.838077 w_k1_ADM_s116
                                 + 0.797662 w_k1_AXP_s116
                                 + 0.64901 w_k1_DIS_s116
                                 + 0.815157 w_k1_FDX_s116
                                 + 0.743768 w_k1_IBM_s116
                                 + 0.700128 w_k1_INTC_s116
                                 + 0.602702 w_k1_KO_s116
                                 + 0.911519 w_k1_MAT_s116
                                 + 0.736534 w_k1_MSFT_s116
                                 + 0.893891 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.03913 w_k1_ADM_s116
                                 + 1.00489 w_k1_AXP_s116
                                 + 0.926592 w_k1_DIS_s116
                                 + 1.06066 w_k1_FDX_s116
                                 + 0.738139 w_k1_IBM_s116
                                 + 0.892084 w_k1_INTC_s116
                                 + 0.705227 w_k1_KO_s116
                                 + 0.773559 w_k1_MAT_s116
                                 + 1.01955 w_k1_MSFT_s116
                                 + 0.953911 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.52542 w_k1_ADM_s116
                                 + 1.69011 w_k1_AXP_s116
                                 + 1.76091 w_k1_DIS_s116
                                 + 1.37961 w_k1_FDX_s116
                                 + 1.40788 w_k1_IBM_s116
                                 + 1.25208 w_k1_INTC_s116
                                 + 1.28054 w_k1_KO_s116 + 1.45849 w_k1_MAT_s116
                                 + 1.06388 w_k1_MSFT_s116
                                 + 1.41302 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.0266 w_k1_ADM_s116
                                 + 1.24842 w_k1_AXP_s116
                                 + 1.32429 w_k1_DIS_s116 + 1.0489 w_k1_FDX_s116
                                 + 1.1761 w_k1_IBM_s116
                                 + 0.922744 w_k1_INTC_s116
                                 + 1.10602 w_k1_KO_s116 + 1.51468 w_k1_MAT_s116
                                 + 1.12398 w_k1_MSFT_s116
                                 + 1.12246 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.91462 w_k1_ADM_s116
                                 + 1.03858 w_k1_AXP_s116
                                 + 0.775635 w_k1_DIS_s116
                                 + 0.891121 w_k1_FDX_s116
                                 + 1.22538 w_k1_IBM_s116
                                 + 0.92273 w_k1_INTC_s116
                                 + 0.917052 w_k1_KO_s116
                                 + 1.52528 w_k1_MAT_s116
                                 + 0.763116 w_k1_MSFT_s116
                                 + 0.952548 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.07815 w_k1_ADM_s116
                                 + 1.19703 w_k1_AXP_s116
                                 + 1.50598 w_k1_DIS_s116
                                 + 1.14183 w_k1_FDX_s116
                                 + 1.56449 w_k1_IBM_s116
                                 + 0.985842 w_k1_INTC_s116
                                 + 1.3403 w_k1_KO_s116 + 1.70858 w_k1_MAT_s116
                                 + 1.05727 w_k1_MSFT_s116
                                 + 1.21598 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.07361 w_k1_ADM_s116
                                 + 0.850429 w_k1_AXP_s116
                                 + 1.09747 w_k1_DIS_s116
                                 + 1.01033 w_k1_FDX_s116
                                 + 1.08153 w_k1_IBM_s116
                                 + 1.00967 w_k1_INTC_s116
                                 + 0.85291 w_k1_KO_s116
                                 + 0.842173 w_k1_MAT_s116
                                 + 1.0133 w_k1_MSFT_s116 + 1.03597 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.27869 w_k1_ADM_s116
                                 + 1.21956 w_k1_AXP_s116
                                 + 1.22887 w_k1_DIS_s116
                                 + 1.30595 w_k1_FDX_s116
                                 + 1.02981 w_k1_IBM_s116
                                 + 1.12984 w_k1_INTC_s116
                                 + 1.13029 w_k1_KO_s116 + 1.12172 w_k1_MAT_s116
                                 + 1.18845 w_k1_MSFT_s116
                                 + 1.22258 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.733966 w_k1_ADM_s117
                                 + 0.963702 w_k1_AXP_s117
                                 + 0.909182 w_k1_DIS_s117
                                 + 0.914966 w_k1_FDX_s117
                                 + 1.02678 w_k1_IBM_s117
                                 + 0.872183 w_k1_INTC_s117
                                 + 0.879348 w_k1_KO_s117
                                 + 0.775388 w_k1_MAT_s117
                                 + 0.87545 w_k1_MSFT_s117
                                 + 0.847529 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.907318 w_k1_ADM_s117
                                 + 1.29901 w_k1_AXP_s117
                                 + 1.48123 w_k1_DIS_s117
                                 + 1.32276 w_k1_FDX_s117
                                 + 1.01536 w_k1_IBM_s117
                                 + 1.26817 w_k1_INTC_s117
                                 + 1.12309 w_k1_KO_s117 + 1.02116 w_k1_MAT_s117
                                 + 1.0694 w_k1_MSFT_s117 + 1.01774 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.913316 w_k1_ADM_s117
                                 + 1.12808 w_k1_AXP_s117
                                 + 0.978304 w_k1_DIS_s117
                                 + 0.955346 w_k1_FDX_s117
                                 + 1.36708 w_k1_IBM_s117
                                 + 0.873448 w_k1_INTC_s117
                                 + 1.11854 w_k1_KO_s117 + 1.3999 w_k1_MAT_s117
                                 + 1.04356 w_k1_MSFT_s117
                                 + 0.894123 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.920648 w_k1_ADM_s117
                                 + 1.32271 w_k1_AXP_s117
                                 + 1.10084 w_k1_DIS_s117
                                 + 1.06679 w_k1_FDX_s117
                                 + 1.14162 w_k1_IBM_s117
                                 + 0.954562 w_k1_INTC_s117
                                 + 0.913888 w_k1_KO_s117
                                 + 1.01168 w_k1_MAT_s117
                                 + 0.848895 w_k1_MSFT_s117
                                 + 0.981321 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.542247 w_k1_ADM_s117
                                 + 0.752329 w_k1_AXP_s117
                                 + 0.821655 w_k1_DIS_s117
                                 + 0.775338 w_k1_FDX_s117
                                 + 1.09054 w_k1_IBM_s117
                                 + 0.791249 w_k1_INTC_s117
                                 + 0.978782 w_k1_KO_s117
                                 + 0.825481 w_k1_MAT_s117
                                 + 0.801703 w_k1_MSFT_s117
                                 + 0.9146 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.21824 w_k1_ADM_s117
                                 + 1.56842 w_k1_AXP_s117
                                 + 1.51016 w_k1_DIS_s117
                                 + 1.18757 w_k1_FDX_s117
                                 + 1.62485 w_k1_IBM_s117
                                 + 1.12074 w_k1_INTC_s117
                                 + 1.38914 w_k1_KO_s117
                                 + 0.907137 w_k1_MAT_s117
                                 + 1.00904 w_k1_MSFT_s117
                                 + 1.17468 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.05162 w_k1_ADM_s117
                                 + 0.858426 w_k1_AXP_s117
                                 + 1.21629 w_k1_DIS_s117
                                 + 1.00558 w_k1_FDX_s117
                                 + 0.896624 w_k1_IBM_s117
                                 + 0.815681 w_k1_INTC_s117
                                 + 0.965642 w_k1_KO_s117
                                 + 0.584311 w_k1_MAT_s117
                                 + 1.05006 w_k1_MSFT_s117
                                 + 1.05539 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.803365 w_k1_ADM_s117
                                 + 0.83995 w_k1_AXP_s117
                                 + 0.709692 w_k1_DIS_s117
                                 + 0.824106 w_k1_FDX_s117
                                 + 0.610394 w_k1_IBM_s117
                                 + 0.837822 w_k1_INTC_s117
                                 + 0.692924 w_k1_KO_s117
                                 + 0.61371 w_k1_MAT_s117
                                 + 0.695777 w_k1_MSFT_s117
                                 + 0.774917 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0365156757080504 w_k1_ADM_s000
                                 + 0.0129069315874275 w_k1_ADM_s001
                                 + 0.00960703750700271 w_k1_ADM_s002
                                 + 0.0187231843736184 w_k1_ADM_s003
                                 + 0.0196647025485496 w_k1_ADM_s004
                                 + 0.022760846356751 w_k1_ADM_s005
                                 + 0.0518460844052528 w_k1_ADM_s006
                                 + 0.0224218393302209 w_k1_ADM_s007
                                 + 0.0291831462479306 w_k1_ADM_s010
                                 + 0.0153516774334007 w_k1_ADM_s011
                                 + 0.0141003143852856 w_k1_ADM_s012
                                 + 0.0223997844535128 w_k1_ADM_s013
                                 + 0.034110547371292 w_k1_ADM_s014
                                 + 0.0120036515044323 w_k1_ADM_s015
                                 + 0.0161213655938029 w_k1_ADM_s016
                                 + 0.0125766508722736 w_k1_ADM_s017
                                 + 0.105713010655027 w_k1_ADM_s100
                                 + 0.0499727612137418 w_k1_ADM_s101
                                 + 0.0778756969235617 w_k1_ADM_s102
                                 + 0.12945644715882 w_k1_ADM_s103
                                 + 0.062220296460544 w_k1_ADM_s104
                                 + 0.0318493592636184 w_k1_ADM_s105
                                 + 0.0337381239520496 w_k1_ADM_s106
                                 + 0.0276393471190283 w_k1_ADM_s107
                                 + 0.0287068953857793 w_k1_ADM_s110
                                 + 0.0107080163727284 w_k1_ADM_s111
                                 + 0.0109543914825612 w_k1_ADM_s112
                                 + 0.0114030522291108 w_k1_ADM_s113
                                 + 0.0181506584239421 w_k1_ADM_s114
                                 + 0.0363846857278288 w_k1_ADM_s115
                                 + 0.0366324645154097 w_k1_ADM_s116
                                 + 0.0221453464483424 w_k1_ADM_s117
                                 + 0.0353874755773146 w_k1_AXP_s000
                                 + 0.0121462279058654 w_k1_AXP_s001
                                 + 0.0120565005068635 w_k1_AXP_s002
                                 + 0.0178460812364381 w_k1_AXP_s003
                                 + 0.0180348551225941 w_k1_AXP_s004
                                 + 0.0219110542800883 w_k1_AXP_s005
                                 + 0.0502398844828818 w_k1_AXP_s006
                                 + 0.0228879852941263 w_k1_AXP_s007
                                 + 0.0330551607565431 w_k1_AXP_s010
                                 + 0.0146849182342997 w_k1_AXP_s011
                                 + 0.0151092815630055 w_k1_AXP_s012
                                 + 0.0217313020524102 w_k1_AXP_s013
                                 + 0.0353952497071654 w_k1_AXP_s014
                                 + 0.0121431386476683 w_k1_AXP_s015
                                 + 0.0177936857239034 w_k1_AXP_s016
                                 + 0.0133545526315922 w_k1_AXP_s017
                                 + 0.112917828417801 w_k1_AXP_s100
                                 + 0.0548866849163052 w_k1_AXP_s101
                                 + 0.0753179024485764 w_k1_AXP_s102
                                 + 0.123622402443913 w_k1_AXP_s103
                                 + 0.0622619259001553 w_k1_AXP_s104
                                 + 0.0317771943501732 w_k1_AXP_s105
                                 + 0.0345850679730985 w_k1_AXP_s106
                                 + 0.0300685388302883 w_k1_AXP_s107
                                 + 0.0269843769168897 w_k1_AXP_s110
                                 + 0.0112300480495515 w_k1_AXP_s111
                                 + 0.0112547116102664 w_k1_AXP_s112
                                 + 0.00939670958475355 w_k1_AXP_s113
                                 + 0.0186545709439759 w_k1_AXP_s114
                                 + 0.0372833873251789 w_k1_AXP_s115
                                 + 0.0368937781873486 w_k1_AXP_s116
                                 + 0.026082755822486 w_k1_AXP_s117
                                 + 0.033676098197331 w_k1_DIS_s000
                                 + 0.0127789984313426 w_k1_DIS_s001
                                 + 0.0107614126933105 w_k1_DIS_s002
                                 + 0.0190954065371585 w_k1_DIS_s003
                                 + 0.0191066721878829 w_k1_DIS_s004
                                 + 0.0203025133186808 w_k1_DIS_s005
                                 + 0.0466887082652757 w_k1_DIS_s006
                                 + 0.0241288452515539 w_k1_DIS_s007
                                 + 0.0354210016525417 w_k1_DIS_s010
                                 + 0.0152644678304816 w_k1_DIS_s011
                                 + 0.0157309101017994 w_k1_DIS_s012
                                 + 0.0214387702801666 w_k1_DIS_s013
                                 + 0.0334044347359901 w_k1_DIS_s014
                                 + 0.0119926873834203 w_k1_DIS_s015
                                 + 0.0154326453113637 w_k1_DIS_s016
                                 + 0.0136608408375968 w_k1_DIS_s017
                                 + 0.104826662152673 w_k1_DIS_s100
                                 + 0.0565034518003966 w_k1_DIS_s101
                                 + 0.0728140901165846 w_k1_DIS_s102
                                 + 0.115366293905955 w_k1_DIS_s103
                                 + 0.063627746853949 w_k1_DIS_s104
                                 + 0.0338986958074715 w_k1_DIS_s105
                                 + 0.0341840367307632 w_k1_DIS_s106
                                 + 0.0287205721766644 w_k1_DIS_s107
                                 + 0.0268192257521334 w_k1_DIS_s110
                                 + 0.0112921408424702 w_k1_DIS_s111
                                 + 0.0118387825075947 w_k1_DIS_s112
                                 + 0.00954759491780621 w_k1_DIS_s113
                                 + 0.0181077209457091 w_k1_DIS_s114
                                 + 0.0363077209805209 w_k1_DIS_s115
                                 + 0.0363594571779728 w_k1_DIS_s116
                                 + 0.0268968771858594 w_k1_DIS_s117
                                 + 0.0321062991826028 w_k1_FDX_s000
                                 + 0.0127033319457622 w_k1_FDX_s001
                                 + 0.011455143242021 w_k1_FDX_s002
                                 + 0.0176989912240308 w_k1_FDX_s003
                                 + 0.018345823233151 w_k1_FDX_s004
                                 + 0.0199638312436918 w_k1_FDX_s005
                                 + 0.0465971626486821 w_k1_FDX_s006
                                 + 0.0229710901407627 w_k1_FDX_s007
                                 + 0.0339724999231735 w_k1_FDX_s010
                                 + 0.0147787975636562 w_k1_FDX_s011
                                 + 0.014627154917027 w_k1_FDX_s012
                                 + 0.0210540043291082 w_k1_FDX_s013
                                 + 0.0341417397134639 w_k1_FDX_s014
                                 + 0.0119041335393841 w_k1_FDX_s015
                                 + 0.0164906566248115 w_k1_FDX_s016
                                 + 0.0131282591460419 w_k1_FDX_s017
                                 + 0.101365042775344 w_k1_FDX_s100
                                 + 0.0557968055595868 w_k1_FDX_s101
                                 + 0.072204055335521 w_k1_FDX_s102
                                 + 0.113127587769081 w_k1_FDX_s103
                                 + 0.0604357443901764 w_k1_FDX_s104
                                 + 0.0336190189361326 w_k1_FDX_s105
                                 + 0.0326821342190993 w_k1_FDX_s106
                                 + 0.0292715779668557 w_k1_FDX_s107
                                 + 0.0244341196198202 w_k1_FDX_s110
                                 + 0.0113402741643099 w_k1_FDX_s111
                                 + 0.0112528982198668 w_k1_FDX_s112
                                 + 0.00942366299720633 w_k1_FDX_s113
                                 + 0.0190771029764407 w_k1_FDX_s114
                                 + 0.035336564175018 w_k1_FDX_s115
                                 + 0.0358988805121921 w_k1_FDX_s116
                                 + 0.024866834029903 w_k1_FDX_s117
                                 + 0.0320910066624949 w_k1_IBM_s000
                                 + 0.0129555625530687 w_k1_IBM_s001
                                 + 0.0110567114220168 w_k1_IBM_s002
                                 + 0.0189947176940458 w_k1_IBM_s003
                                 + 0.0194874162535099 w_k1_IBM_s004
                                 + 0.0207158420031775 w_k1_IBM_s005
                                 + 0.0487969397963646 w_k1_IBM_s006
                                 + 0.0230615588243212 w_k1_IBM_s007
                                 + 0.0323892824766311 w_k1_IBM_s010
                                 + 0.0156881377801725 w_k1_IBM_s011
                                 + 0.0156573892473993 w_k1_IBM_s012
                                 + 0.0202829437682343 w_k1_IBM_s013
                                 + 0.0354184642919126 w_k1_IBM_s014
                                 + 0.0114199086968721 w_k1_IBM_s015
                                 + 0.0161995441665716 w_k1_IBM_s016
                                 + 0.0133791839250752 w_k1_IBM_s017
                                 + 0.110066729981108 w_k1_IBM_s100
                                 + 0.054022965408875 w_k1_IBM_s101
                                 + 0.0715510559526968 w_k1_IBM_s102
                                 + 0.110733485228785 w_k1_IBM_s103
                                 + 0.0581107668781361 w_k1_IBM_s104
                                 + 0.0341084044942976 w_k1_IBM_s105
                                 + 0.0362352416835898 w_k1_IBM_s106
                                 + 0.0295437066233437 w_k1_IBM_s107
                                 + 0.0252931935204126 w_k1_IBM_s110
                                 + 0.0120098606210328 w_k1_IBM_s111
                                 + 0.0122205048389499 w_k1_IBM_s112
                                 + 0.00971946717275515 w_k1_IBM_s113
                                 + 0.0190096577748558 w_k1_IBM_s114
                                 + 0.0372036067347411 w_k1_IBM_s115
                                 + 0.0365065504650307 w_k1_IBM_s116
                                 + 0.0259439801571359 w_k1_IBM_s117
                                 + 0.0317036935351953 w_k1_INTC_s000
                                 + 0.014785402582816 w_k1_INTC_s001
                                 + 0.0130447857902043 w_k1_INTC_s002
                                 + 0.0174721139053017 w_k1_INTC_s003
                                 + 0.0165554632050062 w_k1_INTC_s004
                                 + 0.020990766259129 w_k1_INTC_s005
                                 + 0.0439497005422495 w_k1_INTC_s006
                                 + 0.0213910331317692 w_k1_INTC_s007
                                 + 0.0303543918997379 w_k1_INTC_s010
                                 + 0.0156793682724014 w_k1_INTC_s011
                                 + 0.0139510819536656 w_k1_INTC_s012
                                 + 0.022043081288519 w_k1_INTC_s013
                                 + 0.0313899427864355 w_k1_INTC_s014
                                 + 0.0108688028936181 w_k1_INTC_s015
                                 + 0.015201651887743 w_k1_INTC_s016
                                 + 0.0128551651166056 w_k1_INTC_s017
                                 + 0.0884310333191369 w_k1_INTC_s100
                                 + 0.0585739931734244 w_k1_INTC_s101
                                 + 0.0591693917174804 w_k1_INTC_s102
                                 + 0.112811759153522 w_k1_INTC_s103
                                 + 0.0614722162032515 w_k1_INTC_s104
                                 + 0.0342651183509544 w_k1_INTC_s105
                                 + 0.0301187395560942 w_k1_INTC_s106
                                 + 0.0324813623385488 w_k1_INTC_s107
                                 + 0.0266863952312553 w_k1_INTC_s110
                                 + 0.0115256161217375 w_k1_INTC_s111
                                 + 0.0132108028509198 w_k1_INTC_s112
                                 + 0.00912298512599434 w_k1_INTC_s113
                                 + 0.0197606991981747 w_k1_INTC_s114
                                 + 0.0316317555295802 w_k1_INTC_s115
                                 + 0.0326370013260027 w_k1_INTC_s116
                                 + 0.0230167238386779 w_k1_INTC_s117
                                 + 0.0325432971295978 w_k1_KO_s000
                                 + 0.0138572711709835 w_k1_KO_s001
                                 + 0.0108621621042013 w_k1_KO_s002
                                 + 0.0185234841214655 w_k1_KO_s003
                                 + 0.0201343423810634 w_k1_KO_s004
                                 + 0.020936317351278 w_k1_KO_s005
                                 + 0.0469342791937773 w_k1_KO_s006
                                 + 0.0234900164008872 w_k1_KO_s007
                                 + 0.0338196593218239 w_k1_KO_s010
                                 + 0.0157958860354222 w_k1_KO_s011
                                 + 0.0137832742124373 w_k1_KO_s012
                                 + 0.021836610302832 w_k1_KO_s013
                                 + 0.0297811679320306 w_k1_KO_s014
                                 + 0.0107531816885804 w_k1_KO_s015
                                 + 0.015448175945031 w_k1_KO_s016
                                 + 0.0138658088060004 w_k1_KO_s017
                                 + 0.109437465195641 w_k1_KO_s100
                                 + 0.0522293734509529 w_k1_KO_s101
                                 + 0.06086894627644 w_k1_KO_s102
                                 + 0.111009735676054 w_k1_KO_s103
                                 + 0.0583625224164936 w_k1_KO_s104
                                 + 0.0348889978497028 w_k1_KO_s105
                                 + 0.0324045067338194 w_k1_KO_s106
                                 + 0.0334017043803224 w_k1_KO_s107
                                 + 0.025475180319012 w_k1_KO_s110
                                 + 0.0117297627506412 w_k1_KO_s111
                                 + 0.0130071808339857 w_k1_KO_s112
                                 + 0.00929569472825606 w_k1_KO_s113
                                 + 0.0183906639049284 w_k1_KO_s114
                                 + 0.0334752019005962 w_k1_KO_s115
                                 + 0.0318283144921458 w_k1_KO_s116
                                 + 0.0244429109102757 w_k1_KO_s117
                                 + 0.0318530797671111 w_k1_MAT_s000
                                 + 0.0104840813653189 w_k1_MAT_s001
                                 + 0.0116492193899837 w_k1_MAT_s002
                                 + 0.0189604488794513 w_k1_MAT_s003
                                 + 0.0176752278484097 w_k1_MAT_s004
                                 + 0.0205580134255371 w_k1_MAT_s005
                                 + 0.0503766707197481 w_k1_MAT_s006
                                 + 0.0266880477314169 w_k1_MAT_s007
                                 + 0.0373630439243808 w_k1_MAT_s010
                                 + 0.012800288538068 w_k1_MAT_s011
                                 + 0.0163403506571196 w_k1_MAT_s012
                                 + 0.0244878793778165 w_k1_MAT_s013
                                 + 0.0372953746931458 w_k1_MAT_s014
                                 + 0.0139024680510658 w_k1_MAT_s015
                                 + 0.0189532060036858 w_k1_MAT_s016
                                 + 0.0141125459051912 w_k1_MAT_s017
                                 + 0.0908460914707169 w_k1_MAT_s100
                                 + 0.0624142143760568 w_k1_MAT_s101
                                 + 0.0668476646991482 w_k1_MAT_s102
                                 + 0.118161719588174 w_k1_MAT_s103
                                 + 0.0679190238815273 w_k1_MAT_s104
                                 + 0.0324297665267628 w_k1_MAT_s105
                                 + 0.0442906919517249 w_k1_MAT_s106
                                 + 0.0326153711204927 w_k1_MAT_s107
                                 + 0.0244280509868091 w_k1_MAT_s110
                                 + 0.0119079234486179 w_k1_MAT_s111
                                 + 0.0118980079164304 w_k1_MAT_s112
                                 + 0.00747342359292983 w_k1_MAT_s113
                                 + 0.018442765744401 w_k1_MAT_s114
                                 + 0.0412469630749165 w_k1_MAT_s115
                                 + 0.0402461300549263 w_k1_MAT_s116
                                 + 0.0213216004204367 w_k1_MAT_s117
                                 + 0.0310713692935849 w_k1_MSFT_s000
                                 + 0.0144386321180997 w_k1_MSFT_s001
                                 + 0.0118940246816246 w_k1_MSFT_s002
                                 + 0.0178136773710231 w_k1_MSFT_s003
                                 + 0.0167881681489819 w_k1_MSFT_s004
                                 + 0.0205732027015325 w_k1_MSFT_s005
                                 + 0.0482649770473544 w_k1_MSFT_s006
                                 + 0.0218986495562611 w_k1_MSFT_s007
                                 + 0.0338603805173411 w_k1_MSFT_s010
                                 + 0.0158310999674876 w_k1_MSFT_s011
                                 + 0.01424606007284 w_k1_MSFT_s012
                                 + 0.0218627302929966 w_k1_MSFT_s013
                                 + 0.0339432094949299 w_k1_MSFT_s014
                                 + 0.0112047339628906 w_k1_MSFT_s015
                                 + 0.0141703011310518 w_k1_MSFT_s016
                                 + 0.0134807369185229 w_k1_MSFT_s017
                                 + 0.0997532436517174 w_k1_MSFT_s100
                                 + 0.0580630299001055 w_k1_MSFT_s101
                                 + 0.0709364913461734 w_k1_MSFT_s102
                                 + 0.112432989428534 w_k1_MSFT_s103
                                 + 0.0656231883338888 w_k1_MSFT_s104
                                 + 0.0329636772516895 w_k1_MSFT_s105
                                 + 0.0337289500850436 w_k1_MSFT_s106
                                 + 0.0308091414539584 w_k1_MSFT_s107
                                 + 0.0277691240059454 w_k1_MSFT_s110
                                 + 0.0123282336351966 w_k1_MSFT_s111
                                 + 0.01082974281949 w_k1_MSFT_s112
                                 + 0.0104145543788644 w_k1_MSFT_s113
                                 + 0.0186438870351442 w_k1_MSFT_s114
                                 + 0.0329094130287836 w_k1_MSFT_s115
                                 + 0.0327546047137189 w_k1_MSFT_s116
                                 + 0.0230851742409115 w_k1_MSFT_s117
                                 + 0.0333345955054901 w_k1_PG_s000
                                 + 0.0121456118178695 w_k1_PG_s001
                                 + 0.0105968154676688 w_k1_PG_s002
                                 + 0.0196409986559058 w_k1_PG_s003
                                 + 0.018806301118496 w_k1_PG_s004
                                 + 0.0210235709918342 w_k1_PG_s005
                                 + 0.0460099571937075 w_k1_PG_s006
                                 + 0.0217662470583989 w_k1_PG_s007
                                 + 0.0338933889254064 w_k1_PG_s010
                                 + 0.0158588714178319 w_k1_PG_s011
                                 + 0.016093917370979 w_k1_PG_s012
                                 + 0.0212083674668363 w_k1_PG_s013
                                 + 0.0324060370371181 w_k1_PG_s014
                                 + 0.0115525477773919 w_k1_PG_s015
                                 + 0.0158766712593608 w_k1_PG_s016
                                 + 0.0137597396867783 w_k1_PG_s017
                                 + 0.10235174576227 w_k1_PG_s100
                                 + 0.0499550874409227 w_k1_PG_s101
                                 + 0.0688999255446005 w_k1_PG_s102
                                 + 0.116045365894318 w_k1_PG_s103
                                 + 0.0617893395718944 w_k1_PG_s104
                                 + 0.035915934307057 w_k1_PG_s105
                                 + 0.0336900544816224 w_k1_PG_s106
                                 + 0.0279592154246588 w_k1_PG_s107
                                 + 0.0252025940200817 w_k1_PG_s110
                                 + 0.0119986134809424 w_k1_PG_s111
                                 + 0.0130261103960386 w_k1_PG_s112
                                 + 0.0088127058416304 w_k1_PG_s113
                                 + 0.0186348738799476 w_k1_PG_s114
                                 + 0.0340841709308986 w_k1_PG_s115
                                 + 0.036743139230881 w_k1_PG_s116
                                 + 0.0238588581052427 w_k1_PG_s117 >= 1.09
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
