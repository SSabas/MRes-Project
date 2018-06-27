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
 w_k1_ADM_s0_return_constraint:  1.06931 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.930211 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.00974 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.12048 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.14662 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.960117 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.43494 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.20844 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.954996 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.782069 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.606528 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.909334 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.656969 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.27009 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.19059 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.952248 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.21796 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.36772 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.05439 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.887008 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.623238 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.18339 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.11946 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.981258 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.714179 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.58694 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.832812 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.911896 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.12398 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.989835 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.32583 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.00086 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.722782 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.873443 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.30281 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.858528 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.758518 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.03156 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.08857 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.02521 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.950695 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.02298 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.19019 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.792028 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.43246 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.32292 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.859542 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.806125 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.50213 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.937944 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.660064 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.33938 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.63499 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.05981 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.58389 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.88611 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.23798 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.856835 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.609949 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.08134 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.16511 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.01153 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.645642 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.64903 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.667106 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.918919 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.40599 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.962949 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.46519 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.957636 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.708226 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.02187 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.44419 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.861581 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.733988 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.17375 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.12013 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.983237 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.922243 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.1615 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.3258 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.797245 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.8292 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.40742 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.963673 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.698757 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.43704 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.93031 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.632037 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.09777 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.72303 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.844653 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.10448 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 2.14576 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.26562 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.767984 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.454992 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.18574 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.980082 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.821959 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.496 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 2.00797 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.658405 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.909616 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.29613 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.95746 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.66701 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.993949 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.604511 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.860322 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.50848 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.919924 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.602936 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.02042 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.10843 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.972731 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.919311 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.13064 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.11262 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.857839 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.4189 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.30983 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.01367 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.854525 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.651418 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.883359 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.738102 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.09395 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.42406 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.901422 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.20811 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.42824 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.13443 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.885822 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.659925 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.12633 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.0748 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.9759 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.642454 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.54762 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.796342 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.09329 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.27514 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.885057 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.50462 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.08809 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.699112 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.888434 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.32541 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.858956 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.76053 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.05518 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.886403 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.11816 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.11503 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.929425 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.28177 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.763322 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.81235 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.18222 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.74055 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.916262 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.517686 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.29986 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.712073 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.66119 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.19448 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.13047 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.1126 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 2.10491 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.49833 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.67621 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.627785 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.987976 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.36987 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.830158 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.957308 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.40039 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.670823 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.62182 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.00061 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.07499 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.864131 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.724089 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.02034 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.60341 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.3631 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.20239 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.663554 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.02985 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.08144 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.03997 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.929599 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.14113 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.958608 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.838833 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.30001 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.41181 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.987317 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.971612 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.732973 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.848887 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.749804 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.09947 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.23655 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.936944 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.04688 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.10727 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.09194 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.833019 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.657893 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.99699 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.19842 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.00496 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.758522 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.45032 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.846367 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.05184 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.31962 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.03422 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.38851 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.00135 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.708109 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.00425 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.08808 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.915958 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.726438 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.893239 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.97242 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.05055 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.06253 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.04154 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.2082 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.841773 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.51348 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.1191 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.768741 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.80252 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.548912 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.08261 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.693459 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.43166 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.26226 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.01966 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.13856 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.91377 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.28558 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.743571 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.594314 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.19152 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.13018 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.907829 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.750761 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.42921 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.83592 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.777286 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.07179 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.07957 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.19808 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.903324 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.768306 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.02437 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.19557 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.976452 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.669661 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.904214 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.00404 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.0665 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.953339 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.809364 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.26263 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.819364 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.66023 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.15193 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.521819 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.946838 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.51512 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.39046 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.647192 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 2.07882 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.4078 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.13263 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.47153 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 2.81017 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.53386 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.74136 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.602693 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.00359 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.30329 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.846451 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.936026 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.71192 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.705124 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.925663 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.3447 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.14064 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.960627 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.667086 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.963245 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.37004 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.61774 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.11569 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.642028 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.22277 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09865 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.961568 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.911063 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.13629 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.09628 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.899232 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.36313 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.24755 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.948778 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.865342 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.694974 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.949353 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.829175 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.09302 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.2433 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.854682 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.994576 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.2613 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.02974 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.91778 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.64127 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.990492 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.0464 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.00531 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.750687 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.45725 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.808817 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.07211 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.30829 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.03805 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.44191 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.05841 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.731088 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.837241 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.37174 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.867352 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.779649 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.03728 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.02644 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.0062 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.960327 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.03071 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.10112 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.835374 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.31419 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.17449 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.98141 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.969014 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.704541 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.965623 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.779288 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.21783 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.21088 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.996603 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.09217 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.40054 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.16501 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.816153 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.774426 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.11093 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.0936 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.00356 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.811192 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.2667 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.839539 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.959957 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.11045 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.00989 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.37326 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.999677 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.930887 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.04524 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.37947 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.01777 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.821154 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.13197 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.500001340840299 z_k1_s0000
                                 - 0.269000738622132 z_k1_s0001
                                 - 0.108000324121606 z_k1_s0002
                                 - 0.121000307233302 z_k1_s0003
                                 - 0.115000394643522 z_k1_s0004
                                 - 0.104000382395086 z_k1_s0005
                                 - 0.348000795224444 z_k1_s0006
                                 - 0.160000429068896 z_k1_s0007
                                 - 0.577999058159211 z_k1_s0010
                                 - 0.655000001367483 z_k1_s0011
                                 - 0.466000508038645 z_k1_s0012
                                 - 0.209000121279114 z_k1_s0013
                                 - 0.172000123571333 z_k1_s0014
                                 - 0.2720001593221 z_k1_s0015
                                 - 0.195000147314012 z_k1_s0016
                                 - 0.557000494011834 z_k1_s0017
                                 - 0.552999573550527 z_k1_s0020
                                 - 0.496999396013739 z_k1_s0021
                                 - 0.431000343309802 z_k1_s0022
                                 - 0.112999993490262 z_k1_s0023
                                 - 0.157000026226297 z_k1_s0024
                                 - 0.328998863403375 z_k1_s0025
                                 - 0.260000646349352 z_k1_s0026
                                 - 0.150999953825921 z_k1_s0027
                                 - 0.201000210357675 z_k1_s0030
                                 - 0.339000084177662 z_k1_s0031
                                 - 0.106000156566272 z_k1_s0032
                                 - 0.416999688271317 z_k1_s0033
                                 - 0.27200036955871 z_k1_s0034
                                 - 0.0960000049618011 z_k1_s0035
                                 - 0.323000438850969 z_k1_s0036
                                 - 0.379000263995216 z_k1_s0037
                                 - 0.119999816631304 z_k1_s0040
                                 - 0.208999431141508 z_k1_s0041
                                 - 0.183999761301278 z_k1_s0042
                                 - 0.193999374437698 z_k1_s0043
                                 - 0.144000034757237 z_k1_s0044
                                 - 0.0779999126603065 z_k1_s0045
                                 - 0.164999349743555 z_k1_s0046
                                 - 0.179999406447365 z_k1_s0047
                                 - 0.201999227672605 z_k1_s0050
                                 - 0.336998728295328 z_k1_s0051
                                 - 0.282999606043882 z_k1_s0052
                                 - 0.0899996105820115 z_k1_s0053
                                 - 0.341998960910111 z_k1_s0054
                                 - 0.183000038730536 z_k1_s0055
                                 - 0.168999387409142 z_k1_s0056
                                 - 0.0889997335584655 z_k1_s0057
                                 - 0.124999410956588 z_k1_s0060
                                 - 0.243998000003785 z_k1_s0061
                                 - 0.342999071263465 z_k1_s0062
                                 - 0.132999518906772 z_k1_s0063
                                 - 0.430998141724759 z_k1_s0064
                                 - 0.130999386069689 z_k1_s0065
                                 - 0.143999402714433 z_k1_s0066
                                 - 0.565996523274177 z_k1_s0067
                                 - 0.146999884828596 z_k1_s0070
                                 - 0.0509999132365346 z_k1_s0071
                                 - 0.0629999009980577 z_k1_s0072
                                 - 0.0789998151802115 z_k1_s0073
                                 - 0.0459999530858385 z_k1_s0074
                                 - 0.0859998427910384 z_k1_s0075
                                 - 0.124999698766173 z_k1_s0076
                                 - 0.0979996915528075 z_k1_s0077
                                 - 0.347002327774159 z_k1_s0100
                                 - 0.202000944908875 z_k1_s0101
                                 - 0.195000883380228 z_k1_s0102
                                 - 0.173000617775605 z_k1_s0103
                                 - 0.194000802361546 z_k1_s0104
                                 - 0.335001861152098 z_k1_s0105
                                 - 0.451002158480912 z_k1_s0106
                                 - 0.631001573779832 z_k1_s0107
                                 - 0.966999988905921 z_k1_s0110
                                 - 0.344999497684821 z_k1_s0111
                                 - 0.217999528705408 z_k1_s0112
                                 - 0.555999060257461 z_k1_s0113
                                 - 0.517000999229567 z_k1_s0114
                                 - 0.325999531267388 z_k1_s0115
                                 - 0.23799966531299 z_k1_s0116
                                 - 0.916999647386966 z_k1_s0117
                                 - 0.114000204329887 z_k1_s0120
                                 - 0.120000275136806 z_k1_s0121
                                 - 0.146000277699702 z_k1_s0122
                                 - 0.117000582033815 z_k1_s0123
                                 - 0.0810001571923049 z_k1_s0124
                                 - 0.255000156790127 z_k1_s0125
                                 - 0.20200063429719 z_k1_s0126
                                 - 0.106000185987434 z_k1_s0127
                                 - 0.111000410819651 z_k1_s0130
                                 - 0.0960004968713129 z_k1_s0131
                                 - 0.171000066613001 z_k1_s0132
                                 - 0.0440000676823869 z_k1_s0133
                                 - 0.0950000370072229 z_k1_s0134
                                 - 0.164999926433341 z_k1_s0135
                                 - 0.126999911630452 z_k1_s0136
                                 - 0.0640000984471082 z_k1_s0137
                                 - 0.166000426007905 z_k1_s0140
                                 - 0.0879999460439531 z_k1_s0141
                                 - 0.173000229923904 z_k1_s0142
                                 - 0.059999968979968 z_k1_s0143
                                 - 0.265000190819855 z_k1_s0144
                                 - 0.210000462479881 z_k1_s0145
                                 - 0.0750000563999588 z_k1_s0146
                                 - 0.231999846215877 z_k1_s0147
                                 - 0.330001032850413 z_k1_s0150
                                 - 0.278000998094986 z_k1_s0151
                                 - 0.406002166696311 z_k1_s0152
                                 - 0.321001011792696 z_k1_s0153
                                 - 1.09900222373564 z_k1_s0154
                                 - 0.493002630988378 z_k1_s0155
                                 - 0.738004073340018 z_k1_s0156
                                 - 0.246000705944655 z_k1_s0157
                                 - 0.553000685054294 z_k1_s0160
                                 - 0.892998813160343 z_k1_s0161
                                 - 0.395998543604622 z_k1_s0162
                                 - 0.437000800948425 z_k1_s0163
                                 - 0.404000067816179 z_k1_s0164
                                 - 0.208999744046 z_k1_s0165
                                 - 0.227999689704781 z_k1_s0166
                                 - 0.297999525479175 z_k1_s0167
                                 - 0.440000821083768 z_k1_s0170
                                 - 0.273000278422024 z_k1_s0171
                                 - 0.341999349340125 z_k1_s0172
                                 - 0.231000297318744 z_k1_s0173
                                 - 0.35600032387627 z_k1_s0174
                                 - 0.465001414896305 z_k1_s0175
                                 - 0.177000245185821 z_k1_s0176
                                 - 0.391000085211946 z_k1_s0177
                                 - 1.10099940552591 z_k1_s1000
                                 - 0.433999756653728 z_k1_s1001
                                 - 0.49199954122977 z_k1_s1002
                                 - 0.746001090663526 z_k1_s1003
                                 - 0.562000340285914 z_k1_s1004
                                 - 0.790000242928853 z_k1_s1005
                                 - 0.263999638586831 z_k1_s1006
                                 - 0.335999582974102 z_k1_s1007
                                 - 0.216999723070727 z_k1_s1010
                                 - 0.547999212303152 z_k1_s1011
                                 - 0.313998539025423 z_k1_s1012
                                 - 0.399000551064109 z_k1_s1013
                                 - 0.267999560795775 z_k1_s1014
                                 - 0.140999696363173 z_k1_s1015
                                 - 0.502000507973137 z_k1_s1016
                                 - 0.349999906758763 z_k1_s1017
                                 - 0.553999226850487 z_k1_s1020
                                 - 0.526997560294776 z_k1_s1021
                                 - 0.430999320887743 z_k1_s1022
                                 - 0.864997476683249 z_k1_s1023
                                 - 0.692999150980706 z_k1_s1024
                                 - 0.637000022193451 z_k1_s1025
                                 - 0.592999037433981 z_k1_s1026
                                 - 0.373999489865838 z_k1_s1027
                                 - 0.472997347651018 z_k1_s1030
                                 - 0.514997414737315 z_k1_s1031
                                 - 0.501996978378794 z_k1_s1032
                                 - 0.442999294582068 z_k1_s1033
                                 - 0.25599897926457 z_k1_s1034
                                 - 0.307998979205051 z_k1_s1035
                                 - 0.757998052235688 z_k1_s1036
                                 - 0.235999113556201 z_k1_s1037
                                 - 1.16999890980924 z_k1_s1040
                                 - 0.397999784818715 z_k1_s1041
                                 - 0.345999620249625 z_k1_s1042
                                 - 0.263999618821191 z_k1_s1043
                                 - 0.241999770410974 z_k1_s1044
                                 - 0.788998920707591 z_k1_s1045
                                 - 0.456999639718586 z_k1_s1046
                                 - 1.12700094900574 z_k1_s1047
                                 - 0.439999941067635 z_k1_s1050
                                 - 0.125999947969385 z_k1_s1051
                                 - 0.307998669747956 z_k1_s1052
                                 - 0.201999867481101 z_k1_s1053
                                 - 0.326000835199694 z_k1_s1054
                                 - 0.39799918501154 z_k1_s1055
                                 - 0.531000719856247 z_k1_s1056
                                 - 0.246999867313015 z_k1_s1057
                                 - 1.11100125508166 z_k1_s1060
                                 - 0.590000838426721 z_k1_s1061
                                 - 0.924998995958231 z_k1_s1062
                                 - 0.294999903013602 z_k1_s1063
                                 - 0.641001374605764 z_k1_s1064
                                 - 0.392999911040474 z_k1_s1065
                                 - 0.509999683593076 z_k1_s1066
                                 - 0.696998724450931 z_k1_s1067
                                 - 0.890003795427978 z_k1_s1070
                                 - 1.07400192385827 z_k1_s1071
                                 - 0.607000401073379 z_k1_s1072
                                 - 0.516000687715524 z_k1_s1073
                                 - 0.5080001919529 z_k1_s1074
                                 - 1.41500116926102 z_k1_s1075
                                 - 0.541000261659726 z_k1_s1076
                                 - 0.769999519491016 z_k1_s1077
                                 - 0.137000135239862 z_k1_s1100
                                 - 0.226000740590321 z_k1_s1101
                                 - 0.206000210919069 z_k1_s1102
                                 - 0.14300023195118 z_k1_s1103
                                 - 0.239999722448921 z_k1_s1104
                                 - 0.328000726281207 z_k1_s1105
                                 - 0.593000851693954 z_k1_s1106
                                 - 0.200000114207751 z_k1_s1107
                                 - 0.955005125926699 z_k1_s1110
                                 - 0.392000541242933 z_k1_s1111
                                 - 0.362001768310108 z_k1_s1112
                                 - 0.300001297294298 z_k1_s1113
                                 - 0.476000919153408 z_k1_s1114
                                 - 0.237001197212335 z_k1_s1115
                                 - 0.370002101155408 z_k1_s1116
                                 - 0.575003067091898 z_k1_s1117
                                 - 0.441000889769006 z_k1_s1120
                                 - 0.549001222569624 z_k1_s1121
                                 - 0.161999936200704 z_k1_s1122
                                 - 0.19199996608974 z_k1_s1123
                                 - 0.266000152416073 z_k1_s1124
                                 - 0.219000049289895 z_k1_s1125
                                 - 0.737001072074115 z_k1_s1126
                                 - 0.249000079178931 z_k1_s1127
                                 - 0.176000189108842 z_k1_s1130
                                 - 0.408000724143495 z_k1_s1131
                                 - 0.188999410947954 z_k1_s1132
                                 - 0.0940001052035903 z_k1_s1133
                                 - 0.193000650715228 z_k1_s1134
                                 - 0.309999933873153 z_k1_s1135
                                 - 0.168000113276559 z_k1_s1136
                                 - 0.311000243814972 z_k1_s1137
                                 - 0.442001507978048 z_k1_s1140
                                 - 0.315001242576142 z_k1_s1141
                                 - 0.713003514836167 z_k1_s1142
                                 - 0.34400115251225 z_k1_s1143
                                 - 0.839003638505268 z_k1_s1144
                                 - 0.25700095600223 z_k1_s1145
                                 - 1.16000573147824 z_k1_s1146
                                 - 0.597000141478172 z_k1_s1147
                                 - 0.397001141226335 z_k1_s1150
                                 - 0.216000104601498 z_k1_s1151
                                 - 0.201999993210612 z_k1_s1152
                                 - 0.218999993970912 z_k1_s1153
                                 - 0.396000460769536 z_k1_s1154
                                 - 0.52400147920049 z_k1_s1155
                                 - 0.603001435263037 z_k1_s1156
                                 - 0.133000116712983 z_k1_s1157
                                 - 0.623000401953498 z_k1_s1160
                                 - 0.240000298581728 z_k1_s1161
                                 - 1.17100087050659 z_k1_s1162
                                 - 0.347000442359841 z_k1_s1163
                                 - 0.613999591224747 z_k1_s1164
                                 - 0.847002386297909 z_k1_s1165
                                 - 0.236000336247069 z_k1_s1166
                                 - 1.03900262516329 z_k1_s1167
                                 - 0.468000430167631 z_k1_s1170
                                 - 0.396000537203594 z_k1_s1171
                                 - 0.9110043481798 z_k1_s1172
                                 - 1.78299996605153 z_k1_s1173
                                 - 0.636000680817669 z_k1_s1174
                                 - 1.39700343803681 z_k1_s1175
                                 - 0.978002987150327 z_k1_s1176
                                 - 0.936997658369688 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.892639 w_k1_ADM_s000
                                 + 1.15487 w_k1_AXP_s000
                                 + 1.00431 w_k1_DIS_s000
                                 + 0.964535 w_k1_FDX_s000
                                 + 0.883839 w_k1_IBM_s000
                                 + 1.17443 w_k1_INTC_s000
                                 + 0.979491 w_k1_KO_s000 + 1.1127 w_k1_MAT_s000
                                 + 1.09706 w_k1_MSFT_s000
                                 + 0.981824 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.606606 w_k1_ADM_s000
                                 + 0.766593 w_k1_AXP_s000
                                 + 0.632734 w_k1_DIS_s000
                                 + 0.696152 w_k1_FDX_s000
                                 + 0.686848 w_k1_IBM_s000
                                 + 0.692184 w_k1_INTC_s000
                                 + 0.646146 w_k1_KO_s000
                                 + 0.667266 w_k1_MAT_s000
                                 + 0.722291 w_k1_MSFT_s000
                                 + 0.733222 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.03981 w_k1_ADM_s000
                                 + 1.38993 w_k1_AXP_s000
                                 + 1.01617 w_k1_DIS_s000
                                 + 0.99989 w_k1_FDX_s000
                                 + 1.26445 w_k1_IBM_s000
                                 + 0.95323 w_k1_INTC_s000 + 1.0693 w_k1_KO_s000
                                 + 1.56538 w_k1_MAT_s000
                                 + 0.910418 w_k1_MSFT_s000
                                 + 0.953642 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.01748 w_k1_ADM_s000
                                 + 1.40872 w_k1_AXP_s000
                                 + 1.65573 w_k1_DIS_s000
                                 + 1.34811 w_k1_FDX_s000
                                 + 1.05477 w_k1_IBM_s000
                                 + 1.19424 w_k1_INTC_s000
                                 + 1.41459 w_k1_KO_s000 + 1.45904 w_k1_MAT_s000
                                 + 1.27285 w_k1_MSFT_s000
                                 + 1.19038 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.39456 w_k1_ADM_s000
                                 + 2.0756 w_k1_AXP_s000 + 2.12979 w_k1_DIS_s000
                                 + 1.53446 w_k1_FDX_s000
                                 + 1.32394 w_k1_IBM_s000
                                 + 1.22969 w_k1_INTC_s000
                                 + 1.51055 w_k1_KO_s000 + 1.90831 w_k1_MAT_s000
                                 + 1.50519 w_k1_MSFT_s000
                                 + 1.48626 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.980956 w_k1_ADM_s000
                                 + 1.65663 w_k1_AXP_s000
                                 + 1.06362 w_k1_DIS_s000
                                 + 1.18617 w_k1_FDX_s000
                                 + 0.741786 w_k1_IBM_s000
                                 + 1.21979 w_k1_INTC_s000
                                 + 0.831626 w_k1_KO_s000
                                 + 0.84136 w_k1_MAT_s000
                                 + 1.03426 w_k1_MSFT_s000
                                 + 0.990238 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.648703 w_k1_ADM_s000
                                 + 0.895851 w_k1_AXP_s000
                                 + 0.932564 w_k1_DIS_s000
                                 + 0.797505 w_k1_FDX_s000
                                 + 1.10116 w_k1_IBM_s000
                                 + 1.06552 w_k1_INTC_s000
                                 + 1.14721 w_k1_KO_s000 + 1.27462 w_k1_MAT_s000
                                 + 1.00625 w_k1_MSFT_s000
                                 + 1.02301 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.21192 w_k1_ADM_s000
                                 + 1.44319 w_k1_AXP_s000
                                 + 1.20062 w_k1_DIS_s000
                                 + 1.06119 w_k1_FDX_s000
                                 + 1.65174 w_k1_IBM_s000
                                 + 0.782986 w_k1_INTC_s000
                                 + 1.31608 w_k1_KO_s000 + 2.36621 w_k1_MAT_s000
                                 + 0.988889 w_k1_MSFT_s000
                                 + 1.16034 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.32199 w_k1_ADM_s001
                                 + 1.28865 w_k1_AXP_s001
                                 + 1.53679 w_k1_DIS_s001
                                 + 1.26849 w_k1_FDX_s001
                                 + 1.21394 w_k1_IBM_s001
                                 + 1.14763 w_k1_INTC_s001
                                 + 1.48777 w_k1_KO_s001 + 1.1881 w_k1_MAT_s001
                                 + 1.27189 w_k1_MSFT_s001
                                 + 1.24069 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.0156 w_k1_ADM_s001
                                 + 1.20912 w_k1_AXP_s001
                                 + 1.09827 w_k1_DIS_s001
                                 + 1.02065 w_k1_FDX_s001
                                 + 0.929898 w_k1_IBM_s001
                                 + 0.781549 w_k1_INTC_s001
                                 + 1.02914 w_k1_KO_s001 + 1.31498 w_k1_MAT_s001
                                 + 0.959341 w_k1_MSFT_s001
                                 + 1.17019 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.732006 w_k1_ADM_s001
                                 + 0.754885 w_k1_AXP_s001
                                 + 0.633053 w_k1_DIS_s001
                                 + 0.747506 w_k1_FDX_s001
                                 + 0.838638 w_k1_IBM_s001
                                 + 0.602933 w_k1_INTC_s001
                                 + 0.881678 w_k1_KO_s001
                                 + 0.851341 w_k1_MAT_s001
                                 + 0.80612 w_k1_MSFT_s001
                                 + 0.99513 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.778911 w_k1_ADM_s001
                                 + 0.748639 w_k1_AXP_s001
                                 + 1.06774 w_k1_DIS_s001
                                 + 0.86077 w_k1_FDX_s001
                                 + 1.11923 w_k1_IBM_s001
                                 + 0.797451 w_k1_INTC_s001
                                 + 1.1608 w_k1_KO_s001 + 1.07487 w_k1_MAT_s001
                                 + 1.06247 w_k1_MSFT_s001
                                 + 0.968347 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.05475 w_k1_ADM_s001
                                 + 0.97783 w_k1_AXP_s001
                                 + 0.829576 w_k1_DIS_s001
                                 + 0.87369 w_k1_FDX_s001
                                 + 1.05791 w_k1_IBM_s001
                                 + 0.716965 w_k1_INTC_s001
                                 + 0.890913 w_k1_KO_s001
                                 + 0.936127 w_k1_MAT_s001
                                 + 0.820212 w_k1_MSFT_s001
                                 + 0.83871 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.991822 w_k1_ADM_s001
                                 + 1.14135 w_k1_AXP_s001
                                 + 1.17814 w_k1_DIS_s001
                                 + 0.961749 w_k1_FDX_s001
                                 + 1.46787 w_k1_IBM_s001
                                 + 0.932195 w_k1_INTC_s001
                                 + 1.38858 w_k1_KO_s001 + 1.34886 w_k1_MAT_s001
                                 + 0.980942 w_k1_MSFT_s001
                                 + 1.21248 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.850811 w_k1_ADM_s001
                                 + 0.777397 w_k1_AXP_s001
                                 + 0.637545 w_k1_DIS_s001
                                 + 0.668621 w_k1_FDX_s001
                                 + 1.0194 w_k1_IBM_s001
                                 + 0.638583 w_k1_INTC_s001
                                 + 0.894693 w_k1_KO_s001
                                 + 1.09964 w_k1_MAT_s001
                                 + 0.734982 w_k1_MSFT_s001
                                 + 0.890806 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.814053 w_k1_ADM_s001
                                 + 0.794419 w_k1_AXP_s001
                                 + 0.903151 w_k1_DIS_s001
                                 + 0.950479 w_k1_FDX_s001
                                 + 0.662968 w_k1_IBM_s001
                                 + 0.874683 w_k1_INTC_s001
                                 + 0.871096 w_k1_KO_s001
                                 + 0.60814 w_k1_MAT_s001
                                 + 0.890933 w_k1_MSFT_s001
                                 + 0.829705 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.978549 w_k1_ADM_s002
                                 + 0.907339 w_k1_AXP_s002
                                 + 1.14698 w_k1_DIS_s002 + 1.0868 w_k1_FDX_s002
                                 + 0.882758 w_k1_IBM_s002
                                 + 1.12664 w_k1_INTC_s002
                                 + 1.23188 w_k1_KO_s002 + 1.12559 w_k1_MAT_s002
                                 + 1.24791 w_k1_MSFT_s002
                                 + 0.932147 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.13428 w_k1_ADM_s002
                                 + 1.13472 w_k1_AXP_s002
                                 + 1.23708 w_k1_DIS_s002
                                 + 1.09473 w_k1_FDX_s002
                                 + 1.41773 w_k1_IBM_s002
                                 + 1.0315 w_k1_INTC_s002 + 1.29051 w_k1_KO_s002
                                 + 1.80971 w_k1_MAT_s002
                                 + 1.14371 w_k1_MSFT_s002
                                 + 1.02582 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.861172 w_k1_ADM_s002
                                 + 0.937723 w_k1_AXP_s002
                                 + 0.822664 w_k1_DIS_s002
                                 + 0.795039 w_k1_FDX_s002
                                 + 1.0652 w_k1_IBM_s002
                                 + 0.893022 w_k1_INTC_s002
                                 + 1.01748 w_k1_KO_s002 + 1.70087 w_k1_MAT_s002
                                 + 1.0382 w_k1_MSFT_s002
                                 + 0.918086 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.20712 w_k1_ADM_s002
                                 + 1.58282 w_k1_AXP_s002
                                 + 1.32343 w_k1_DIS_s002
                                 + 1.21302 w_k1_FDX_s002
                                 + 1.68535 w_k1_IBM_s002
                                 + 1.00027 w_k1_INTC_s002
                                 + 1.44059 w_k1_KO_s002 + 3.04086 w_k1_MAT_s002
                                 + 1.11114 w_k1_MSFT_s002
                                 + 1.26559 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.28835 w_k1_ADM_s002
                                 + 1.48359 w_k1_AXP_s002
                                 + 1.73658 w_k1_DIS_s002
                                 + 1.40168 w_k1_FDX_s002
                                 + 1.38592 w_k1_IBM_s002
                                 + 1.35779 w_k1_INTC_s002
                                 + 1.79492 w_k1_KO_s002 + 1.9654 w_k1_MAT_s002
                                 + 1.48911 w_k1_MSFT_s002
                                 + 1.34011 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.649538 w_k1_ADM_s002
                                 + 0.737045 w_k1_AXP_s002
                                 + 0.650622 w_k1_DIS_s002
                                 + 0.736392 w_k1_FDX_s002
                                 + 0.699346 w_k1_IBM_s002
                                 + 0.911098 w_k1_INTC_s002
                                 + 0.825504 w_k1_KO_s002
                                 + 0.867333 w_k1_MAT_s002
                                 + 0.803626 w_k1_MSFT_s002
                                 + 0.767605 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.06143 w_k1_ADM_s002
                                 + 0.941366 w_k1_AXP_s002
                                 + 0.775779 w_k1_DIS_s002
                                 + 1.00505 w_k1_FDX_s002
                                 + 0.782716 w_k1_IBM_s002
                                 + 0.995108 w_k1_INTC_s002
                                 + 0.897781 w_k1_KO_s002
                                 + 0.88254 w_k1_MAT_s002
                                 + 0.998767 w_k1_MSFT_s002
                                 + 0.954354 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.0767 w_k1_ADM_s002
                                 + 1.27186 w_k1_AXP_s002
                                 + 0.888342 w_k1_DIS_s002
                                 + 0.934622 w_k1_FDX_s002
                                 + 1.1874 w_k1_IBM_s002
                                 + 0.946629 w_k1_INTC_s002
                                 + 1.1203 w_k1_KO_s002 + 2.22813 w_k1_MAT_s002
                                 + 0.945281 w_k1_MSFT_s002
                                 + 1.01743 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.43453 w_k1_ADM_s003
                                 + 1.45829 w_k1_AXP_s003
                                 + 1.60617 w_k1_DIS_s003 + 1.2685 w_k1_FDX_s003
                                 + 1.45751 w_k1_IBM_s003
                                 + 0.983475 w_k1_INTC_s003
                                 + 1.45314 w_k1_KO_s003 + 1.48511 w_k1_MAT_s003
                                 + 1.21606 w_k1_MSFT_s003 + 1.3032 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.16296 w_k1_ADM_s003
                                 + 1.17585 w_k1_AXP_s003
                                 + 1.34046 w_k1_DIS_s003
                                 + 1.28755 w_k1_FDX_s003
                                 + 1.00768 w_k1_IBM_s003
                                 + 1.16095 w_k1_INTC_s003 + 1.0339 w_k1_KO_s003
                                 + 0.883024 w_k1_MAT_s003
                                 + 1.20386 w_k1_MSFT_s003
                                 + 0.969222 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.24331 w_k1_ADM_s003
                                 + 1.11329 w_k1_AXP_s003
                                 + 1.18364 w_k1_DIS_s003
                                 + 1.17175 w_k1_FDX_s003
                                 + 0.747089 w_k1_IBM_s003
                                 + 1.19615 w_k1_INTC_s003
                                 + 1.05141 w_k1_KO_s003
                                 + 0.690528 w_k1_MAT_s003
                                 + 1.11631 w_k1_MSFT_s003
                                 + 0.992043 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.14049 w_k1_ADM_s003
                                 + 0.772496 w_k1_AXP_s003
                                 + 0.875514 w_k1_DIS_s003
                                 + 0.911015 w_k1_FDX_s003
                                 + 0.747991 w_k1_IBM_s003
                                 + 0.807228 w_k1_INTC_s003
                                 + 0.917441 w_k1_KO_s003
                                 + 0.726802 w_k1_MAT_s003
                                 + 1.01178 w_k1_MSFT_s003
                                 + 0.920382 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.06978 w_k1_ADM_s003
                                 + 1.04589 w_k1_AXP_s003
                                 + 1.24855 w_k1_DIS_s003
                                 + 0.933882 w_k1_FDX_s003
                                 + 1.32514 w_k1_IBM_s003
                                 + 0.841856 w_k1_INTC_s003
                                 + 1.25321 w_k1_KO_s003 + 1.13877 w_k1_MAT_s003
                                 + 1.02137 w_k1_MSFT_s003
                                 + 0.946865 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.885469 w_k1_ADM_s003
                                 + 1.29872 w_k1_AXP_s003
                                 + 1.52552 w_k1_DIS_s003
                                 + 1.04187 w_k1_FDX_s003
                                 + 1.73917 w_k1_IBM_s003
                                 + 1.09815 w_k1_INTC_s003
                                 + 1.42462 w_k1_KO_s003 + 1.682 w_k1_MAT_s003
                                 + 1.27174 w_k1_MSFT_s003
                                 + 0.97064 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.10675 w_k1_ADM_s003
                                 + 1.25406 w_k1_AXP_s003
                                 + 0.954614 w_k1_DIS_s003
                                 + 1.00055 w_k1_FDX_s003
                                 + 1.11769 w_k1_IBM_s003
                                 + 0.943246 w_k1_INTC_s003
                                 + 0.964255 w_k1_KO_s003 + 1.1643 w_k1_MAT_s003
                                 + 0.914075 w_k1_MSFT_s003
                                 + 0.950512 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.0037 w_k1_ADM_s003
                                 + 0.88013 w_k1_AXP_s003
                                 + 0.688792 w_k1_DIS_s003
                                 + 0.743655 w_k1_FDX_s003
                                 + 1.00944 w_k1_IBM_s003
                                 + 0.736992 w_k1_INTC_s003
                                 + 0.828398 w_k1_KO_s003
                                 + 0.819469 w_k1_MAT_s003
                                 + 0.822147 w_k1_MSFT_s003
                                 + 0.819091 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.22001 w_k1_ADM_s004
                                 + 1.29489 w_k1_AXP_s004
                                 + 1.66854 w_k1_DIS_s004
                                 + 1.31427 w_k1_FDX_s004
                                 + 1.01004 w_k1_IBM_s004
                                 + 1.29801 w_k1_INTC_s004
                                 + 1.27962 w_k1_KO_s004 + 1.36838 w_k1_MAT_s004
                                 + 1.45345 w_k1_MSFT_s004
                                 + 1.42321 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.986889 w_k1_ADM_s004
                                 + 0.921182 w_k1_AXP_s004
                                 + 0.800425 w_k1_DIS_s004
                                 + 0.781346 w_k1_FDX_s004
                                 + 0.964095 w_k1_IBM_s004
                                 + 0.79677 w_k1_INTC_s004
                                 + 0.750694 w_k1_KO_s004
                                 + 0.896218 w_k1_MAT_s004
                                 + 0.699544 w_k1_MSFT_s004
                                 + 0.843638 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.814952 w_k1_ADM_s004
                                 + 0.740551 w_k1_AXP_s004
                                 + 0.63241 w_k1_DIS_s004
                                 + 0.636621 w_k1_FDX_s004
                                 + 0.871373 w_k1_IBM_s004
                                 + 0.889635 w_k1_INTC_s004
                                 + 0.721649 w_k1_KO_s004
                                 + 0.447906 w_k1_MAT_s004
                                 + 0.676595 w_k1_MSFT_s004
                                 + 0.78226 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.12474 w_k1_ADM_s004
                                 + 0.966019 w_k1_AXP_s004
                                 + 1.02681 w_k1_DIS_s004
                                 + 1.04846 w_k1_FDX_s004
                                 + 0.759781 w_k1_IBM_s004
                                 + 0.967779 w_k1_INTC_s004 + 1.016 w_k1_KO_s004
                                 + 0.688126 w_k1_MAT_s004
                                 + 1.02696 w_k1_MSFT_s004
                                 + 1.00695 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.26881 w_k1_ADM_s004
                                 + 1.33228 w_k1_AXP_s004
                                 + 1.36856 w_k1_DIS_s004
                                 + 1.11398 w_k1_FDX_s004
                                 + 1.50449 w_k1_IBM_s004
                                 + 1.00276 w_k1_INTC_s004
                                 + 1.11027 w_k1_KO_s004
                                 + 0.884262 w_k1_MAT_s004
                                 + 1.11751 w_k1_MSFT_s004
                                 + 1.00723 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.13242 w_k1_ADM_s004
                                 + 0.915787 w_k1_AXP_s004
                                 + 1.03744 w_k1_DIS_s004
                                 + 0.868379 w_k1_FDX_s004
                                 + 1.55745 w_k1_IBM_s004
                                 + 1.01639 w_k1_INTC_s004
                                 + 1.25317 w_k1_KO_s004 + 1.2104 w_k1_MAT_s004
                                 + 0.977464 w_k1_MSFT_s004
                                 + 0.981276 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.14657 w_k1_ADM_s004
                                 + 0.983552 w_k1_AXP_s004
                                 + 0.964303 w_k1_DIS_s004
                                 + 0.927004 w_k1_FDX_s004
                                 + 1.07449 w_k1_IBM_s004
                                 + 1.06851 w_k1_INTC_s004
                                 + 0.984635 w_k1_KO_s004
                                 + 0.705933 w_k1_MAT_s004
                                 + 1.05117 w_k1_MSFT_s004
                                 + 0.946626 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.27571 w_k1_ADM_s004
                                 + 1.3096 w_k1_AXP_s004 + 1.24583 w_k1_DIS_s004
                                 + 1.15769 w_k1_FDX_s004
                                 + 1.05302 w_k1_IBM_s004
                                 + 0.905195 w_k1_INTC_s004
                                 + 0.991673 w_k1_KO_s004
                                 + 1.34069 w_k1_MAT_s004
                                 + 1.07251 w_k1_MSFT_s004
                                 + 1.07154 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.684878 w_k1_ADM_s005
                                 + 0.667952 w_k1_AXP_s005
                                 + 0.78099 w_k1_DIS_s005
                                 + 0.774176 w_k1_FDX_s005
                                 + 0.977438 w_k1_IBM_s005
                                 + 0.972648 w_k1_INTC_s005
                                 + 0.892146 w_k1_KO_s005
                                 + 0.916641 w_k1_MAT_s005
                                 + 0.898152 w_k1_MSFT_s005
                                 + 0.987461 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.962738 w_k1_ADM_s005
                                 + 1.25374 w_k1_AXP_s005 + 1.1642 w_k1_DIS_s005
                                 + 1.00421 w_k1_FDX_s005
                                 + 1.45022 w_k1_IBM_s005
                                 + 1.28724 w_k1_INTC_s005
                                 + 1.02542 w_k1_KO_s005 + 1.12516 w_k1_MAT_s005
                                 + 1.03969 w_k1_MSFT_s005
                                 + 1.13133 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.2037 w_k1_ADM_s005
                                 + 1.48752 w_k1_AXP_s005
                                 + 1.54783 w_k1_DIS_s005
                                 + 1.36433 w_k1_FDX_s005
                                 + 1.00595 w_k1_IBM_s005
                                 + 1.23904 w_k1_INTC_s005
                                 + 1.21677 w_k1_KO_s005 + 1.11196 w_k1_MAT_s005
                                 + 1.24767 w_k1_MSFT_s005
                                 + 1.20341 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.79086 w_k1_ADM_s005
                                 + 0.552946 w_k1_AXP_s005
                                 + 0.614794 w_k1_DIS_s005
                                 + 0.792849 w_k1_FDX_s005
                                 + 0.669079 w_k1_IBM_s005
                                 + 0.848869 w_k1_INTC_s005
                                 + 0.678133 w_k1_KO_s005
                                 + 0.561219 w_k1_MAT_s005
                                 + 0.753651 w_k1_MSFT_s005
                                 + 0.863412 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.11409 w_k1_ADM_s005
                                 + 1.18024 w_k1_AXP_s005
                                 + 1.08605 w_k1_DIS_s005
                                 + 1.11303 w_k1_FDX_s005
                                 + 0.855893 w_k1_IBM_s005
                                 + 1.32045 w_k1_INTC_s005
                                 + 0.960421 w_k1_KO_s005
                                 + 0.752877 w_k1_MAT_s005
                                 + 1.15494 w_k1_MSFT_s005
                                 + 1.02663 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.06595 w_k1_ADM_s005
                                 + 1.0512 w_k1_AXP_s005 + 1.09073 w_k1_DIS_s005
                                 + 1.04903 w_k1_FDX_s005
                                 + 1.04661 w_k1_IBM_s005
                                 + 1.19126 w_k1_INTC_s005
                                 + 1.17149 w_k1_KO_s005
                                 + 0.850711 w_k1_MAT_s005
                                 + 1.07373 w_k1_MSFT_s005
                                 + 1.12897 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.756152 w_k1_ADM_s005
                                 + 0.781685 w_k1_AXP_s005
                                 + 0.783701 w_k1_DIS_s005
                                 + 0.888951 w_k1_FDX_s005
                                 + 0.683838 w_k1_IBM_s005
                                 + 1.00962 w_k1_INTC_s005
                                 + 0.741226 w_k1_KO_s005
                                 + 0.605463 w_k1_MAT_s005
                                 + 0.873828 w_k1_MSFT_s005
                                 + 0.84416 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.913511 w_k1_ADM_s005
                                 + 1.2937 w_k1_AXP_s005 + 1.02024 w_k1_DIS_s005
                                 + 1.07061 w_k1_FDX_s005
                                 + 1.15575 w_k1_IBM_s005
                                 + 1.28187 w_k1_INTC_s005
                                 + 0.972458 w_k1_KO_s005
                                 + 1.55046 w_k1_MAT_s005
                                 + 1.07555 w_k1_MSFT_s005
                                 + 1.02308 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.826764 w_k1_ADM_s006
                                 + 0.915563 w_k1_AXP_s006
                                 + 1.00108 w_k1_DIS_s006
                                 + 0.785168 w_k1_FDX_s006
                                 + 1.71645 w_k1_IBM_s006
                                 + 0.996188 w_k1_INTC_s006
                                 + 1.10907 w_k1_KO_s006 + 1.77426 w_k1_MAT_s006
                                 + 0.918291 w_k1_MSFT_s006
                                 + 1.04472 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.13114 w_k1_ADM_s006
                                 + 0.861568 w_k1_AXP_s006
                                 + 0.745487 w_k1_DIS_s006
                                 + 0.857562 w_k1_FDX_s006
                                 + 0.777596 w_k1_IBM_s006
                                 + 0.842577 w_k1_INTC_s006
                                 + 0.785956 w_k1_KO_s006
                                 + 0.68454 w_k1_MAT_s006
                                 + 0.826353 w_k1_MSFT_s006
                                 + 0.922701 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.87481 w_k1_ADM_s006
                                 + 0.844051 w_k1_AXP_s006
                                 + 0.955336 w_k1_DIS_s006
                                 + 0.866054 w_k1_FDX_s006
                                 + 1.17974 w_k1_IBM_s006
                                 + 0.995222 w_k1_INTC_s006
                                 + 1.03182 w_k1_KO_s006
                                 + 0.891619 w_k1_MAT_s006
                                 + 0.781625 w_k1_MSFT_s006
                                 + 0.997358 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.720972 w_k1_ADM_s006
                                 + 0.592898 w_k1_AXP_s006
                                 + 0.469477 w_k1_DIS_s006
                                 + 0.669491 w_k1_FDX_s006
                                 + 0.806591 w_k1_IBM_s006
                                 + 1.004 w_k1_INTC_s006 + 0.702315 w_k1_KO_s006
                                 + 0.668643 w_k1_MAT_s006
                                 + 0.752283 w_k1_MSFT_s006
                                 + 0.748378 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.898414 w_k1_ADM_s006
                                 + 1.05282 w_k1_AXP_s006
                                 + 0.779548 w_k1_DIS_s006
                                 + 0.986423 w_k1_FDX_s006
                                 + 0.858066 w_k1_IBM_s006
                                 + 1.04309 w_k1_INTC_s006
                                 + 0.865707 w_k1_KO_s006
                                 + 1.04195 w_k1_MAT_s006
                                 + 0.840972 w_k1_MSFT_s006
                                 + 0.966455 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.23869 w_k1_ADM_s006
                                 + 1.27315 w_k1_AXP_s006
                                 + 1.80602 w_k1_DIS_s006
                                 + 1.37293 w_k1_FDX_s006
                                 + 1.20422 w_k1_IBM_s006
                                 + 1.60142 w_k1_INTC_s006
                                 + 1.41273 w_k1_KO_s006 + 1.04046 w_k1_MAT_s006
                                 + 1.16442 w_k1_MSFT_s006
                                 + 1.33065 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.16698 w_k1_ADM_s006
                                 + 1.82099 w_k1_AXP_s006
                                 + 1.39084 w_k1_DIS_s006
                                 + 1.30682 w_k1_FDX_s006
                                 + 1.50529 w_k1_IBM_s006
                                 + 1.33709 w_k1_INTC_s006 + 1.3187 w_k1_KO_s006
                                 + 2.04461 w_k1_MAT_s006
                                 + 1.11337 w_k1_MSFT_s006
                                 + 1.30725 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.12723 w_k1_ADM_s006
                                 + 1.13654 w_k1_AXP_s006
                                 + 1.29702 w_k1_DIS_s006
                                 + 1.22542 w_k1_FDX_s006
                                 + 1.06758 w_k1_IBM_s006
                                 + 1.14966 w_k1_INTC_s006
                                 + 1.22041 w_k1_KO_s006
                                 + 0.926058 w_k1_MAT_s006
                                 + 1.08689 w_k1_MSFT_s006
                                 + 1.12805 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.714537 w_k1_ADM_s007
                                 + 0.854431 w_k1_AXP_s007
                                 + 0.920762 w_k1_DIS_s007
                                 + 0.823594 w_k1_FDX_s007
                                 + 0.986553 w_k1_IBM_s007
                                 + 0.782617 w_k1_INTC_s007
                                 + 0.792049 w_k1_KO_s007
                                 + 0.96939 w_k1_MAT_s007
                                 + 0.751266 w_k1_MSFT_s007
                                 + 0.912686 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.734225 w_k1_ADM_s007
                                 + 1.30142 w_k1_AXP_s007
                                 + 0.991023 w_k1_DIS_s007
                                 + 1.01716 w_k1_FDX_s007
                                 + 1.25458 w_k1_IBM_s007
                                 + 1.15556 w_k1_INTC_s007
                                 + 0.934544 w_k1_KO_s007
                                 + 0.990973 w_k1_MAT_s007
                                 + 0.983248 w_k1_MSFT_s007
                                 + 0.967913 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.989498 w_k1_ADM_s007
                                 + 1.00835 w_k1_AXP_s007
                                 + 1.32531 w_k1_DIS_s007 + 1.1837 w_k1_FDX_s007
                                 + 0.76404 w_k1_IBM_s007
                                 + 0.920492 w_k1_INTC_s007
                                 + 0.921672 w_k1_KO_s007
                                 + 0.782479 w_k1_MAT_s007
                                 + 0.981813 w_k1_MSFT_s007
                                 + 1.20836 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.00662 w_k1_ADM_s007
                                 + 1.17749 w_k1_AXP_s007
                                 + 1.23393 w_k1_DIS_s007
                                 + 1.06978 w_k1_FDX_s007
                                 + 1.76004 w_k1_IBM_s007
                                 + 1.03057 w_k1_INTC_s007
                                 + 1.29249 w_k1_KO_s007 + 1.85329 w_k1_MAT_s007
                                 + 1.091 w_k1_MSFT_s007 + 1.21391 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.998983 w_k1_ADM_s007
                                 + 1.40278 w_k1_AXP_s007
                                 + 1.07735 w_k1_DIS_s007
                                 + 1.22074 w_k1_FDX_s007
                                 + 0.832067 w_k1_IBM_s007
                                 + 0.951526 w_k1_INTC_s007
                                 + 0.810564 w_k1_KO_s007
                                 + 0.908142 w_k1_MAT_s007
                                 + 1.00308 w_k1_MSFT_s007
                                 + 1.03546 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.97692 w_k1_ADM_s007
                                 + 0.994209 w_k1_AXP_s007
                                 + 1.10242 w_k1_DIS_s007
                                 + 1.03447 w_k1_FDX_s007
                                 + 0.822594 w_k1_IBM_s007
                                 + 1.01821 w_k1_INTC_s007
                                 + 0.849481 w_k1_KO_s007
                                 + 0.913451 w_k1_MAT_s007
                                 + 1.07415 w_k1_MSFT_s007
                                 + 0.970474 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.948881 w_k1_ADM_s007
                                 + 1.44307 w_k1_AXP_s007
                                 + 1.53431 w_k1_DIS_s007
                                 + 1.30641 w_k1_FDX_s007
                                 + 1.15217 w_k1_IBM_s007
                                 + 1.17954 w_k1_INTC_s007 + 1.1166 w_k1_KO_s007
                                 + 1.1712 w_k1_MAT_s007
                                 + 1.38541 w_k1_MSFT_s007
                                 + 1.19543 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.768652 w_k1_ADM_s007
                                 + 0.925185 w_k1_AXP_s007
                                 + 0.651238 w_k1_DIS_s007
                                 + 0.886135 w_k1_FDX_s007
                                 + 0.652322 w_k1_IBM_s007
                                 + 0.863203 w_k1_INTC_s007
                                 + 0.582624 w_k1_KO_s007
                                 + 0.512242 w_k1_MAT_s007
                                 + 0.798388 w_k1_MSFT_s007
                                 + 0.791821 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.24445 w_k1_ADM_s010
                                 + 1.3354 w_k1_AXP_s010 + 1.04998 w_k1_DIS_s010
                                 + 1.01507 w_k1_FDX_s010
                                 + 1.45942 w_k1_IBM_s010
                                 + 1.02751 w_k1_INTC_s010
                                 + 1.37583 w_k1_KO_s010 + 1.56554 w_k1_MAT_s010
                                 + 1.01213 w_k1_MSFT_s010
                                 + 1.13328 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.20583 w_k1_ADM_s010
                                 + 0.90203 w_k1_AXP_s010
                                 + 0.972375 w_k1_DIS_s010
                                 + 1.15078 w_k1_FDX_s010
                                 + 0.581595 w_k1_IBM_s010
                                 + 0.942504 w_k1_INTC_s010
                                 + 0.886981 w_k1_KO_s010
                                 + 0.579195 w_k1_MAT_s010
                                 + 1.23898 w_k1_MSFT_s010 + 1.0704 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.51116 w_k1_ADM_s010
                                 + 1.69491 w_k1_AXP_s010
                                 + 1.90699 w_k1_DIS_s010
                                 + 1.49142 w_k1_FDX_s010
                                 + 1.65981 w_k1_IBM_s010
                                 + 1.35158 w_k1_INTC_s010
                                 + 1.65959 w_k1_KO_s010 + 1.52434 w_k1_MAT_s010
                                 + 1.42035 w_k1_MSFT_s010
                                 + 1.30141 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.698944 w_k1_ADM_s010
                                 + 0.579015 w_k1_AXP_s010
                                 + 0.604843 w_k1_DIS_s010
                                 + 0.693687 w_k1_FDX_s010
                                 + 0.776709 w_k1_IBM_s010
                                 + 0.723171 w_k1_INTC_s010
                                 + 0.727441 w_k1_KO_s010
                                 + 0.718684 w_k1_MAT_s010
                                 + 0.746182 w_k1_MSFT_s010
                                 + 0.824985 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.10769 w_k1_ADM_s010
                                 + 0.813975 w_k1_AXP_s010
                                 + 1.05514 w_k1_DIS_s010
                                 + 1.02146 w_k1_FDX_s010
                                 + 1.03292 w_k1_IBM_s010
                                 + 0.945866 w_k1_INTC_s010
                                 + 1.04188 w_k1_KO_s010
                                 + 0.972298 w_k1_MAT_s010
                                 + 1.06968 w_k1_MSFT_s010
                                 + 1.05502 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.01745 w_k1_ADM_s010
                                 + 0.783266 w_k1_AXP_s010
                                 + 0.920957 w_k1_DIS_s010
                                 + 0.944474 w_k1_FDX_s010
                                 + 1.12128 w_k1_IBM_s010
                                 + 0.979951 w_k1_INTC_s010
                                 + 1.13863 w_k1_KO_s010
                                 + 0.945118 w_k1_MAT_s010
                                 + 1.03994 w_k1_MSFT_s010
                                 + 1.03527 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.991104 w_k1_ADM_s010
                                 + 0.84027 w_k1_AXP_s010
                                 + 0.750627 w_k1_DIS_s010
                                 + 0.89939 w_k1_FDX_s010
                                 + 0.799388 w_k1_IBM_s010
                                 + 0.81918 w_k1_INTC_s010
                                 + 0.803871 w_k1_KO_s010
                                 + 0.724859 w_k1_MAT_s010
                                 + 0.90351 w_k1_MSFT_s010
                                 + 0.906373 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.21391 w_k1_ADM_s010
                                 + 1.1839 w_k1_AXP_s010 + 1.29734 w_k1_DIS_s010
                                 + 1.22788 w_k1_FDX_s010
                                 + 1.16311 w_k1_IBM_s010
                                 + 1.34221 w_k1_INTC_s010
                                 + 1.50176 w_k1_KO_s010 + 1.27757 w_k1_MAT_s010
                                 + 1.26369 w_k1_MSFT_s010 + 1.2407 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.907298 w_k1_ADM_s011
                                 + 0.907633 w_k1_AXP_s011
                                 + 0.899828 w_k1_DIS_s011
                                 + 0.908486 w_k1_FDX_s011
                                 + 0.981375 w_k1_IBM_s011
                                 + 1.00565 w_k1_INTC_s011
                                 + 0.992023 w_k1_KO_s011
                                 + 0.944087 w_k1_MAT_s011
                                 + 0.959452 w_k1_MSFT_s011
                                 + 1.01526 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.04176 w_k1_ADM_s011
                                 + 1.32871 w_k1_AXP_s011
                                 + 1.73365 w_k1_DIS_s011
                                 + 1.34949 w_k1_FDX_s011
                                 + 1.10035 w_k1_IBM_s011
                                 + 1.39235 w_k1_INTC_s011
                                 + 1.20351 w_k1_KO_s011 + 1.11095 w_k1_MAT_s011
                                 + 1.3222 w_k1_MSFT_s011 + 1.07542 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.955465 w_k1_ADM_s011
                                 + 0.998398 w_k1_AXP_s011
                                 + 0.731555 w_k1_DIS_s011
                                 + 0.986091 w_k1_FDX_s011
                                 + 0.681695 w_k1_IBM_s011
                                 + 0.757139 w_k1_INTC_s011
                                 + 0.62899 w_k1_KO_s011
                                 + 0.867018 w_k1_MAT_s011
                                 + 0.911888 w_k1_MSFT_s011
                                 + 0.808021 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.735576 w_k1_ADM_s011
                                 + 0.637828 w_k1_AXP_s011
                                 + 0.742279 w_k1_DIS_s011
                                 + 0.770914 w_k1_FDX_s011
                                 + 0.684721 w_k1_IBM_s011
                                 + 0.855389 w_k1_INTC_s011
                                 + 0.8129 w_k1_KO_s011 + 0.690651 w_k1_MAT_s011
                                 + 0.879131 w_k1_MSFT_s011
                                 + 0.843631 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.977905 w_k1_ADM_s011
                                 + 0.924218 w_k1_AXP_s011
                                 + 1.0711 w_k1_DIS_s011 + 1.10846 w_k1_FDX_s011
                                 + 0.828841 w_k1_IBM_s011
                                 + 1.03346 w_k1_INTC_s011
                                 + 0.986392 w_k1_KO_s011
                                 + 0.821396 w_k1_MAT_s011
                                 + 1.21963 w_k1_MSFT_s011
                                 + 1.00975 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.969491 w_k1_ADM_s011
                                 + 0.81216 w_k1_AXP_s011
                                 + 1.43007 w_k1_DIS_s011
                                 + 1.06653 w_k1_FDX_s011
                                 + 1.14713 w_k1_IBM_s011
                                 + 1.1552 w_k1_INTC_s011 + 1.33094 w_k1_KO_s011
                                 + 1.19443 w_k1_MAT_s011
                                 + 1.28523 w_k1_MSFT_s011 + 1.1726 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.623443 w_k1_ADM_s011
                                 + 0.532166 w_k1_AXP_s011
                                 + 0.551394 w_k1_DIS_s011
                                 + 0.660321 w_k1_FDX_s011
                                 + 1.04626 w_k1_IBM_s011
                                 + 0.7203 w_k1_INTC_s011
                                 + 0.826539 w_k1_KO_s011
                                 + 1.11989 w_k1_MAT_s011
                                 + 0.798756 w_k1_MSFT_s011
                                 + 0.837716 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.942944 w_k1_ADM_s011
                                 + 1.39842 w_k1_AXP_s011
                                 + 1.20516 w_k1_DIS_s011
                                 + 1.05133 w_k1_FDX_s011
                                 + 1.34127 w_k1_IBM_s011
                                 + 1.03066 w_k1_INTC_s011
                                 + 1.15497 w_k1_KO_s011 + 1.75105 w_k1_MAT_s011
                                 + 0.99886 w_k1_MSFT_s011 + 1.0085 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.987257 w_k1_ADM_s012
                                 + 1.44213 w_k1_AXP_s012
                                 + 1.26702 w_k1_DIS_s012
                                 + 1.08308 w_k1_FDX_s012
                                 + 1.81651 w_k1_IBM_s012
                                 + 1.06797 w_k1_INTC_s012
                                 + 1.20289 w_k1_KO_s012 + 1.7369 w_k1_MAT_s012
                                 + 1.10605 w_k1_MSFT_s012
                                 + 1.19927 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.848752 w_k1_ADM_s012
                                 + 0.750949 w_k1_AXP_s012
                                 + 0.65754 w_k1_DIS_s012
                                 + 0.784665 w_k1_FDX_s012
                                 + 0.831139 w_k1_IBM_s012
                                 + 0.78655 w_k1_INTC_s012
                                 + 0.772509 w_k1_KO_s012
                                 + 0.635342 w_k1_MAT_s012
                                 + 0.705053 w_k1_MSFT_s012
                                 + 0.818658 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.01154 w_k1_ADM_s012
                                 + 1.41403 w_k1_AXP_s012
                                 + 1.34286 w_k1_DIS_s012
                                 + 1.26248 w_k1_FDX_s012
                                 + 1.23518 w_k1_IBM_s012
                                 + 1.14624 w_k1_INTC_s012
                                 + 1.18456 w_k1_KO_s012 + 1.72293 w_k1_MAT_s012
                                 + 1.19683 w_k1_MSFT_s012
                                 + 1.17175 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.960373 w_k1_ADM_s012
                                 + 0.859014 w_k1_AXP_s012
                                 + 0.786079 w_k1_DIS_s012
                                 + 0.91333 w_k1_FDX_s012
                                 + 0.739279 w_k1_IBM_s012
                                 + 0.811409 w_k1_INTC_s012
                                 + 0.790339 w_k1_KO_s012
                                 + 0.972672 w_k1_MAT_s012
                                 + 0.922326 w_k1_MSFT_s012
                                 + 1.07483 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.09835 w_k1_ADM_s012
                                 + 1.08271 w_k1_AXP_s012
                                 + 0.996472 w_k1_DIS_s012
                                 + 1.15691 w_k1_FDX_s012
                                 + 0.640902 w_k1_IBM_s012
                                 + 1.06661 w_k1_INTC_s012
                                 + 0.73265 w_k1_KO_s012
                                 + 0.618262 w_k1_MAT_s012
                                 + 1.0779 w_k1_MSFT_s012 + 1.04586 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.06887 w_k1_ADM_s012
                                 + 1.00621 w_k1_AXP_s012
                                 + 0.973142 w_k1_DIS_s012
                                 + 1.01165 w_k1_FDX_s012
                                 + 1.03537 w_k1_IBM_s012
                                 + 1.1205 w_k1_INTC_s012 + 1.0781 w_k1_KO_s012
                                 + 0.87852 w_k1_MAT_s012
                                 + 1.03376 w_k1_MSFT_s012
                                 + 1.00157 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.754853 w_k1_ADM_s012
                                 + 0.89572 w_k1_AXP_s012
                                 + 0.838352 w_k1_DIS_s012
                                 + 0.894863 w_k1_FDX_s012
                                 + 1.18773 w_k1_IBM_s012
                                 + 0.919624 w_k1_INTC_s012
                                 + 0.966336 w_k1_KO_s012
                                 + 1.24341 w_k1_MAT_s012
                                 + 0.975211 w_k1_MSFT_s012
                                 + 1.06234 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.37917 w_k1_ADM_s012
                                 + 1.68797 w_k1_AXP_s012
                                 + 1.57855 w_k1_DIS_s012
                                 + 1.45794 w_k1_FDX_s012
                                 + 1.30842 w_k1_IBM_s012
                                 + 1.26761 w_k1_INTC_s012
                                 + 1.14945 w_k1_KO_s012 + 1.41634 w_k1_MAT_s012
                                 + 1.30473 w_k1_MSFT_s012
                                 + 1.19807 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.742892 w_k1_ADM_s013
                                 + 0.592173 w_k1_AXP_s013
                                 + 0.568296 w_k1_DIS_s013
                                 + 0.719008 w_k1_FDX_s013
                                 + 0.716829 w_k1_IBM_s013
                                 + 0.951725 w_k1_INTC_s013
                                 + 0.747959 w_k1_KO_s013
                                 + 0.610839 w_k1_MAT_s013
                                 + 0.834109 w_k1_MSFT_s013
                                 + 0.858783 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.996345 w_k1_ADM_s013
                                 + 0.698914 w_k1_AXP_s013
                                 + 0.995548 w_k1_DIS_s013
                                 + 0.888366 w_k1_FDX_s013
                                 + 0.988898 w_k1_IBM_s013
                                 + 0.83193 w_k1_INTC_s013
                                 + 0.93934 w_k1_KO_s013
                                 + 0.829889 w_k1_MAT_s013
                                 + 0.930227 w_k1_MSFT_s013
                                 + 1.09408 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.39032 w_k1_ADM_s013
                                 + 1.19902 w_k1_AXP_s013
                                 + 1.42376 w_k1_DIS_s013
                                 + 1.26426 w_k1_FDX_s013
                                 + 1.05518 w_k1_IBM_s013
                                 + 1.14523 w_k1_INTC_s013
                                 + 1.21425 w_k1_KO_s013 + 1.14192 w_k1_MAT_s013
                                 + 1.19447 w_k1_MSFT_s013
                                 + 1.31496 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.79821 w_k1_ADM_s013
                                 + 0.967219 w_k1_AXP_s013
                                 + 0.627108 w_k1_DIS_s013
                                 + 0.777548 w_k1_FDX_s013
                                 + 1.01909 w_k1_IBM_s013
                                 + 1.04565 w_k1_INTC_s013
                                 + 0.803068 w_k1_KO_s013
                                 + 0.905692 w_k1_MAT_s013
                                 + 0.807949 w_k1_MSFT_s013
                                 + 0.785184 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.2342 w_k1_ADM_s013
                                 + 1.93173 w_k1_AXP_s013
                                 + 1.32272 w_k1_DIS_s013
                                 + 1.43091 w_k1_FDX_s013
                                 + 1.35313 w_k1_IBM_s013
                                 + 1.91118 w_k1_INTC_s013
                                 + 1.49614 w_k1_KO_s013 + 1.36127 w_k1_MAT_s013
                                 + 1.44413 w_k1_MSFT_s013
                                 + 1.16205 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.813885 w_k1_ADM_s013
                                 + 1.07863 w_k1_AXP_s013
                                 + 0.962531 w_k1_DIS_s013
                                 + 0.951487 w_k1_FDX_s013
                                 + 1.26922 w_k1_IBM_s013
                                 + 1.07646 w_k1_INTC_s013
                                 + 1.10342 w_k1_KO_s013 + 1.02798 w_k1_MAT_s013
                                 + 0.97163 w_k1_MSFT_s013
                                 + 1.06128 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.886948 w_k1_ADM_s013
                                 + 0.794205 w_k1_AXP_s013
                                 + 0.79987 w_k1_DIS_s013
                                 + 0.842783 w_k1_FDX_s013
                                 + 0.794539 w_k1_IBM_s013
                                 + 0.858262 w_k1_INTC_s013
                                 + 0.90953 w_k1_KO_s013
                                 + 0.684414 w_k1_MAT_s013
                                 + 0.960164 w_k1_MSFT_s013
                                 + 0.896732 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.944572 w_k1_ADM_s013
                                 + 1.08783 w_k1_AXP_s013
                                 + 0.846127 w_k1_DIS_s013
                                 + 1.00843 w_k1_FDX_s013
                                 + 0.764355 w_k1_IBM_s013
                                 + 1.02439 w_k1_INTC_s013
                                 + 0.851892 w_k1_KO_s013
                                 + 0.919273 w_k1_MAT_s013
                                 + 1.03146 w_k1_MSFT_s013
                                 + 0.957631 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.05195 w_k1_ADM_s014
                                 + 1.131 w_k1_AXP_s014 + 1.2365 w_k1_DIS_s014
                                 + 1.05309 w_k1_FDX_s014
                                 + 0.79478 w_k1_IBM_s014
                                 + 1.06293 w_k1_INTC_s014
                                 + 0.994937 w_k1_KO_s014
                                 + 0.906087 w_k1_MAT_s014
                                 + 1.14046 w_k1_MSFT_s014
                                 + 0.92022 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.36091 w_k1_ADM_s014
                                 + 1.42512 w_k1_AXP_s014
                                 + 1.27595 w_k1_DIS_s014
                                 + 1.25596 w_k1_FDX_s014
                                 + 0.944787 w_k1_IBM_s014
                                 + 1.01807 w_k1_INTC_s014
                                 + 1.11143 w_k1_KO_s014 + 1.10065 w_k1_MAT_s014
                                 + 1.20536 w_k1_MSFT_s014
                                 + 1.13309 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.10117 w_k1_ADM_s014
                                 + 1.43876 w_k1_AXP_s014
                                 + 1.24961 w_k1_DIS_s014 + 1.0835 w_k1_FDX_s014
                                 + 1.43396 w_k1_IBM_s014
                                 + 1.15478 w_k1_INTC_s014 + 1.2284 w_k1_KO_s014
                                 + 1.41959 w_k1_MAT_s014
                                 + 1.31427 w_k1_MSFT_s014
                                 + 1.03227 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.14979 w_k1_ADM_s014
                                 + 1.32013 w_k1_AXP_s014
                                 + 1.80097 w_k1_DIS_s014
                                 + 1.12509 w_k1_FDX_s014
                                 + 1.67528 w_k1_IBM_s014
                                 + 0.964216 w_k1_INTC_s014
                                 + 1.71109 w_k1_KO_s014 + 1.6556 w_k1_MAT_s014
                                 + 1.12936 w_k1_MSFT_s014
                                 + 1.18927 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.873418 w_k1_ADM_s014
                                 + 0.968187 w_k1_AXP_s014
                                 + 0.860164 w_k1_DIS_s014
                                 + 0.820325 w_k1_FDX_s014
                                 + 1.05834 w_k1_IBM_s014
                                 + 0.860042 w_k1_INTC_s014
                                 + 0.971466 w_k1_KO_s014
                                 + 1.11351 w_k1_MAT_s014
                                 + 1.00355 w_k1_MSFT_s014
                                 + 0.869437 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.803482 w_k1_ADM_s014
                                 + 0.867402 w_k1_AXP_s014
                                 + 0.720921 w_k1_DIS_s014
                                 + 0.802517 w_k1_FDX_s014
                                 + 0.79565 w_k1_IBM_s014
                                 + 0.894612 w_k1_INTC_s014
                                 + 0.75675 w_k1_KO_s014
                                 + 0.689288 w_k1_MAT_s014
                                 + 0.926485 w_k1_MSFT_s014
                                 + 0.832497 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.1395 w_k1_ADM_s014
                                 + 1.65042 w_k1_AXP_s014
                                 + 1.64806 w_k1_DIS_s014
                                 + 1.32944 w_k1_FDX_s014
                                 + 1.06018 w_k1_IBM_s014
                                 + 1.41885 w_k1_INTC_s014
                                 + 1.30398 w_k1_KO_s014 + 1.3295 w_k1_MAT_s014
                                 + 1.47736 w_k1_MSFT_s014
                                 + 1.11328 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.00651 w_k1_ADM_s014
                                 + 0.967586 w_k1_AXP_s014
                                 + 1.15114 w_k1_DIS_s014
                                 + 0.92684 w_k1_FDX_s014
                                 + 1.11046 w_k1_IBM_s014
                                 + 0.829685 w_k1_INTC_s014
                                 + 1.0902 w_k1_KO_s014 + 1.10965 w_k1_MAT_s014
                                 + 1.04245 w_k1_MSFT_s014
                                 + 0.918011 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.962638 w_k1_ADM_s015
                                 + 1.19179 w_k1_AXP_s015
                                 + 1.31433 w_k1_DIS_s015
                                 + 1.27366 w_k1_FDX_s015
                                 + 0.901175 w_k1_IBM_s015
                                 + 1.31286 w_k1_INTC_s015
                                 + 1.16416 w_k1_KO_s015
                                 + 0.819872 w_k1_MAT_s015
                                 + 1.1039 w_k1_MSFT_s015 + 1.16628 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.838372 w_k1_ADM_s015
                                 + 0.682913 w_k1_AXP_s015
                                 + 0.773808 w_k1_DIS_s015
                                 + 0.875578 w_k1_FDX_s015
                                 + 0.722745 w_k1_IBM_s015
                                 + 0.672232 w_k1_INTC_s015
                                 + 0.759142 w_k1_KO_s015
                                 + 0.64527 w_k1_MAT_s015
                                 + 0.835926 w_k1_MSFT_s015
                                 + 0.947336 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.37737 w_k1_ADM_s015
                                 + 1.3498 w_k1_AXP_s015 + 1.61994 w_k1_DIS_s015
                                 + 1.36053 w_k1_FDX_s015
                                 + 1.26999 w_k1_IBM_s015
                                 + 1.36637 w_k1_INTC_s015
                                 + 1.45909 w_k1_KO_s015 + 1.5731 w_k1_MAT_s015
                                 + 1.27239 w_k1_MSFT_s015
                                 + 1.47576 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.29359 w_k1_ADM_s015
                                 + 1.01918 w_k1_AXP_s015 + 1.1596 w_k1_DIS_s015
                                 + 1.17959 w_k1_FDX_s015
                                 + 1.13521 w_k1_IBM_s015
                                 + 1.13766 w_k1_INTC_s015
                                 + 1.21609 w_k1_KO_s015
                                 + 0.887942 w_k1_MAT_s015
                                 + 1.10457 w_k1_MSFT_s015
                                 + 1.23703 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.04594 w_k1_ADM_s015
                                 + 1.12316 w_k1_AXP_s015
                                 + 0.99358 w_k1_DIS_s015
                                 + 0.932683 w_k1_FDX_s015
                                 + 1.22635 w_k1_IBM_s015
                                 + 1.06864 w_k1_INTC_s015
                                 + 1.15958 w_k1_KO_s015 + 1.25779 w_k1_MAT_s015
                                 + 0.866426 w_k1_MSFT_s015
                                 + 1.14382 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.73258 w_k1_ADM_s015
                                 + 0.601469 w_k1_AXP_s015
                                 + 0.574582 w_k1_DIS_s015
                                 + 0.760065 w_k1_FDX_s015
                                 + 0.883391 w_k1_IBM_s015
                                 + 0.825476 w_k1_INTC_s015
                                 + 0.841612 w_k1_KO_s015
                                 + 0.804749 w_k1_MAT_s015
                                 + 0.723465 w_k1_MSFT_s015
                                 + 0.905488 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.02631 w_k1_ADM_s015
                                 + 0.944445 w_k1_AXP_s015
                                 + 0.850525 w_k1_DIS_s015
                                 + 1.05308 w_k1_FDX_s015
                                 + 0.724023 w_k1_IBM_s015
                                 + 0.912372 w_k1_INTC_s015
                                 + 0.819189 w_k1_KO_s015
                                 + 0.722433 w_k1_MAT_s015
                                 + 0.86887 w_k1_MSFT_s015
                                 + 0.96552 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.974587 w_k1_ADM_s015
                                 + 0.868143 w_k1_AXP_s015
                                 + 0.875877 w_k1_DIS_s015
                                 + 0.876948 w_k1_FDX_s015
                                 + 0.925457 w_k1_IBM_s015
                                 + 0.997607 w_k1_INTC_s015
                                 + 1.0104 w_k1_KO_s015 + 1.02236 w_k1_MAT_s015
                                 + 0.786427 w_k1_MSFT_s015
                                 + 1.01755 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.902427 w_k1_ADM_s016
                                 + 0.858349 w_k1_AXP_s016
                                 + 0.886333 w_k1_DIS_s016
                                 + 0.789693 w_k1_FDX_s016
                                 + 1.18782 w_k1_IBM_s016
                                 + 0.925131 w_k1_INTC_s016
                                 + 1.11053 w_k1_KO_s016
                                 + 0.912016 w_k1_MAT_s016
                                 + 0.882699 w_k1_MSFT_s016
                                 + 0.948776 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.21303 w_k1_ADM_s016
                                 + 1.21261 w_k1_AXP_s016
                                 + 1.47895 w_k1_DIS_s016
                                 + 1.18333 w_k1_FDX_s016
                                 + 1.06161 w_k1_IBM_s016
                                 + 1.17499 w_k1_INTC_s016
                                 + 1.23294 w_k1_KO_s016 + 1.27834 w_k1_MAT_s016
                                 + 1.18924 w_k1_MSFT_s016
                                 + 1.00133 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.18424 w_k1_ADM_s016
                                 + 0.780299 w_k1_AXP_s016
                                 + 0.817867 w_k1_DIS_s016
                                 + 0.920476 w_k1_FDX_s016
                                 + 0.574752 w_k1_IBM_s016
                                 + 0.758967 w_k1_INTC_s016
                                 + 0.790944 w_k1_KO_s016
                                 + 0.828387 w_k1_MAT_s016
                                 + 0.913447 w_k1_MSFT_s016
                                 + 0.844296 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.960985 w_k1_ADM_s016
                                 + 0.814387 w_k1_AXP_s016
                                 + 0.745105 w_k1_DIS_s016
                                 + 0.810316 w_k1_FDX_s016
                                 + 0.722484 w_k1_IBM_s016
                                 + 0.852104 w_k1_INTC_s016
                                 + 0.721393 w_k1_KO_s016
                                 + 0.806512 w_k1_MAT_s016
                                 + 0.799912 w_k1_MSFT_s016
                                 + 0.825003 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.15998 w_k1_ADM_s016
                                 + 1.15497 w_k1_AXP_s016
                                 + 1.72397 w_k1_DIS_s016
                                 + 1.16281 w_k1_FDX_s016
                                 + 1.57533 w_k1_IBM_s016
                                 + 1.26116 w_k1_INTC_s016 + 1.8264 w_k1_KO_s016
                                 + 1.32488 w_k1_MAT_s016
                                 + 1.41469 w_k1_MSFT_s016
                                 + 1.19324 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.21531 w_k1_ADM_s016
                                 + 1.29844 w_k1_AXP_s016
                                 + 0.943468 w_k1_DIS_s016
                                 + 0.975349 w_k1_FDX_s016
                                 + 1.78932 w_k1_IBM_s016
                                 + 1.03125 w_k1_INTC_s016
                                 + 1.34056 w_k1_KO_s016 + 1.68331 w_k1_MAT_s016
                                 + 0.927875 w_k1_MSFT_s016
                                 + 0.984812 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.31651 w_k1_ADM_s016
                                 + 1.31556 w_k1_AXP_s016
                                 + 0.983706 w_k1_DIS_s016
                                 + 1.0757 w_k1_FDX_s016
                                 + 0.906403 w_k1_IBM_s016
                                 + 1.15143 w_k1_INTC_s016 + 1.0246 w_k1_KO_s016
                                 + 1.0339 w_k1_MAT_s016
                                 + 1.00426 w_k1_MSFT_s016
                                 + 0.966328 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.995526 w_k1_ADM_s016
                                 + 1.04239 w_k1_AXP_s016
                                 + 1.23969 w_k1_DIS_s016
                                 + 0.901953 w_k1_FDX_s016
                                 + 1.15808 w_k1_IBM_s016
                                 + 1.05074 w_k1_INTC_s016
                                 + 1.14093 w_k1_KO_s016 + 1.0258 w_k1_MAT_s016
                                 + 0.998659 w_k1_MSFT_s016
                                 + 1.06872 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.08361 w_k1_ADM_s017
                                 + 1.29067 w_k1_AXP_s017
                                 + 1.17461 w_k1_DIS_s017
                                 + 1.20155 w_k1_FDX_s017
                                 + 0.987647 w_k1_IBM_s017
                                 + 1.20961 w_k1_INTC_s017
                                 + 0.872541 w_k1_KO_s017
                                 + 0.90465 w_k1_MAT_s017
                                 + 1.1497 w_k1_MSFT_s017 + 1.05366 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.913207 w_k1_ADM_s017
                                 + 1.35888 w_k1_AXP_s017
                                 + 1.62318 w_k1_DIS_s017
                                 + 1.30319 w_k1_FDX_s017 + 1.2248 w_k1_IBM_s017
                                 + 1.11748 w_k1_INTC_s017
                                 + 1.14444 w_k1_KO_s017 + 1.34649 w_k1_MAT_s017
                                 + 1.27825 w_k1_MSFT_s017
                                 + 1.03729 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.86295 w_k1_ADM_s017
                                 + 0.90277 w_k1_AXP_s017
                                 + 0.688548 w_k1_DIS_s017
                                 + 0.888207 w_k1_FDX_s017
                                 + 0.615985 w_k1_IBM_s017
                                 + 0.937877 w_k1_INTC_s017
                                 + 0.616742 w_k1_KO_s017
                                 + 0.723787 w_k1_MAT_s017
                                 + 0.866886 w_k1_MSFT_s017
                                 + 0.77637 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.912411 w_k1_ADM_s017
                                 + 0.875312 w_k1_AXP_s017
                                 + 1.08217 w_k1_DIS_s017
                                 + 1.12965 w_k1_FDX_s017
                                 + 0.758541 w_k1_IBM_s017
                                 + 0.950893 w_k1_INTC_s017
                                 + 0.936379 w_k1_KO_s017
                                 + 0.969282 w_k1_MAT_s017
                                 + 1.27902 w_k1_MSFT_s017
                                 + 0.993258 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.14215 w_k1_ADM_s017
                                 + 1.23033 w_k1_AXP_s017
                                 + 0.925962 w_k1_DIS_s017
                                 + 0.988365 w_k1_FDX_s017
                                 + 1.09204 w_k1_IBM_s017
                                 + 0.990655 w_k1_INTC_s017
                                 + 0.835948 w_k1_KO_s017
                                 + 1.18464 w_k1_MAT_s017
                                 + 0.959433 w_k1_MSFT_s017
                                 + 0.968966 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.913814 w_k1_ADM_s017
                                 + 0.932652 w_k1_AXP_s017
                                 + 0.914764 w_k1_DIS_s017
                                 + 0.853769 w_k1_FDX_s017
                                 + 1.45887 w_k1_IBM_s017
                                 + 0.970086 w_k1_INTC_s017
                                 + 1.11679 w_k1_KO_s017 + 1.36733 w_k1_MAT_s017
                                 + 1.00656 w_k1_MSFT_s017
                                 + 0.93434 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.810229 w_k1_ADM_s017
                                 + 0.638686 w_k1_AXP_s017
                                 + 0.572811 w_k1_DIS_s017
                                 + 0.707727 w_k1_FDX_s017
                                 + 0.791802 w_k1_IBM_s017
                                 + 0.787319 w_k1_INTC_s017
                                 + 0.655794 w_k1_KO_s017
                                 + 0.689714 w_k1_MAT_s017
                                 + 0.860368 w_k1_MSFT_s017
                                 + 0.726047 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.745202 w_k1_ADM_s017
                                 + 0.918307 w_k1_AXP_s017
                                 + 0.95549 w_k1_DIS_s017
                                 + 0.98126 w_k1_FDX_s017
                                 + 0.923731 w_k1_IBM_s017
                                 + 1.0151 w_k1_INTC_s017
                                 + 0.861979 w_k1_KO_s017
                                 + 0.920306 w_k1_MAT_s017
                                 + 1.03792 w_k1_MSFT_s017
                                 + 0.905867 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.909438 w_k1_ADM_s100
                                 + 1.00154 w_k1_AXP_s100
                                 + 1.09761 w_k1_DIS_s100
                                 + 0.983508 w_k1_FDX_s100
                                 + 1.32078 w_k1_IBM_s100
                                 + 0.888932 w_k1_INTC_s100
                                 + 1.06851 w_k1_KO_s100 + 1.62273 w_k1_MAT_s100
                                 + 1.0476 w_k1_MSFT_s100
                                 + 0.943112 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.13523 w_k1_ADM_s100
                                 + 1.08889 w_k1_AXP_s100
                                 + 1.52414 w_k1_DIS_s100
                                 + 1.10025 w_k1_FDX_s100
                                 + 1.09913 w_k1_IBM_s100
                                 + 1.20995 w_k1_INTC_s100
                                 + 1.30437 w_k1_KO_s100 + 1.17687 w_k1_MAT_s100
                                 + 1.21142 w_k1_MSFT_s100
                                 + 1.15363 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.774519 w_k1_ADM_s100
                                 + 0.631026 w_k1_AXP_s100
                                 + 0.689569 w_k1_DIS_s100
                                 + 0.799689 w_k1_FDX_s100
                                 + 0.695818 w_k1_IBM_s100
                                 + 0.754068 w_k1_INTC_s100
                                 + 0.705125 w_k1_KO_s100
                                 + 0.68654 w_k1_MAT_s100
                                 + 0.739192 w_k1_MSFT_s100
                                 + 0.910029 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.832877 w_k1_ADM_s100
                                 + 0.954391 w_k1_AXP_s100
                                 + 0.848718 w_k1_DIS_s100
                                 + 0.817697 w_k1_FDX_s100
                                 + 0.947968 w_k1_IBM_s100
                                 + 0.859481 w_k1_INTC_s100
                                 + 0.802947 w_k1_KO_s100
                                 + 1.13735 w_k1_MAT_s100
                                 + 0.791148 w_k1_MSFT_s100
                                 + 0.940115 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.830627 w_k1_ADM_s100
                                 + 0.833617 w_k1_AXP_s100
                                 + 1.08936 w_k1_DIS_s100
                                 + 0.966822 w_k1_FDX_s100
                                 + 0.704001 w_k1_IBM_s100
                                 + 1.01478 w_k1_INTC_s100
                                 + 0.86961 w_k1_KO_s100
                                 + 0.443205 w_k1_MAT_s100
                                 + 1.01791 w_k1_MSFT_s100
                                 + 0.894265 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.15631 w_k1_ADM_s100
                                 + 1.18853 w_k1_AXP_s100
                                 + 1.52582 w_k1_DIS_s100
                                 + 1.23094 w_k1_FDX_s100
                                 + 0.906262 w_k1_IBM_s100
                                 + 1.22372 w_k1_INTC_s100
                                 + 1.14558 w_k1_KO_s100 + 1.11371 w_k1_MAT_s100
                                 + 1.36583 w_k1_MSFT_s100
                                 + 1.16839 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.2736 w_k1_ADM_s100
                                 + 1.2661 w_k1_AXP_s100 + 1.43429 w_k1_DIS_s100
                                 + 1.07188 w_k1_FDX_s100
                                 + 1.93723 w_k1_IBM_s100
                                 + 1.04102 w_k1_INTC_s100 + 1.4821 w_k1_KO_s100
                                 + 2.45477 w_k1_MAT_s100
                                 + 1.21668 w_k1_MSFT_s100
                                 + 1.21263 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.8763 w_k1_ADM_s100
                                 + 1.02122 w_k1_AXP_s100
                                 + 0.883369 w_k1_DIS_s100
                                 + 1.04141 w_k1_FDX_s100
                                 + 0.620924 w_k1_IBM_s100
                                 + 0.923098 w_k1_INTC_s100
                                 + 0.793126 w_k1_KO_s100
                                 + 0.942646 w_k1_MAT_s100
                                 + 0.954207 w_k1_MSFT_s100
                                 + 0.883871 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.14177 w_k1_ADM_s101
                                 + 1.00962 w_k1_AXP_s101
                                 + 0.944184 w_k1_DIS_s101
                                 + 1.08886 w_k1_FDX_s101
                                 + 0.818994 w_k1_IBM_s101
                                 + 1.07144 w_k1_INTC_s101
                                 + 0.943101 w_k1_KO_s101
                                 + 0.93934 w_k1_MAT_s101
                                 + 0.974602 w_k1_MSFT_s101
                                 + 1.01232 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.946443 w_k1_ADM_s101
                                 + 0.762672 w_k1_AXP_s101
                                 + 0.772256 w_k1_DIS_s101
                                 + 0.841446 w_k1_FDX_s101
                                 + 0.901237 w_k1_IBM_s101
                                 + 0.919794 w_k1_INTC_s101
                                 + 0.776595 w_k1_KO_s101 + 1.1141 w_k1_MAT_s101
                                 + 0.934543 w_k1_MSFT_s101
                                 + 0.83762 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.808919 w_k1_ADM_s101
                                 + 0.954461 w_k1_AXP_s101
                                 + 0.954616 w_k1_DIS_s101
                                 + 0.946295 w_k1_FDX_s101
                                 + 1.07454 w_k1_IBM_s101
                                 + 1.18756 w_k1_INTC_s101
                                 + 0.869738 w_k1_KO_s101
                                 + 1.01079 w_k1_MAT_s101
                                 + 0.977895 w_k1_MSFT_s101
                                 + 0.870988 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.23376 w_k1_ADM_s101
                                 + 1.4216 w_k1_AXP_s101 + 1.48251 w_k1_DIS_s101
                                 + 1.29277 w_k1_FDX_s101
                                 + 1.15521 w_k1_IBM_s101
                                 + 1.36701 w_k1_INTC_s101
                                 + 1.06663 w_k1_KO_s101 + 1.36266 w_k1_MAT_s101
                                 + 1.10182 w_k1_MSFT_s101
                                 + 1.06375 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.16265 w_k1_ADM_s101
                                 + 0.836191 w_k1_AXP_s101
                                 + 0.70168 w_k1_DIS_s101
                                 + 0.893436 w_k1_FDX_s101
                                 + 0.670503 w_k1_IBM_s101
                                 + 0.795406 w_k1_INTC_s101
                                 + 0.649991 w_k1_KO_s101
                                 + 0.675418 w_k1_MAT_s101
                                 + 0.837656 w_k1_MSFT_s101
                                 + 0.86035 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.29524 w_k1_ADM_s101
                                 + 0.976916 w_k1_AXP_s101
                                 + 1.2177 w_k1_DIS_s101 + 1.25184 w_k1_FDX_s101
                                 + 0.791761 w_k1_IBM_s101
                                 + 1.22696 w_k1_INTC_s101
                                 + 0.94719 w_k1_KO_s101 + 0.85053 w_k1_MAT_s101
                                 + 1.3043 w_k1_MSFT_s101 + 0.95137 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.15014 w_k1_ADM_s101
                                 + 1.15672 w_k1_AXP_s101
                                 + 1.18303 w_k1_DIS_s101
                                 + 1.16922 w_k1_FDX_s101
                                 + 1.28884 w_k1_IBM_s101
                                 + 1.03588 w_k1_INTC_s101 + 1.0191 w_k1_KO_s101
                                 + 1.44239 w_k1_MAT_s101
                                 + 0.942956 w_k1_MSFT_s101
                                 + 1.13479 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.857821 w_k1_ADM_s101
                                 + 0.891751 w_k1_AXP_s101
                                 + 0.792577 w_k1_DIS_s101
                                 + 0.822953 w_k1_FDX_s101
                                 + 1.22394 w_k1_IBM_s101
                                 + 0.935644 w_k1_INTC_s101
                                 + 1.00186 w_k1_KO_s101 + 1.32456 w_k1_MAT_s101
                                 + 0.815141 w_k1_MSFT_s101
                                 + 1.01408 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.903491 w_k1_ADM_s102
                                 + 1.0372 w_k1_AXP_s102 + 1.16289 w_k1_DIS_s102
                                 + 1.04063 w_k1_FDX_s102
                                 + 0.681703 w_k1_IBM_s102
                                 + 1.20846 w_k1_INTC_s102
                                 + 0.88848 w_k1_KO_s102
                                 + 0.594429 w_k1_MAT_s102
                                 + 0.977488 w_k1_MSFT_s102
                                 + 0.926439 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.06521 w_k1_ADM_s102
                                 + 0.990369 w_k1_AXP_s102
                                 + 0.971484 w_k1_DIS_s102
                                 + 0.908744 w_k1_FDX_s102
                                 + 1.13051 w_k1_IBM_s102
                                 + 0.976726 w_k1_INTC_s102
                                 + 1.01661 w_k1_KO_s102 + 1.2431 w_k1_MAT_s102
                                 + 0.869637 w_k1_MSFT_s102
                                 + 1.01332 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.789126 w_k1_ADM_s102
                                 + 0.685328 w_k1_AXP_s102
                                 + 0.883331 w_k1_DIS_s102
                                 + 0.820252 w_k1_FDX_s102
                                 + 0.962177 w_k1_IBM_s102
                                 + 1.05931 w_k1_INTC_s102
                                 + 0.991018 w_k1_KO_s102
                                 + 0.777387 w_k1_MAT_s102
                                 + 0.85808 w_k1_MSFT_s102
                                 + 0.901189 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.735312 w_k1_ADM_s102
                                 + 0.759722 w_k1_AXP_s102
                                 + 0.688341 w_k1_DIS_s102
                                 + 0.842852 w_k1_FDX_s102
                                 + 0.69546 w_k1_IBM_s102
                                 + 0.963577 w_k1_INTC_s102
                                 + 0.700356 w_k1_KO_s102
                                 + 0.793418 w_k1_MAT_s102
                                 + 0.710299 w_k1_MSFT_s102
                                 + 0.791896 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.29838 w_k1_ADM_s102
                                 + 1.40752 w_k1_AXP_s102
                                 + 1.41813 w_k1_DIS_s102
                                 + 1.14387 w_k1_FDX_s102 + 1.6471 w_k1_IBM_s102
                                 + 1.09196 w_k1_INTC_s102
                                 + 1.34642 w_k1_KO_s102 + 1.70263 w_k1_MAT_s102
                                 + 0.900021 w_k1_MSFT_s102
                                 + 1.24612 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.982939 w_k1_ADM_s102
                                 + 1.33231 w_k1_AXP_s102
                                 + 1.56883 w_k1_DIS_s102
                                 + 1.25734 w_k1_FDX_s102
                                 + 1.00584 w_k1_IBM_s102
                                 + 1.29485 w_k1_INTC_s102
                                 + 1.25393 w_k1_KO_s102 + 1.1875 w_k1_MAT_s102
                                 + 1.12599 w_k1_MSFT_s102 + 1.1322 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.19668 w_k1_ADM_s102
                                 + 1.01474 w_k1_AXP_s102
                                 + 1.12907 w_k1_DIS_s102
                                 + 0.958483 w_k1_FDX_s102
                                 + 1.32592 w_k1_IBM_s102
                                 + 1.23951 w_k1_INTC_s102 + 1.2963 w_k1_KO_s102
                                 + 1.12122 w_k1_MAT_s102
                                 + 0.989226 w_k1_MSFT_s102
                                 + 1.1023 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.13207 w_k1_ADM_s102
                                 + 1.05591 w_k1_AXP_s102
                                 + 1.08844 w_k1_DIS_s102
                                 + 1.06404 w_k1_FDX_s102
                                 + 0.935982 w_k1_IBM_s102
                                 + 1.04888 w_k1_INTC_s102 + 1.0685 w_k1_KO_s102
                                 + 0.925901 w_k1_MAT_s102
                                 + 0.959244 w_k1_MSFT_s102
                                 + 0.987511 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.973514 w_k1_ADM_s103
                                 + 1.02141 w_k1_AXP_s103
                                 + 1.19666 w_k1_DIS_s103
                                 + 1.18943 w_k1_FDX_s103
                                 + 0.906759 w_k1_IBM_s103
                                 + 1.30385 w_k1_INTC_s103
                                 + 1.14554 w_k1_KO_s103 + 1.16907 w_k1_MAT_s103
                                 + 1.34392 w_k1_MSFT_s103
                                 + 1.22695 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.776211 w_k1_ADM_s103
                                 + 0.812635 w_k1_AXP_s103
                                 + 0.687222 w_k1_DIS_s103
                                 + 0.843057 w_k1_FDX_s103
                                 + 0.669208 w_k1_IBM_s103
                                 + 0.815624 w_k1_INTC_s103
                                 + 0.771104 w_k1_KO_s103
                                 + 0.582261 w_k1_MAT_s103
                                 + 0.79363 w_k1_MSFT_s103
                                 + 0.887876 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.925498 w_k1_ADM_s103
                                 + 1.00119 w_k1_AXP_s103
                                 + 0.913852 w_k1_DIS_s103
                                 + 0.967811 w_k1_FDX_s103
                                 + 0.969777 w_k1_IBM_s103
                                 + 1.25477 w_k1_INTC_s103
                                 + 1.02037 w_k1_KO_s103
                                 + 0.906714 w_k1_MAT_s103
                                 + 1.0295 w_k1_MSFT_s103
                                 + 0.965719 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.984583 w_k1_ADM_s103
                                 + 0.983676 w_k1_AXP_s103
                                 + 0.9647 w_k1_DIS_s103 + 1.16213 w_k1_FDX_s103
                                 + 0.634408 w_k1_IBM_s103
                                 + 1.0968 w_k1_INTC_s103 + 0.94321 w_k1_KO_s103
                                 + 0.862762 w_k1_MAT_s103
                                 + 1.11954 w_k1_MSFT_s103
                                 + 1.12866 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.777931 w_k1_ADM_s103
                                 + 0.717409 w_k1_AXP_s103
                                 + 0.760464 w_k1_DIS_s103
                                 + 0.739009 w_k1_FDX_s103
                                 + 1.1562 w_k1_IBM_s103
                                 + 0.932878 w_k1_INTC_s103
                                 + 0.95007 w_k1_KO_s103 + 1.10531 w_k1_MAT_s103
                                 + 0.91008 w_k1_MSFT_s103
                                 + 0.915343 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.659705 w_k1_ADM_s103
                                 + 0.491551 w_k1_AXP_s103
                                 + 0.797167 w_k1_DIS_s103
                                 + 0.729536 w_k1_FDX_s103
                                 + 0.659932 w_k1_IBM_s103
                                 + 0.80682 w_k1_INTC_s103
                                 + 0.808352 w_k1_KO_s103
                                 + 0.571415 w_k1_MAT_s103
                                 + 0.919568 w_k1_MSFT_s103
                                 + 0.940833 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.17039 w_k1_ADM_s103
                                 + 1.07202 w_k1_AXP_s103 + 1.1358 w_k1_DIS_s103
                                 + 1.07625 w_k1_FDX_s103
                                 + 1.61931 w_k1_IBM_s103
                                 + 1.14295 w_k1_INTC_s103 + 1.358 w_k1_KO_s103
                                 + 1.56711 w_k1_MAT_s103
                                 + 1.09517 w_k1_MSFT_s103
                                 + 1.05122 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.32803 w_k1_ADM_s103
                                 + 1.31324 w_k1_AXP_s103
                                 + 1.73631 w_k1_DIS_s103
                                 + 1.51094 w_k1_FDX_s103
                                 + 0.971126 w_k1_IBM_s103
                                 + 1.40898 w_k1_INTC_s103
                                 + 1.42207 w_k1_KO_s103
                                 + 0.885087 w_k1_MAT_s103
                                 + 1.59401 w_k1_MSFT_s103 + 1.2684 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.862147 w_k1_ADM_s104
                                 + 0.807165 w_k1_AXP_s104
                                 + 0.722806 w_k1_DIS_s104
                                 + 0.85921 w_k1_FDX_s104
                                 + 0.74412 w_k1_IBM_s104
                                 + 1.00567 w_k1_INTC_s104
                                 + 0.731761 w_k1_KO_s104
                                 + 0.719257 w_k1_MAT_s104
                                 + 0.979398 w_k1_MSFT_s104
                                 + 0.806986 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.19969 w_k1_ADM_s104
                                 + 1.43307 w_k1_AXP_s104
                                 + 1.61687 w_k1_DIS_s104 + 1.2761 w_k1_FDX_s104
                                 + 1.61325 w_k1_IBM_s104
                                 + 1.41051 w_k1_INTC_s104 + 1.2641 w_k1_KO_s104
                                 + 1.56657 w_k1_MAT_s104
                                 + 1.41747 w_k1_MSFT_s104
                                 + 1.15689 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.07579 w_k1_ADM_s104
                                 + 1.5037 w_k1_AXP_s104 + 1.27073 w_k1_DIS_s104
                                 + 1.26161 w_k1_FDX_s104
                                 + 1.14647 w_k1_IBM_s104
                                 + 1.0881 w_k1_INTC_s104
                                 + 0.973691 w_k1_KO_s104 + 1.4749 w_k1_MAT_s104
                                 + 1.17191 w_k1_MSFT_s104
                                 + 1.14429 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.04 w_k1_ADM_s104
                                 + 1.40971 w_k1_AXP_s104
                                 + 0.844164 w_k1_DIS_s104
                                 + 1.05984 w_k1_FDX_s104 + 1.259 w_k1_IBM_s104
                                 + 1.13015 w_k1_INTC_s104
                                 + 0.987668 w_k1_KO_s104
                                 + 1.41024 w_k1_MAT_s104
                                 + 1.05778 w_k1_MSFT_s104
                                 + 1.10231 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.47074 w_k1_ADM_s104
                                 + 1.3855 w_k1_AXP_s104 + 1.83238 w_k1_DIS_s104
                                 + 1.36327 w_k1_FDX_s104
                                 + 1.00019 w_k1_IBM_s104
                                 + 1.28575 w_k1_INTC_s104 + 1.1136 w_k1_KO_s104
                                 + 1.43459 w_k1_MAT_s104
                                 + 1.58172 w_k1_MSFT_s104
                                 + 1.06901 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.08787 w_k1_ADM_s104
                                 + 1.07784 w_k1_AXP_s104
                                 + 1.11577 w_k1_DIS_s104
                                 + 1.18671 w_k1_FDX_s104
                                 + 0.816449 w_k1_IBM_s104
                                 + 1.1278 w_k1_INTC_s104
                                 + 0.787821 w_k1_KO_s104
                                 + 0.739699 w_k1_MAT_s104
                                 + 1.15264 w_k1_MSFT_s104
                                 + 1.01632 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.649444 w_k1_ADM_s104
                                 + 0.944093 w_k1_AXP_s104
                                 + 0.915284 w_k1_DIS_s104
                                 + 0.86233 w_k1_FDX_s104 + 1.3758 w_k1_IBM_s104
                                 + 0.913327 w_k1_INTC_s104
                                 + 0.912339 w_k1_KO_s104
                                 + 1.42611 w_k1_MAT_s104
                                 + 0.96157 w_k1_MSFT_s104
                                 + 0.974765 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.10506 w_k1_ADM_s104
                                 + 0.934065 w_k1_AXP_s104
                                 + 0.941078 w_k1_DIS_s104
                                 + 0.887217 w_k1_FDX_s104
                                 + 1.11561 w_k1_IBM_s104
                                 + 1.11944 w_k1_INTC_s104
                                 + 0.916037 w_k1_KO_s104
                                 + 0.95665 w_k1_MAT_s104
                                 + 1.06213 w_k1_MSFT_s104
                                 + 0.973052 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.59345 w_k1_ADM_s105
                                 + 1.63716 w_k1_AXP_s105 + 1.4337 w_k1_DIS_s105
                                 + 1.28387 w_k1_FDX_s105
                                 + 1.26485 w_k1_IBM_s105
                                 + 1.11977 w_k1_INTC_s105
                                 + 1.24379 w_k1_KO_s105 + 1.22466 w_k1_MAT_s105
                                 + 1.20256 w_k1_MSFT_s105 + 1.15 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.06589 w_k1_ADM_s105
                                 + 0.723205 w_k1_AXP_s105
                                 + 0.732158 w_k1_DIS_s105
                                 + 0.859003 w_k1_FDX_s105
                                 + 0.862844 w_k1_IBM_s105
                                 + 0.919271 w_k1_INTC_s105
                                 + 0.932338 w_k1_KO_s105
                                 + 0.732547 w_k1_MAT_s105
                                 + 1.01547 w_k1_MSFT_s105
                                 + 0.884705 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.24139 w_k1_ADM_s105
                                 + 1.20394 w_k1_AXP_s105
                                 + 1.69342 w_k1_DIS_s105
                                 + 1.21481 w_k1_FDX_s105 + 1.0767 w_k1_IBM_s105
                                 + 1.18949 w_k1_INTC_s105 + 1.4359 w_k1_KO_s105
                                 + 0.935358 w_k1_MAT_s105
                                 + 1.33007 w_k1_MSFT_s105
                                 + 1.17403 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.28977 w_k1_ADM_s105
                                 + 1.09575 w_k1_AXP_s105
                                 + 0.844876 w_k1_DIS_s105
                                 + 0.909435 w_k1_FDX_s105
                                 + 0.756273 w_k1_IBM_s105
                                 + 0.912876 w_k1_INTC_s105
                                 + 0.826925 w_k1_KO_s105
                                 + 0.56074 w_k1_MAT_s105
                                 + 0.909971 w_k1_MSFT_s105
                                 + 0.876486 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.780631 w_k1_ADM_s105
                                 + 0.697673 w_k1_AXP_s105
                                 + 0.692029 w_k1_DIS_s105
                                 + 0.687043 w_k1_FDX_s105
                                 + 0.800879 w_k1_IBM_s105
                                 + 0.903351 w_k1_INTC_s105
                                 + 0.771272 w_k1_KO_s105
                                 + 0.482304 w_k1_MAT_s105
                                 + 0.794106 w_k1_MSFT_s105
                                 + 0.806146 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.25972 w_k1_ADM_s105
                                 + 0.997275 w_k1_AXP_s105
                                 + 1.08783 w_k1_DIS_s105
                                 + 1.07906 w_k1_FDX_s105
                                 + 0.819615 w_k1_IBM_s105
                                 + 1.02451 w_k1_INTC_s105
                                 + 1.03955 w_k1_KO_s105
                                 + 0.805287 w_k1_MAT_s105
                                 + 1.20329 w_k1_MSFT_s105
                                 + 0.976193 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.1471 w_k1_ADM_s105
                                 + 1.00639 w_k1_AXP_s105
                                 + 1.09655 w_k1_DIS_s105
                                 + 0.918463 w_k1_FDX_s105
                                 + 1.31422 w_k1_IBM_s105
                                 + 1.05992 w_k1_INTC_s105
                                 + 1.09996 w_k1_KO_s105
                                 + 0.859114 w_k1_MAT_s105
                                 + 1.0476 w_k1_MSFT_s105
                                 + 0.981705 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.951743 w_k1_ADM_s105
                                 + 0.95082 w_k1_AXP_s105
                                 + 0.732936 w_k1_DIS_s105
                                 + 0.869169 w_k1_FDX_s105
                                 + 1.06256 w_k1_IBM_s105
                                 + 1.03237 w_k1_INTC_s105
                                 + 0.977438 w_k1_KO_s105
                                 + 0.732017 w_k1_MAT_s105
                                 + 0.842763 w_k1_MSFT_s105
                                 + 1.00067 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.945406 w_k1_ADM_s106
                                 + 0.734916 w_k1_AXP_s106
                                 + 0.709261 w_k1_DIS_s106 + 0.831 w_k1_FDX_s106
                                 + 0.710403 w_k1_IBM_s106
                                 + 0.750251 w_k1_INTC_s106
                                 + 0.824389 w_k1_KO_s106
                                 + 0.78977 w_k1_MAT_s106
                                 + 0.782551 w_k1_MSFT_s106
                                 + 0.89533 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.2511 w_k1_ADM_s106
                                 + 1.45953 w_k1_AXP_s106
                                 + 1.10382 w_k1_DIS_s106
                                 + 1.21964 w_k1_FDX_s106
                                 + 1.00209 w_k1_IBM_s106
                                 + 1.00496 w_k1_INTC_s106
                                 + 0.886972 w_k1_KO_s106
                                 + 1.01773 w_k1_MAT_s106
                                 + 0.96261 w_k1_MSFT_s106
                                 + 1.03747 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.00854 w_k1_ADM_s106
                                 + 0.947261 w_k1_AXP_s106
                                 + 0.921929 w_k1_DIS_s106
                                 + 1.00349 w_k1_FDX_s106
                                 + 0.849487 w_k1_IBM_s106
                                 + 0.962443 w_k1_INTC_s106
                                 + 0.995915 w_k1_KO_s106
                                 + 0.799545 w_k1_MAT_s106
                                 + 0.890943 w_k1_MSFT_s106
                                 + 1.01771 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.62677 w_k1_ADM_s106
                                 + 1.65324 w_k1_AXP_s106
                                 + 1.86371 w_k1_DIS_s106
                                 + 1.40345 w_k1_FDX_s106
                                 + 1.52655 w_k1_IBM_s106
                                 + 1.22077 w_k1_INTC_s106
                                 + 1.62355 w_k1_KO_s106 + 1.43068 w_k1_MAT_s106
                                 + 1.36095 w_k1_MSFT_s106 + 1.4932 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.139 w_k1_ADM_s106
                                 + 1.13544 w_k1_AXP_s106
                                 + 1.05068 w_k1_DIS_s106
                                 + 0.967002 w_k1_FDX_s106
                                 + 1.27957 w_k1_IBM_s106
                                 + 1.00265 w_k1_INTC_s106
                                 + 1.15848 w_k1_KO_s106 + 1.19081 w_k1_MAT_s106
                                 + 0.92533 w_k1_MSFT_s106
                                 + 1.08435 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.36861 w_k1_ADM_s106
                                 + 0.940082 w_k1_AXP_s106
                                 + 1.17263 w_k1_DIS_s106
                                 + 1.27798 w_k1_FDX_s106
                                 + 0.697777 w_k1_IBM_s106
                                 + 1.09146 w_k1_INTC_s106
                                 + 0.955022 w_k1_KO_s106
                                 + 0.545515 w_k1_MAT_s106
                                 + 1.1666 w_k1_MSFT_s106 + 1.12599 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.34019 w_k1_ADM_s106
                                 + 1.08944 w_k1_AXP_s106
                                 + 1.04649 w_k1_DIS_s106
                                 + 0.984846 w_k1_FDX_s106
                                 + 1.75026 w_k1_IBM_s106
                                 + 0.780923 w_k1_INTC_s106
                                 + 1.21089 w_k1_KO_s106 + 1.37709 w_k1_MAT_s106
                                 + 0.85412 w_k1_MSFT_s106
                                 + 1.06125 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.02562 w_k1_ADM_s106
                                 + 0.840518 w_k1_AXP_s106
                                 + 1.20758 w_k1_DIS_s106
                                 + 1.07073 w_k1_FDX_s106
                                 + 1.16375 w_k1_IBM_s106
                                 + 1.10399 w_k1_INTC_s106
                                 + 1.14473 w_k1_KO_s106 + 1.12611 w_k1_MAT_s106
                                 + 1.0295 w_k1_MSFT_s106 + 1.06425 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.823561 w_k1_ADM_s107
                                 + 0.719173 w_k1_AXP_s107
                                 + 0.65362 w_k1_DIS_s107
                                 + 0.886645 w_k1_FDX_s107
                                 + 0.738375 w_k1_IBM_s107
                                 + 0.947983 w_k1_INTC_s107
                                 + 0.725861 w_k1_KO_s107
                                 + 0.731053 w_k1_MAT_s107
                                 + 0.865201 w_k1_MSFT_s107
                                 + 0.806788 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.04498 w_k1_ADM_s107
                                 + 1.19478 w_k1_AXP_s107
                                 + 1.11717 w_k1_DIS_s107
                                 + 1.23494 w_k1_FDX_s107
                                 + 0.785703 w_k1_IBM_s107
                                 + 1.11206 w_k1_INTC_s107
                                 + 0.885384 w_k1_KO_s107
                                 + 1.00671 w_k1_MAT_s107
                                 + 1.16151 w_k1_MSFT_s107
                                 + 1.02904 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.812222 w_k1_ADM_s107
                                 + 0.771037 w_k1_AXP_s107
                                 + 0.842066 w_k1_DIS_s107
                                 + 0.862084 w_k1_FDX_s107
                                 + 1.22874 w_k1_IBM_s107
                                 + 0.825568 w_k1_INTC_s107
                                 + 0.966105 w_k1_KO_s107
                                 + 1.14971 w_k1_MAT_s107
                                 + 0.908945 w_k1_MSFT_s107
                                 + 1.00374 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.813771 w_k1_ADM_s107
                                 + 0.923086 w_k1_AXP_s107
                                 + 1.10995 w_k1_DIS_s107
                                 + 0.896885 w_k1_FDX_s107
                                 + 1.79849 w_k1_IBM_s107
                                 + 0.802215 w_k1_INTC_s107
                                 + 1.29678 w_k1_KO_s107 + 1.74127 w_k1_MAT_s107
                                 + 0.924707 w_k1_MSFT_s107
                                 + 0.974105 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.938737 w_k1_ADM_s107
                                 + 0.807569 w_k1_AXP_s107
                                 + 1.30487 w_k1_DIS_s107
                                 + 1.11227 w_k1_FDX_s107
                                 + 0.889943 w_k1_IBM_s107
                                 + 0.990275 w_k1_INTC_s107
                                 + 0.949262 w_k1_KO_s107
                                 + 0.622996 w_k1_MAT_s107
                                 + 1.02754 w_k1_MSFT_s107
                                 + 0.918064 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.12652 w_k1_ADM_s107
                                 + 1.21719 w_k1_AXP_s107
                                 + 1.42813 w_k1_DIS_s107
                                 + 1.16211 w_k1_FDX_s107
                                 + 1.34833 w_k1_IBM_s107
                                 + 1.30369 w_k1_INTC_s107
                                 + 1.44345 w_k1_KO_s107 + 1.49384 w_k1_MAT_s107
                                 + 1.13645 w_k1_MSFT_s107
                                 + 1.21202 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.992958 w_k1_ADM_s107
                                 + 0.860944 w_k1_AXP_s107
                                 + 0.750441 w_k1_DIS_s107
                                 + 0.84219 w_k1_FDX_s107
                                 + 1.06916 w_k1_IBM_s107
                                 + 0.862277 w_k1_INTC_s107
                                 + 0.951432 w_k1_KO_s107
                                 + 1.20146 w_k1_MAT_s107
                                 + 0.899718 w_k1_MSFT_s107
                                 + 1.00719 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.713399 w_k1_ADM_s107
                                 + 0.845313 w_k1_AXP_s107
                                 + 0.857384 w_k1_DIS_s107
                                 + 0.92979 w_k1_FDX_s107
                                 + 0.955122 w_k1_IBM_s107
                                 + 0.864621 w_k1_INTC_s107
                                 + 0.811439 w_k1_KO_s107
                                 + 0.886727 w_k1_MAT_s107
                                 + 0.867024 w_k1_MSFT_s107
                                 + 0.856778 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.898094 w_k1_ADM_s110
                                 + 0.785256 w_k1_AXP_s110
                                 + 0.707408 w_k1_DIS_s110
                                 + 0.835767 w_k1_FDX_s110
                                 + 1.20857 w_k1_IBM_s110
                                 + 0.835504 w_k1_INTC_s110
                                 + 0.833649 w_k1_KO_s110
                                 + 1.39563 w_k1_MAT_s110
                                 + 0.863217 w_k1_MSFT_s110
                                 + 0.781393 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.775198 w_k1_ADM_s110
                                 + 0.54641 w_k1_AXP_s110
                                 + 0.726979 w_k1_DIS_s110
                                 + 0.730901 w_k1_FDX_s110
                                 + 0.755139 w_k1_IBM_s110
                                 + 0.701879 w_k1_INTC_s110
                                 + 0.735277 w_k1_KO_s110
                                 + 0.657869 w_k1_MAT_s110
                                 + 0.847118 w_k1_MSFT_s110
                                 + 0.792925 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.939359 w_k1_ADM_s110
                                 + 0.872855 w_k1_AXP_s110
                                 + 0.767914 w_k1_DIS_s110
                                 + 0.922112 w_k1_FDX_s110
                                 + 0.902853 w_k1_IBM_s110
                                 + 0.923539 w_k1_INTC_s110
                                 + 0.779719 w_k1_KO_s110
                                 + 0.931992 w_k1_MAT_s110
                                 + 0.926962 w_k1_MSFT_s110
                                 + 0.934873 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.38244 w_k1_ADM_s110
                                 + 0.938347 w_k1_AXP_s110
                                 + 1.88721 w_k1_DIS_s110
                                 + 1.32802 w_k1_FDX_s110 + 1.3351 w_k1_IBM_s110
                                 + 1.16541 w_k1_INTC_s110
                                 + 1.38958 w_k1_KO_s110
                                 + 0.845774 w_k1_MAT_s110
                                 + 1.3197 w_k1_MSFT_s110 + 1.01933 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.843141 w_k1_ADM_s110
                                 + 1.06037 w_k1_AXP_s110
                                 + 0.668319 w_k1_DIS_s110
                                 + 0.933524 w_k1_FDX_s110
                                 + 0.801189 w_k1_IBM_s110
                                 + 0.831337 w_k1_INTC_s110
                                 + 0.685955 w_k1_KO_s110
                                 + 1.08428 w_k1_MAT_s110
                                 + 0.833478 w_k1_MSFT_s110
                                 + 0.800931 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.27273 w_k1_ADM_s110
                                 + 1.57382 w_k1_AXP_s110
                                 + 1.45121 w_k1_DIS_s110
                                 + 1.36886 w_k1_FDX_s110
                                 + 1.16988 w_k1_IBM_s110
                                 + 1.34186 w_k1_INTC_s110
                                 + 1.21665 w_k1_KO_s110 + 1.42773 w_k1_MAT_s110
                                 + 1.09979 w_k1_MSFT_s110
                                 + 1.05273 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.12085 w_k1_ADM_s110
                                 + 1.01992 w_k1_AXP_s110
                                 + 1.19761 w_k1_DIS_s110 + 1.0565 w_k1_FDX_s110
                                 + 1.2793 w_k1_IBM_s110
                                 + 0.910044 w_k1_INTC_s110
                                 + 1.02829 w_k1_KO_s110 + 1.09143 w_k1_MAT_s110
                                 + 1.02174 w_k1_MSFT_s110
                                 + 0.964071 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.921944 w_k1_ADM_s110
                                 + 1.16928 w_k1_AXP_s110
                                 + 1.06808 w_k1_DIS_s110 + 1.1787 w_k1_FDX_s110
                                 + 0.761882 w_k1_IBM_s110
                                 + 1.02051 w_k1_INTC_s110
                                 + 0.865218 w_k1_KO_s110
                                 + 0.887706 w_k1_MAT_s110
                                 + 1.07867 w_k1_MSFT_s110
                                 + 0.815889 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.29587 w_k1_ADM_s111
                                 + 0.953328 w_k1_AXP_s111
                                 + 1.01672 w_k1_DIS_s111
                                 + 1.08128 w_k1_FDX_s111
                                 + 0.839821 w_k1_IBM_s111
                                 + 0.923367 w_k1_INTC_s111
                                 + 0.882203 w_k1_KO_s111
                                 + 0.79133 w_k1_MAT_s111
                                 + 0.926119 w_k1_MSFT_s111
                                 + 1.00772 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.791828 w_k1_ADM_s111
                                 + 0.702368 w_k1_AXP_s111
                                 + 0.624842 w_k1_DIS_s111
                                 + 0.798853 w_k1_FDX_s111
                                 + 0.706393 w_k1_IBM_s111
                                 + 0.945331 w_k1_INTC_s111
                                 + 0.741534 w_k1_KO_s111
                                 + 0.650443 w_k1_MAT_s111
                                 + 0.831413 w_k1_MSFT_s111
                                 + 0.878357 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.794389 w_k1_ADM_s111
                                 + 0.786373 w_k1_AXP_s111
                                 + 0.756771 w_k1_DIS_s111
                                 + 0.832475 w_k1_FDX_s111
                                 + 1.00277 w_k1_IBM_s111
                                 + 0.891456 w_k1_INTC_s111
                                 + 0.828362 w_k1_KO_s111
                                 + 0.935287 w_k1_MAT_s111
                                 + 0.857482 w_k1_MSFT_s111
                                 + 0.925636 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.962297 w_k1_ADM_s111
                                 + 0.695879 w_k1_AXP_s111
                                 + 1.01246 w_k1_DIS_s111
                                 + 0.805898 w_k1_FDX_s111
                                 + 1.15132 w_k1_IBM_s111
                                 + 0.889666 w_k1_INTC_s111
                                 + 0.973315 w_k1_KO_s111
                                 + 0.805289 w_k1_MAT_s111
                                 + 0.944478 w_k1_MSFT_s111
                                 + 0.889176 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.0448 w_k1_ADM_s111
                                 + 1.03333 w_k1_AXP_s111
                                 + 0.926756 w_k1_DIS_s111
                                 + 0.953111 w_k1_FDX_s111
                                 + 1.27823 w_k1_IBM_s111
                                 + 1.03828 w_k1_INTC_s111
                                 + 1.04788 w_k1_KO_s111 + 1.2194 w_k1_MAT_s111
                                 + 0.913396 w_k1_MSFT_s111
                                 + 1.06529 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.36169 w_k1_ADM_s111
                                 + 1.32803 w_k1_AXP_s111 + 1.5349 w_k1_DIS_s111
                                 + 1.34118 w_k1_FDX_s111
                                 + 0.862268 w_k1_IBM_s111
                                 + 1.30165 w_k1_INTC_s111
                                 + 1.11728 w_k1_KO_s111 + 1.07876 w_k1_MAT_s111
                                 + 1.3424 w_k1_MSFT_s111 + 1.24619 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.948161 w_k1_ADM_s111
                                 + 0.821859 w_k1_AXP_s111
                                 + 0.954031 w_k1_DIS_s111
                                 + 0.940043 w_k1_FDX_s111
                                 + 0.808318 w_k1_IBM_s111
                                 + 0.911232 w_k1_INTC_s111
                                 + 0.804034 w_k1_KO_s111
                                 + 0.852583 w_k1_MAT_s111
                                 + 0.981739 w_k1_MSFT_s111
                                 + 0.95407 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.26944 w_k1_ADM_s111
                                 + 1.4482 w_k1_AXP_s111 + 1.42019 w_k1_DIS_s111
                                 + 1.25884 w_k1_FDX_s111
                                 + 1.43088 w_k1_IBM_s111
                                 + 1.06041 w_k1_INTC_s111
                                 + 1.12587 w_k1_KO_s111 + 1.47682 w_k1_MAT_s111
                                 + 1.0271 w_k1_MSFT_s111 + 1.17426 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.76862 w_k1_ADM_s112
                                 + 0.714115 w_k1_AXP_s112
                                 + 0.753172 w_k1_DIS_s112
                                 + 0.862613 w_k1_FDX_s112
                                 + 0.808566 w_k1_IBM_s112
                                 + 1.08476 w_k1_INTC_s112
                                 + 0.886141 w_k1_KO_s112
                                 + 0.602519 w_k1_MAT_s112
                                 + 0.923311 w_k1_MSFT_s112
                                 + 0.787517 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.911665 w_k1_ADM_s112
                                 + 0.862275 w_k1_AXP_s112
                                 + 0.864778 w_k1_DIS_s112
                                 + 0.82248 w_k1_FDX_s112
                                 + 1.13109 w_k1_IBM_s112
                                 + 1.07927 w_k1_INTC_s112
                                 + 1.06799 w_k1_KO_s112
                                 + 0.770435 w_k1_MAT_s112
                                 + 0.871057 w_k1_MSFT_s112
                                 + 0.826336 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.928627 w_k1_ADM_s112
                                 + 0.564233 w_k1_AXP_s112
                                 + 0.558078 w_k1_DIS_s112
                                 + 0.708877 w_k1_FDX_s112
                                 + 0.587022 w_k1_IBM_s112
                                 + 0.877233 w_k1_INTC_s112
                                 + 0.645189 w_k1_KO_s112
                                 + 0.398363 w_k1_MAT_s112
                                 + 0.706659 w_k1_MSFT_s112
                                 + 0.777211 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.06462 w_k1_ADM_s112
                                 + 0.951494 w_k1_AXP_s112
                                 + 0.974584 w_k1_DIS_s112
                                 + 1.10066 w_k1_FDX_s112
                                 + 0.721008 w_k1_IBM_s112
                                 + 1.09038 w_k1_INTC_s112
                                 + 1.03829 w_k1_KO_s112 + 0.79658 w_k1_MAT_s112
                                 + 1.08986 w_k1_MSFT_s112
                                 + 0.980243 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.30163 w_k1_ADM_s112
                                 + 1.12186 w_k1_AXP_s112
                                 + 1.68626 w_k1_DIS_s112
                                 + 1.35916 w_k1_FDX_s112
                                 + 1.45511 w_k1_IBM_s112
                                 + 1.36342 w_k1_INTC_s112
                                 + 1.45612 w_k1_KO_s112 + 1.10838 w_k1_MAT_s112
                                 + 1.31489 w_k1_MSFT_s112
                                 + 1.06938 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.09898 w_k1_ADM_s112
                                 + 1.2956 w_k1_AXP_s112 + 1.1712 w_k1_DIS_s112
                                 + 0.97467 w_k1_FDX_s112
                                 + 1.46602 w_k1_IBM_s112
                                 + 1.29891 w_k1_INTC_s112
                                 + 1.31328 w_k1_KO_s112 + 1.2061 w_k1_MAT_s112
                                 + 1.09276 w_k1_MSFT_s112
                                 + 0.988713 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.29678 w_k1_ADM_s112
                                 + 1.38864 w_k1_AXP_s112
                                 + 1.21669 w_k1_DIS_s112
                                 + 1.19749 w_k1_FDX_s112
                                 + 0.993203 w_k1_IBM_s112
                                 + 1.01551 w_k1_INTC_s112
                                 + 0.961875 w_k1_KO_s112
                                 + 1.08495 w_k1_MAT_s112
                                 + 1.05212 w_k1_MSFT_s112
                                 + 1.06132 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.09041 w_k1_ADM_s112
                                 + 0.879708 w_k1_AXP_s112
                                 + 1.29103 w_k1_DIS_s112
                                 + 1.08679 w_k1_FDX_s112
                                 + 0.909798 w_k1_IBM_s112
                                 + 1.04726 w_k1_INTC_s112
                                 + 1.09701 w_k1_KO_s112
                                 + 0.677079 w_k1_MAT_s112
                                 + 0.99597 w_k1_MSFT_s112
                                 + 1.04863 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.706429 w_k1_ADM_s113
                                 + 0.791789 w_k1_AXP_s113
                                 + 1.16344 w_k1_DIS_s113
                                 + 0.930567 w_k1_FDX_s113
                                 + 1.12704 w_k1_IBM_s113
                                 + 1.07105 w_k1_INTC_s113
                                 + 1.27258 w_k1_KO_s113
                                 + 0.883355 w_k1_MAT_s113
                                 + 1.1112 w_k1_MSFT_s113 + 1.0318 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.96762 w_k1_ADM_s113
                                 + 1.26997 w_k1_AXP_s113
                                 + 1.46203 w_k1_DIS_s113
                                 + 1.06493 w_k1_FDX_s113
                                 + 1.43991 w_k1_IBM_s113
                                 + 1.14998 w_k1_INTC_s113
                                 + 1.52373 w_k1_KO_s113 + 1.49705 w_k1_MAT_s113
                                 + 1.20325 w_k1_MSFT_s113
                                 + 1.28522 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.909123 w_k1_ADM_s113
                                 + 0.768357 w_k1_AXP_s113
                                 + 0.793855 w_k1_DIS_s113
                                 + 0.805256 w_k1_FDX_s113
                                 + 1.07992 w_k1_IBM_s113
                                 + 0.816219 w_k1_INTC_s113
                                 + 0.882548 w_k1_KO_s113
                                 + 0.981558 w_k1_MAT_s113
                                 + 0.873226 w_k1_MSFT_s113
                                 + 0.89894 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.966229 w_k1_ADM_s113
                                 + 0.791551 w_k1_AXP_s113
                                 + 0.627828 w_k1_DIS_s113
                                 + 0.731153 w_k1_FDX_s113
                                 + 0.895355 w_k1_IBM_s113
                                 + 0.647432 w_k1_INTC_s113
                                 + 0.827163 w_k1_KO_s113
                                 + 0.986897 w_k1_MAT_s113
                                 + 0.698592 w_k1_MSFT_s113
                                 + 0.963091 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.941983 w_k1_ADM_s113
                                 + 0.992938 w_k1_AXP_s113
                                 + 0.995386 w_k1_DIS_s113
                                 + 0.904931 w_k1_FDX_s113
                                 + 1.00789 w_k1_IBM_s113
                                 + 0.840389 w_k1_INTC_s113
                                 + 1.19101 w_k1_KO_s113 + 1.32807 w_k1_MAT_s113
                                 + 0.973878 w_k1_MSFT_s113
                                 + 1.08551 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.25052 w_k1_ADM_s113
                                 + 1.01816 w_k1_AXP_s113
                                 + 1.35194 w_k1_DIS_s113 + 1.1416 w_k1_FDX_s113
                                 + 0.903115 w_k1_IBM_s113
                                 + 0.918473 w_k1_INTC_s113
                                 + 1.18956 w_k1_KO_s113 + 1.05 w_k1_MAT_s113
                                 + 1.2387 w_k1_MSFT_s113 + 1.11225 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.15616 w_k1_ADM_s113
                                 + 1.40345 w_k1_AXP_s113
                                 + 1.20959 w_k1_DIS_s113
                                 + 1.28458 w_k1_FDX_s113
                                 + 0.972204 w_k1_IBM_s113
                                 + 1.01678 w_k1_INTC_s113
                                 + 1.14587 w_k1_KO_s113 + 1.62952 w_k1_MAT_s113
                                 + 1.31229 w_k1_MSFT_s113
                                 + 1.19025 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.830012 w_k1_ADM_s113
                                 + 0.70036 w_k1_AXP_s113
                                 + 0.78996 w_k1_DIS_s113
                                 + 0.776318 w_k1_FDX_s113
                                 + 0.672619 w_k1_IBM_s113
                                 + 0.830477 w_k1_INTC_s113
                                 + 0.80288 w_k1_KO_s113
                                 + 0.630839 w_k1_MAT_s113
                                 + 0.933483 w_k1_MSFT_s113
                                 + 0.88132 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.40042 w_k1_ADM_s114
                                 + 1.9526 w_k1_AXP_s114 + 1.48762 w_k1_DIS_s114
                                 + 1.29835 w_k1_FDX_s114
                                 + 1.38686 w_k1_IBM_s114
                                 + 1.42607 w_k1_INTC_s114
                                 + 1.28894 w_k1_KO_s114 + 1.1881 w_k1_MAT_s114
                                 + 0.971848 w_k1_MSFT_s114
                                 + 1.06521 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.787699 w_k1_ADM_s114
                                 + 0.947602 w_k1_AXP_s114
                                 + 1.31579 w_k1_DIS_s114
                                 + 0.976661 w_k1_FDX_s114
                                 + 1.35802 w_k1_IBM_s114
                                 + 1.17155 w_k1_INTC_s114
                                 + 1.38948 w_k1_KO_s114
                                 + 0.939602 w_k1_MAT_s114
                                 + 0.930809 w_k1_MSFT_s114
                                 + 0.977871 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.14708 w_k1_ADM_s114
                                 + 1.17207 w_k1_AXP_s114 + 1.4393 w_k1_DIS_s114
                                 + 1.26428 w_k1_FDX_s114
                                 + 0.853325 w_k1_IBM_s114
                                 + 1.5569 w_k1_INTC_s114 + 1.17873 w_k1_KO_s114
                                 + 0.567391 w_k1_MAT_s114
                                 + 1.15456 w_k1_MSFT_s114
                                 + 1.01424 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.894791 w_k1_ADM_s114
                                 + 1.21571 w_k1_AXP_s114
                                 + 1.02179 w_k1_DIS_s114
                                 + 0.961959 w_k1_FDX_s114
                                 + 1.81865 w_k1_IBM_s114
                                 + 1.12175 w_k1_INTC_s114
                                 + 1.38426 w_k1_KO_s114 + 1.83449 w_k1_MAT_s114
                                 + 0.835736 w_k1_MSFT_s114
                                 + 1.00953 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.773447 w_k1_ADM_s114
                                 + 0.94582 w_k1_AXP_s114
                                 + 0.914058 w_k1_DIS_s114
                                 + 0.93998 w_k1_FDX_s114 + 1.05 w_k1_IBM_s114
                                 + 1.19887 w_k1_INTC_s114
                                 + 0.934377 w_k1_KO_s114
                                 + 0.864041 w_k1_MAT_s114
                                 + 0.968608 w_k1_MSFT_s114
                                 + 0.918218 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.07309 w_k1_ADM_s114
                                 + 1.16439 w_k1_AXP_s114
                                 + 0.933158 w_k1_DIS_s114
                                 + 1.09996 w_k1_FDX_s114
                                 + 0.663705 w_k1_IBM_s114
                                 + 1.0291 w_k1_INTC_s114
                                 + 0.847715 w_k1_KO_s114
                                 + 0.857961 w_k1_MAT_s114
                                 + 0.939215 w_k1_MSFT_s114
                                 + 1.00256 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.710502 w_k1_ADM_s114
                                 + 0.682476 w_k1_AXP_s114
                                 + 0.749241 w_k1_DIS_s114
                                 + 0.827316 w_k1_FDX_s114
                                 + 0.842428 w_k1_IBM_s114
                                 + 1.01837 w_k1_INTC_s114
                                 + 0.864793 w_k1_KO_s114
                                 + 0.691474 w_k1_MAT_s114
                                 + 0.839793 w_k1_MSFT_s114
                                 + 0.803891 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.08894 w_k1_ADM_s114
                                 + 1.05382 w_k1_AXP_s114 + 1.3089 w_k1_DIS_s114
                                 + 1.1616 w_k1_FDX_s114 + 1.19932 w_k1_IBM_s114
                                 + 1.21531 w_k1_INTC_s114
                                 + 1.41881 w_k1_KO_s114
                                 + 0.956749 w_k1_MAT_s114
                                 + 0.991645 w_k1_MSFT_s114
                                 + 1.01771 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.948582 w_k1_ADM_s115
                                 + 0.710159 w_k1_AXP_s115
                                 + 0.761926 w_k1_DIS_s115
                                 + 0.857588 w_k1_FDX_s115
                                 + 0.70014 w_k1_IBM_s115
                                 + 0.9163 w_k1_INTC_s115
                                 + 0.849611 w_k1_KO_s115
                                 + 0.578241 w_k1_MAT_s115
                                 + 0.861124 w_k1_MSFT_s115
                                 + 0.877252 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.18476 w_k1_ADM_s115
                                 + 1.30982 w_k1_AXP_s115
                                 + 1.45601 w_k1_DIS_s115
                                 + 1.18576 w_k1_FDX_s115
                                 + 1.16593 w_k1_IBM_s115
                                 + 1.26424 w_k1_INTC_s115
                                 + 1.21491 w_k1_KO_s115 + 1.16676 w_k1_MAT_s115
                                 + 1.27481 w_k1_MSFT_s115
                                 + 1.14211 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.23636 w_k1_ADM_s115
                                 + 1.53029 w_k1_AXP_s115 + 1.7278 w_k1_DIS_s115
                                 + 1.35373 w_k1_FDX_s115
                                 + 1.38273 w_k1_IBM_s115
                                 + 1.54395 w_k1_INTC_s115
                                 + 1.65667 w_k1_KO_s115 + 1.32784 w_k1_MAT_s115
                                 + 1.53145 w_k1_MSFT_s115
                                 + 1.21609 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.09437 w_k1_ADM_s115
                                 + 1.49017 w_k1_AXP_s115
                                 + 0.889417 w_k1_DIS_s115
                                 + 0.991116 w_k1_FDX_s115
                                 + 1.42159 w_k1_IBM_s115
                                 + 0.945767 w_k1_INTC_s115
                                 + 1.0851 w_k1_KO_s115 + 1.53403 w_k1_MAT_s115
                                 + 1.05735 w_k1_MSFT_s115
                                 + 1.09841 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.09171 w_k1_ADM_s115
                                 + 0.784098 w_k1_AXP_s115
                                 + 0.680806 w_k1_DIS_s115
                                 + 0.829719 w_k1_FDX_s115
                                 + 1.02192 w_k1_IBM_s115
                                 + 0.713983 w_k1_INTC_s115
                                 + 0.898382 w_k1_KO_s115
                                 + 0.92924 w_k1_MAT_s115
                                 + 0.880912 w_k1_MSFT_s115
                                 + 0.944989 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.08549 w_k1_ADM_s115
                                 + 0.926529 w_k1_AXP_s115
                                 + 1.07027 w_k1_DIS_s115
                                 + 0.99151 w_k1_FDX_s115
                                 + 1.02397 w_k1_IBM_s115
                                 + 0.832931 w_k1_INTC_s115
                                 + 1.02764 w_k1_KO_s115 + 1.13784 w_k1_MAT_s115
                                 + 1.0983 w_k1_MSFT_s115
                                 + 0.964359 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.373 w_k1_ADM_s115
                                 + 1.04496 w_k1_AXP_s115
                                 + 1.10926 w_k1_DIS_s115
                                 + 1.19422 w_k1_FDX_s115
                                 + 0.897557 w_k1_IBM_s115
                                 + 0.956706 w_k1_INTC_s115
                                 + 1.10017 w_k1_KO_s115
                                 + 0.883392 w_k1_MAT_s115
                                 + 1.1262 w_k1_MSFT_s115 + 1.05475 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.20776 w_k1_ADM_s115
                                 + 1.08222 w_k1_AXP_s115
                                 + 1.13791 w_k1_DIS_s115
                                 + 1.14351 w_k1_FDX_s115
                                 + 1.32495 w_k1_IBM_s115
                                 + 1.02653 w_k1_INTC_s115
                                 + 1.24876 w_k1_KO_s115
                                 + 0.943503 w_k1_MAT_s115
                                 + 1.12489 w_k1_MSFT_s115
                                 + 1.02268 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.22696 w_k1_ADM_s116
                                 + 1.45778 w_k1_AXP_s116
                                 + 1.36688 w_k1_DIS_s116
                                 + 1.32592 w_k1_FDX_s116
                                 + 1.08426 w_k1_IBM_s116
                                 + 1.57545 w_k1_INTC_s116
                                 + 1.34746 w_k1_KO_s116
                                 + 0.914587 w_k1_MAT_s116
                                 + 1.23736 w_k1_MSFT_s116
                                 + 1.30795 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.28881 w_k1_ADM_s116
                                 + 1.47072 w_k1_AXP_s116
                                 + 2.00524 w_k1_DIS_s116
                                 + 1.47234 w_k1_FDX_s116
                                 + 1.40132 w_k1_IBM_s116
                                 + 1.28286 w_k1_INTC_s116
                                 + 1.57419 w_k1_KO_s116 + 1.84199 w_k1_MAT_s116
                                 + 1.46196 w_k1_MSFT_s116
                                 + 1.39752 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.998903 w_k1_ADM_s116
                                 + 0.769273 w_k1_AXP_s116
                                 + 1.02137 w_k1_DIS_s116
                                 + 0.99609 w_k1_FDX_s116
                                 + 0.899227 w_k1_IBM_s116
                                 + 1.14161 w_k1_INTC_s116
                                 + 0.912488 w_k1_KO_s116
                                 + 0.941092 w_k1_MAT_s116
                                 + 0.944327 w_k1_MSFT_s116
                                 + 0.994096 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.731803 w_k1_ADM_s116
                                 + 0.606162 w_k1_AXP_s116
                                 + 0.620746 w_k1_DIS_s116
                                 + 0.684701 w_k1_FDX_s116
                                 + 0.650101 w_k1_IBM_s116
                                 + 0.924401 w_k1_INTC_s116
                                 + 0.708462 w_k1_KO_s116
                                 + 0.573127 w_k1_MAT_s116
                                 + 0.802504 w_k1_MSFT_s116
                                 + 0.75582 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.1855 w_k1_ADM_s116
                                 + 0.998433 w_k1_AXP_s116
                                 + 1.4276 w_k1_DIS_s116 + 1.03243 w_k1_FDX_s116
                                 + 1.3513 w_k1_IBM_s116
                                 + 1.18657 w_k1_INTC_s116
                                 + 1.24589 w_k1_KO_s116 + 1.02264 w_k1_MAT_s116
                                 + 1.1507 w_k1_MSFT_s116 + 1.23422 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.889266 w_k1_ADM_s116
                                 + 0.736234 w_k1_AXP_s116
                                 + 0.589129 w_k1_DIS_s116
                                 + 0.812451 w_k1_FDX_s116
                                 + 0.877143 w_k1_IBM_s116
                                 + 1.06543 w_k1_INTC_s116
                                 + 0.771159 w_k1_KO_s116
                                 + 0.707089 w_k1_MAT_s116
                                 + 0.848526 w_k1_MSFT_s116
                                 + 0.827739 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.845665 w_k1_ADM_s116
                                 + 0.71013 w_k1_AXP_s116
                                 + 0.860574 w_k1_DIS_s116
                                 + 1.04873 w_k1_FDX_s116
                                 + 0.583973 w_k1_IBM_s116
                                 + 1.26955 w_k1_INTC_s116
                                 + 0.797763 w_k1_KO_s116
                                 + 0.501564 w_k1_MAT_s116
                                 + 1.008 w_k1_MSFT_s116 + 0.877719 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.983332 w_k1_ADM_s116
                                 + 1.05609 w_k1_AXP_s116
                                 + 1.08045 w_k1_DIS_s116 + 1.0581 w_k1_FDX_s116
                                 + 1.2265 w_k1_IBM_s116
                                 + 1.11909 w_k1_INTC_s116
                                 + 1.08701 w_k1_KO_s116 + 1.12921 w_k1_MAT_s116
                                 + 1.03741 w_k1_MSFT_s116
                                 + 1.12835 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.899594 w_k1_ADM_s117
                                 + 0.745012 w_k1_AXP_s117
                                 + 0.958964 w_k1_DIS_s117
                                 + 0.953947 w_k1_FDX_s117
                                 + 0.948042 w_k1_IBM_s117
                                 + 1.29214 w_k1_INTC_s117
                                 + 0.962652 w_k1_KO_s117
                                 + 0.66807 w_k1_MAT_s117
                                 + 1.0578 w_k1_MSFT_s117 + 1.01958 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.19293 w_k1_ADM_s117
                                 + 1.0671 w_k1_AXP_s117 + 1.40379 w_k1_DIS_s117
                                 + 1.26065 w_k1_FDX_s117
                                 + 0.833129 w_k1_IBM_s117
                                 + 1.46748 w_k1_INTC_s117
                                 + 1.22928 w_k1_KO_s117
                                 + 0.701848 w_k1_MAT_s117
                                 + 1.26839 w_k1_MSFT_s117
                                 + 1.04903 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.755601 w_k1_ADM_s117
                                 + 0.863591 w_k1_AXP_s117
                                 + 0.877243 w_k1_DIS_s117
                                 + 0.817478 w_k1_FDX_s117
                                 + 1.16204 w_k1_IBM_s117
                                 + 1.10633 w_k1_INTC_s117
                                 + 1.06635 w_k1_KO_s117
                                 + 0.752803 w_k1_MAT_s117
                                 + 0.839841 w_k1_MSFT_s117
                                 + 0.903745 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.08214 w_k1_ADM_s117
                                 + 1.10087 w_k1_AXP_s117
                                 + 1.06288 w_k1_DIS_s117
                                 + 1.01989 w_k1_FDX_s117
                                 + 1.19657 w_k1_IBM_s117
                                 + 1.10055 w_k1_INTC_s117
                                 + 1.34981 w_k1_KO_s117 + 1.13159 w_k1_MAT_s117
                                 + 1.00207 w_k1_MSFT_s117
                                 + 1.03494 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.13221 w_k1_ADM_s117
                                 + 0.778289 w_k1_AXP_s117
                                 + 0.984888 w_k1_DIS_s117
                                 + 0.898248 w_k1_FDX_s117
                                 + 0.841923 w_k1_IBM_s117
                                 + 1.09056 w_k1_INTC_s117
                                 + 1.02202 w_k1_KO_s117
                                 + 0.669539 w_k1_MAT_s117
                                 + 0.910208 w_k1_MSFT_s117
                                 + 0.914178 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.34797 w_k1_ADM_s117
                                 + 1.43697 w_k1_AXP_s117
                                 + 1.53163 w_k1_DIS_s117
                                 + 1.36536 w_k1_FDX_s117
                                 + 1.28344 w_k1_IBM_s117
                                 + 1.3672 w_k1_INTC_s117 + 1.36235 w_k1_KO_s117
                                 + 1.27533 w_k1_MAT_s117
                                 + 1.2138 w_k1_MSFT_s117 + 1.08183 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.859693 w_k1_ADM_s117
                                 + 0.751925 w_k1_AXP_s117
                                 + 0.691431 w_k1_DIS_s117
                                 + 0.813847 w_k1_FDX_s117
                                 + 0.732538 w_k1_IBM_s117
                                 + 0.953321 w_k1_INTC_s117
                                 + 0.840719 w_k1_KO_s117
                                 + 0.653174 w_k1_MAT_s117
                                 + 0.827696 w_k1_MSFT_s117
                                 + 0.779974 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.888675 w_k1_ADM_s117
                                 + 0.985988 w_k1_AXP_s117
                                 + 0.948656 w_k1_DIS_s117
                                 + 1.0997 w_k1_FDX_s117 + 0.76018 w_k1_IBM_s117
                                 + 1.14332 w_k1_INTC_s117
                                 + 0.805271 w_k1_KO_s117
                                 + 0.818502 w_k1_MAT_s117
                                 + 0.960765 w_k1_MSFT_s117
                                 + 0.844756 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0152696499789607 w_k1_ADM_s000
                                 + 0.0300376370348191 w_k1_ADM_s001
                                 + 0.0246696601826443 w_k1_ADM_s002
                                 + 0.0240382828320696 w_k1_ADM_s003
                                 + 0.0141065773819478 w_k1_ADM_s004
                                 + 0.0165978800058754 w_k1_ADM_s005
                                 + 0.0213082393534185 w_k1_ADM_s006
                                 + 0.00608248973625712 w_k1_ADM_s007
                                 + 0.0285971068102065 w_k1_ADM_s010
                                 + 0.0368872433165189 w_k1_ADM_s011
                                 + 0.011346488956728 w_k1_ADM_s012
                                 + 0.00875612068298729 w_k1_ADM_s013
                                 + 0.0127303443997073 w_k1_ADM_s014
                                 + 0.0403302299404096 w_k1_ADM_s015
                                 + 0.0379064725421452 w_k1_ADM_s016
                                 + 0.0249830595580559 w_k1_ADM_s017
                                 + 0.0450733459598711 w_k1_ADM_s100
                                 + 0.0288451036670363 w_k1_ADM_s101
                                 + 0.046969842744118 w_k1_ADM_s102
                                 + 0.0336388822379182 w_k1_ADM_s103
                                 + 0.0488941628791694 w_k1_ADM_s104
                                 + 0.0307834504014353 w_k1_ADM_s105
                                 + 0.0586760926277759 w_k1_ADM_s106
                                 + 0.0592562153997515 w_k1_ADM_s107
                                 + 0.0215829559460907 w_k1_ADM_s110
                                 + 0.0402502144450075 w_k1_ADM_s111
                                 + 0.0300846270126061 w_k1_ADM_s112
                                 + 0.0180360403202261 w_k1_ADM_s113
                                 + 0.0439178885198988 w_k1_ADM_s114
                                 + 0.0311157737574372 w_k1_ADM_s115
                                 + 0.0519973229847787 w_k1_ADM_s116
                                 + 0.0778789645085748 w_k1_ADM_s117
                                 + 0.0205787192727065 w_k1_AXP_s000
                                 + 0.0311777309090324 w_k1_AXP_s001
                                 + 0.0256094802524678 w_k1_AXP_s002
                                 + 0.0227966037689461 w_k1_AXP_s003
                                 + 0.0133287370090612 w_k1_AXP_s004
                                 + 0.0187142080818022 w_k1_AXP_s005
                                 + 0.0221907405527697 w_k1_AXP_s006
                                 + 0.00769601800045186 w_k1_AXP_s007
                                 + 0.0262259043858119 w_k1_AXP_s010
                                 + 0.0405995883117568 w_k1_AXP_s011
                                 + 0.0126561564198624 w_k1_AXP_s012
                                 + 0.00912391222927571 w_k1_AXP_s013
                                 + 0.0142825649196547 w_k1_AXP_s014
                                 + 0.0389977114733379 w_k1_AXP_s015
                                 + 0.0357097249783716 w_k1_AXP_s016
                                 + 0.027936026348688 w_k1_AXP_s017
                                 + 0.0468252563199256 w_k1_AXP_s100
                                 + 0.0275857993315025 w_k1_AXP_s101
                                 + 0.0486980486959393 w_k1_AXP_s102
                                 + 0.0329755699965635 w_k1_AXP_s103
                                 + 0.0507703383417787 w_k1_AXP_s104
                                 + 0.0279723021536161 w_k1_AXP_s105
                                 + 0.0528026240798195 w_k1_AXP_s106
                                 + 0.0611682751054161 w_k1_AXP_s107
                                 + 0.0215443356654592 w_k1_AXP_s110
                                 + 0.0362261451049101 w_k1_AXP_s111
                                 + 0.0288703583055436 w_k1_AXP_s112
                                 + 0.0183798769357497 w_k1_AXP_s113
                                 + 0.0492904614791963 w_k1_AXP_s114
                                 + 0.027703744829994 w_k1_AXP_s115
                                 + 0.0487382920981562 w_k1_AXP_s116
                                 + 0.0768252214592838 w_k1_AXP_s117
                                 + 0.0185462909400598 w_k1_DIS_s000
                                 + 0.0321630919750246 w_k1_DIS_s001
                                 + 0.0257576315796027 w_k1_DIS_s002
                                 + 0.0232326073085518 w_k1_DIS_s003
                                 + 0.0134442759587211 w_k1_DIS_s004
                                 + 0.0183773747793111 w_k1_DIS_s005
                                 + 0.0220411151373484 w_k1_DIS_s006
                                 + 0.00766844042795132 w_k1_DIS_s007
                                 + 0.0270764741476299 w_k1_DIS_s010
                                 + 0.0429675191930427 w_k1_DIS_s011
                                 + 0.0117691513381146 w_k1_DIS_s012
                                 + 0.00869921665166109 w_k1_DIS_s013
                                 + 0.0141179028283391 w_k1_DIS_s014
                                 + 0.0389715591900903 w_k1_DIS_s015
                                 + 0.0394770923549334 w_k1_DIS_s016
                                 + 0.0267541620117876 w_k1_DIS_s017
                                 + 0.0533543905051807 w_k1_DIS_s100
                                 + 0.0275038248163976 w_k1_DIS_s101
                                 + 0.0519105920673747 w_k1_DIS_s102
                                 + 0.0351695336017088 w_k1_DIS_s103
                                 + 0.0495438433813797 w_k1_DIS_s104
                                 + 0.028371771522974 w_k1_DIS_s105
                                 + 0.0535154435375286 w_k1_DIS_s106
                                 + 0.0661529024881428 w_k1_DIS_s107
                                 + 0.0224946887824577 w_k1_DIS_s110
                                 + 0.0376811989334069 w_k1_DIS_s111
                                 + 0.0300764907809272 w_k1_DIS_s112
                                 + 0.0207042945009972 w_k1_DIS_s113
                                 + 0.05106986490142 w_k1_DIS_s114
                                 + 0.0281143179746509 w_k1_DIS_s115
                                 + 0.0544547490884556 w_k1_DIS_s116
                                 + 0.0803017619829963 w_k1_DIS_s117
                                 + 0.0168779323242028 w_k1_FDX_s000
                                 + 0.0300162012207148 w_k1_FDX_s001
                                 + 0.024895908756194 w_k1_FDX_s002
                                 + 0.0215460628439616 w_k1_FDX_s003
                                 + 0.0123103693962568 w_k1_FDX_s004
                                 + 0.0177040521406135 w_k1_FDX_s005
                                 + 0.0218025087503477 w_k1_FDX_s006
                                 + 0.007272887532294 w_k1_FDX_s007
                                 + 0.0269051244838872 w_k1_FDX_s010
                                 + 0.0402966278082693 w_k1_FDX_s011
                                 + 0.0119579953461679 w_k1_FDX_s012
                                 + 0.00879999243385604 w_k1_FDX_s013
                                 + 0.012409389502096 w_k1_FDX_s014
                                 + 0.0398739027715842 w_k1_FDX_s015
                                 + 0.0339969279443675 w_k1_FDX_s016
                                 + 0.0270697257059774 w_k1_FDX_s017
                                 + 0.0471248588088729 w_k1_FDX_s100
                                 + 0.0280127727623825 w_k1_FDX_s101
                                 + 0.0469796345487488 w_k1_FDX_s102
                                 + 0.035836881758631 w_k1_FDX_s103
                                 + 0.0488967906211257 w_k1_FDX_s104
                                 + 0.0258683340535381 w_k1_FDX_s105
                                 + 0.0535573975206747 w_k1_FDX_s106
                                 + 0.0648250720751559 w_k1_FDX_s107
                                 + 0.0219482490336207 w_k1_FDX_s110
                                 + 0.0373210434238326 w_k1_FDX_s111
                                 + 0.0288627209256397 w_k1_FDX_s112
                                 + 0.0180498641687554 w_k1_FDX_s113
                                 + 0.0483837725263666 w_k1_FDX_s114
                                 + 0.0280742069800136 w_k1_FDX_s115
                                 + 0.052523533835951 w_k1_FDX_s116
                                 + 0.0781391560719399 w_k1_FDX_s117
                                 + 0.0176775512282339 w_k1_IBM_s000
                                 + 0.0308474248601057 w_k1_IBM_s001
                                 + 0.0267279938253541 w_k1_IBM_s002
                                 + 0.0229665827155437 w_k1_IBM_s003
                                 + 0.0133539008661855 w_k1_IBM_s004
                                 + 0.0173373815232805 w_k1_IBM_s005
                                 + 0.0226479740843337 w_k1_IBM_s006
                                 + 0.00713150182959592 w_k1_IBM_s007
                                 + 0.0275240857942036 w_k1_IBM_s010
                                 + 0.0413935234703571 w_k1_IBM_s011
                                 + 0.0126819976794285 w_k1_IBM_s012
                                 + 0.00887572676844433 w_k1_IBM_s013
                                 + 0.0134835446028743 w_k1_IBM_s014
                                 + 0.0392360522520991 w_k1_IBM_s015
                                 + 0.0371037652743027 w_k1_IBM_s016
                                 + 0.0272329715337111 w_k1_IBM_s017
                                 + 0.0481238167269185 w_k1_IBM_s100
                                 + 0.0283664184469306 w_k1_IBM_s101
                                 + 0.049081928036799 w_k1_IBM_s102
                                 + 0.0349726551534629 w_k1_IBM_s103
                                 + 0.0501400320957478 w_k1_IBM_s104
                                 + 0.0269723870804098 w_k1_IBM_s105
                                 + 0.0541479602843357 w_k1_IBM_s106
                                 + 0.0684870989788301 w_k1_IBM_s107
                                 + 0.022001528486783 w_k1_IBM_s110
                                 + 0.0372197875524984 w_k1_IBM_s111
                                 + 0.0287772854023827 w_k1_IBM_s112
                                 + 0.019211153551519 w_k1_IBM_s113
                                 + 0.0501955577708101 w_k1_IBM_s114
                                 + 0.0277912628241323 w_k1_IBM_s115
                                 + 0.0527518697446075 w_k1_IBM_s116
                                 + 0.0772285628336463 w_k1_IBM_s117
                                 + 0.0178522050287161 w_k1_INTC_s000
                                 + 0.0261147599450299 w_k1_INTC_s001
                                 + 0.0254820253914143 w_k1_INTC_s002
                                 + 0.01973043324319 w_k1_INTC_s003
                                 + 0.0123664041097101 w_k1_INTC_s004
                                 + 0.0201162250316044 w_k1_INTC_s005
                                 + 0.0230760036754881 w_k1_INTC_s006
                                 + 0.00676755636722615 w_k1_INTC_s007
                                 + 0.0266377612639986 w_k1_INTC_s010
                                 + 0.0412091554144843 w_k1_INTC_s011
                                 + 0.011706764786257 w_k1_INTC_s012
                                 + 0.00961088886948686 w_k1_INTC_s013
                                 + 0.0123834792377091 w_k1_INTC_s014
                                 + 0.0404021136937163 w_k1_INTC_s015
                                 + 0.0353446812506882 w_k1_INTC_s016
                                 + 0.0271773633976041 w_k1_INTC_s017
                                 + 0.0463804173197489 w_k1_INTC_s100
                                 + 0.0288853546414594 w_k1_INTC_s101
                                 + 0.0518312734241819 w_k1_INTC_s102
                                 + 0.0383872167499225 w_k1_INTC_s103
                                 + 0.0529284107599706 w_k1_INTC_s104
                                 + 0.0267934986240687 w_k1_INTC_s105
                                 + 0.0491623670135037 w_k1_INTC_s106
                                 + 0.0643315924786115 w_k1_INTC_s107
                                 + 0.020134031513668 w_k1_INTC_s110
                                 + 0.0359161168399957 w_k1_INTC_s111
                                 + 0.030786963511798 w_k1_INTC_s112
                                 + 0.0174884115939053 w_k1_INTC_s113
                                 + 0.0567251451592343 w_k1_INTC_s114
                                 + 0.025884673487502 w_k1_INTC_s115
                                 + 0.0604031354806249 w_k1_INTC_s116
                                 + 0.0876321594876893 w_k1_INTC_s117
                                 + 0.0182021776090803 w_k1_KO_s000
                                 + 0.0337808339332749 w_k1_KO_s001
                                 + 0.028799139485085 w_k1_KO_s002
                                 + 0.0223965082739381 w_k1_KO_s003
                                 + 0.0123892614775416 w_k1_KO_s004
                                 + 0.0168581611774584 w_k1_KO_s005
                                 + 0.0221655034792159 w_k1_KO_s006
                                 + 0.00631277264929898 w_k1_KO_s007
                                 + 0.0299978567588991 w_k1_KO_s010
                                 + 0.0416326228898451 w_k1_KO_s011
                                 + 0.0114655004321786 w_k1_KO_s012
                                 + 0.00910401715930363 w_k1_KO_s013
                                 + 0.0134522590648873 w_k1_KO_s014
                                 + 0.0412039791892745 w_k1_KO_s015
                                 + 0.0393524729975691 w_k1_KO_s016
                                 + 0.0239359663530837 w_k1_KO_s017
                                 + 0.0473993911544549 w_k1_KO_s100
                                 + 0.024988989524029 w_k1_KO_s101
                                 + 0.0496104694245069 w_k1_KO_s102
                                 + 0.0372617512501206 w_k1_KO_s103
                                 + 0.0429730581765864 w_k1_KO_s104
                                 + 0.0276471807356296 w_k1_KO_s105
                                 + 0.0537271860513573 w_k1_KO_s106
                                 + 0.0651672997563233 w_k1_KO_s107
                                 + 0.0198622735089939 w_k1_KO_s110
                                 + 0.0343351590126586 w_k1_KO_s111
                                 + 0.0293798401651054 w_k1_KO_s112
                                 + 0.0213104311710998 w_k1_KO_s113
                                 + 0.0517603089407335 w_k1_KO_s114
                                 + 0.0289573532670241 w_k1_KO_s115
                                 + 0.0526746310511292 w_k1_KO_s116
                                 + 0.084454535456587 w_k1_KO_s117
                                 + 0.0221057392815979 w_k1_MAT_s000
                                 + 0.0325047421867299 w_k1_MAT_s001
                                 + 0.0375839620136761 w_k1_MAT_s002
                                 + 0.0213199116376175 w_k1_MAT_s003
                                 + 0.0114697175509142 w_k1_MAT_s004
                                 + 0.0158300869720317 w_k1_MAT_s005
                                 + 0.0218750789418205 w_k1_MAT_s006
                                 + 0.00705675759370067 w_k1_MAT_s007
                                 + 0.0272012891941841 w_k1_MAT_s010
                                 + 0.0455551198346335 w_k1_MAT_s011
                                 + 0.013150035873339 w_k1_MAT_s012
                                 + 0.00827283446428997 w_k1_MAT_s013
                                 + 0.0138917567925325 w_k1_MAT_s014
                                 + 0.0393737375941163 w_k1_MAT_s015
                                 + 0.0375486777253207 w_k1_MAT_s016
                                 + 0.0277653988076856 w_k1_MAT_s017
                                 + 0.0557732735151889 w_k1_MAT_s100
                                 + 0.0316406370476382 w_k1_MAT_s101
                                 + 0.0495330926384898 w_k1_MAT_s102
                                 + 0.0354590010809927 w_k1_MAT_s103
                                 + 0.0500831146847646 w_k1_MAT_s104
                                 + 0.0214724296797247 w_k1_MAT_s105
                                 + 0.0510443668922095 w_k1_MAT_s106
                                 + 0.0709125634261103 w_k1_MAT_s107
                                 + 0.0220610040364701 w_k1_MAT_s110
                                 + 0.0359159797019402 w_k1_MAT_s111
                                 + 0.0243332653172632 w_k1_MAT_s112
                                 + 0.0209631936892222 w_k1_MAT_s113
                                 + 0.04175459672272 w_k1_MAT_s114
                                 + 0.0270814081496258 w_k1_MAT_s115
                                 + 0.0483118913166631 w_k1_MAT_s116
                                 + 0.0690722920528972 w_k1_MAT_s117
                                 + 0.0178422830763596 w_k1_MSFT_s000
                                 + 0.0300869293183456 w_k1_MSFT_s001
                                 + 0.0273214004879549 w_k1_MSFT_s002
                                 + 0.021995185564732 w_k1_MSFT_s003
                                 + 0.0124799806296035 w_k1_MSFT_s004
                                 + 0.0178759960477092 w_k1_MSFT_s005
                                 + 0.0197506049009337 w_k1_MSFT_s006
                                 + 0.00698560472808297 w_k1_MSFT_s007
                                 + 0.0276832200115094 w_k1_MSFT_s010
                                 + 0.0422712780798679 w_k1_MSFT_s011
                                 + 0.0117956056592729 w_k1_MSFT_s012
                                 + 0.00907158522781291 w_k1_MSFT_s013
                                 + 0.0140367258094654 w_k1_MSFT_s014
                                 + 0.0361140303248296 w_k1_MSFT_s015
                                 + 0.035534402965518 w_k1_MSFT_s016
                                 + 0.0281448243643047 w_k1_MSFT_s017
                                 + 0.0492593039071813 w_k1_MSFT_s100
                                 + 0.0263736258951732 w_k1_MSFT_s101
                                 + 0.0427039918987163 w_k1_MSFT_s102
                                 + 0.0377967654343592 w_k1_MSFT_s103
                                 + 0.0532344753337026 w_k1_MSFT_s104
                                 + 0.0275277567024542 w_k1_MSFT_s105
                                 + 0.0486772996346225 w_k1_MSFT_s106
                                 + 0.0633079990873082 w_k1_MSFT_s107
                                 + 0.0207177513555502 w_k1_MSFT_s110
                                 + 0.0351087624945972 w_k1_MSFT_s111
                                 + 0.0282160973750241 w_k1_MSFT_s112
                                 + 0.0199993951763698 w_k1_MSFT_s113
                                 + 0.0445368544614011 w_k1_MSFT_s114
                                 + 0.0291120193749777 w_k1_MSFT_s115
                                 + 0.0524701652467998 w_k1_MSFT_s116
                                 + 0.0753342295164663 w_k1_MSFT_s117
                                 + 0.0175074937545692 w_k1_PG_s000
                                 + 0.032596142020002 w_k1_PG_s001
                                 + 0.0242871887441055 w_k1_PG_s002
                                 + 0.0204764610724352 w_k1_PG_s003
                                 + 0.0125703796022192 w_k1_PG_s004
                                 + 0.0179041874756933 w_k1_PG_s005
                                 + 0.0221492360430577 w_k1_PG_s006
                                 + 0.00713671768436745 w_k1_PG_s007
                                 + 0.027491235590257 w_k1_PG_s010
                                 + 0.0402646109027826 w_k1_PG_s011
                                 + 0.0121349209937436 w_k1_PG_s012
                                 + 0.00920443164120079 w_k1_PG_s013
                                 + 0.0120410808086868 w_k1_PG_s014
                                 + 0.0431082799837271 w_k1_PG_s015
                                 + 0.0334042227157854 w_k1_PG_s016
                                 + 0.0250387885513535 w_k1_PG_s017
                                 + 0.047308213437516 w_k1_PG_s100
                                 + 0.0266592342071989 w_k1_PG_s101
                                 + 0.0472842652265699 w_k1_PG_s102
                                 + 0.03642648009479 w_k1_PG_s103
                                 + 0.0469422137547127 w_k1_PG_s104
                                 + 0.0257590215690672 w_k1_PG_s105
                                 + 0.0540929737400767 w_k1_PG_s106
                                 + 0.0632113986155771 w_k1_PG_s107
                                 + 0.0189699328507844 w_k1_PG_s110
                                 + 0.0373916855456947 w_k1_PG_s111
                                 + 0.0265935629670307 w_k1_PG_s112
                                 + 0.0199475190411913 w_k1_PG_s113
                                 + 0.0441742947763375 w_k1_PG_s114
                                 + 0.0273274182483676 w_k1_PG_s115
                                 + 0.0541502382366725 w_k1_PG_s116
                                 + 0.0720828328719736 w_k1_PG_s117 >= 1.209
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
