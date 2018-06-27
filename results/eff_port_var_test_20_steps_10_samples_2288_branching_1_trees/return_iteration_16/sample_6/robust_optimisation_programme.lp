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
 w_k1_ADM_s0_return_constraint:  1.14285 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.922897 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.24733 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.933359 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.33288 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.919098 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.908986 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.24397 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.655131 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.02167 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.932854 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.79104 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.900013 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.31469 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.02101 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.2397 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.25911 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.918816 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.08158 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.17037 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.760026 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.853045 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.851456 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.13019 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.786378 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.710291 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.941616 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.27466 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.895241 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.08017 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.741765 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.01847 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.0741 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.941062 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.16739 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.921681 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.18972 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.981317 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.31865 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.887651 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.3694 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.892614 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.32967 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.887323 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.846756 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.27874 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.575727 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.986991 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.9077 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.753111 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.04546 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.48664 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.900313 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.07461 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.0818 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.787358 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.30747 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.11825 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.689345 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.793672 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.27731 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.991723 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.833364 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.787277 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.05474 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.39137 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.913076 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.46761 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.627543 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.910257 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.27215 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.785293 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.56306 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.00783 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.23148 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.969219 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.3729 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.911202 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.38112 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.814997 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.45723 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.85056 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.860415 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.40554 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.607432 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.01892 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.741325 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.846722 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.20247 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.83426 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.842158 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.20688 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.3019 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.806058 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.20344 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.20003 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.598146 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.835394 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.997422 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.00922 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.72095 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.00111 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.905648 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.51119 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.813411 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.27835 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.562379 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.816138 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.35378 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.81136 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.63299 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.690416 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.21603 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.889637 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.24219 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.929314 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.22595 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.895609 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.26544 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.918495 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.00986 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.26054 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.66647 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.03026 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.784617 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.866256 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.938919 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.5856 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.963388 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.03434 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.1603 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.866406 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.14554 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.27193 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.710348 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.800108 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.885846 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.18557 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.762743 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.901077 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.930181 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.28935 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.932147 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.10298 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.65867 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.824702 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.19618 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.87049 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.25391 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.837963 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.06975 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.07212 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.1544 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.973998 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.24788 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.926302 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.28515 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.897779 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.558312 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.09793 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.781601 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.03929 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.15733 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.81043 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.72736 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.00067 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.818792 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.60956 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.92762 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.632753 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.15051 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.658903 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.668848 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.06547 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 2.03458 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.542067 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.19522 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.982991 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.882489 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.01635 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.647969 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.51433 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.732038 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.27069 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.946364 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.973692 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.76649 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.764668 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.42888 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.61025 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.26338 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.924423 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.07739 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.885065 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.20244 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.957728 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.953801 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.23287 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.749771 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.980752 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.964838 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.797152 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.01935 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.44231 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.759547 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.01946 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.24376 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.828945 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.10238 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.09742 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.747336 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.790822 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.997692 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.942638 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.964106 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.13568 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.974882 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.14405 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.979645 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.14124 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.676215 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.838399 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.3234 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.948908 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.35083 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.871742 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.996001 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.975334 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.14689 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.946098 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.25665 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.868725 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.39284 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.925393 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.833136 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.17441 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.721736 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.1015 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.962996 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.933518 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.46436 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.43866 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.885525 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.41184 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.19995 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.779991 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.15303 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.03733 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.783599 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.07524 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.144 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.707909 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.883496 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.938716 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.892237 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.06742 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.713515 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.24033 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.656154 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.942614 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.16993 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.913978 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.49448 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.782182 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.2169 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.754141 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.31744 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.844199 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.32137 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.888004 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.58473 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.11547 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.733441 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.07508 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.726598 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.89613 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.10216 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.814196 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.84192 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.23217 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.980583 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.37309 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.772875 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.581768 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.32429 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.850854 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.586676 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.820728 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.88865 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.426332 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.976357 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.897908 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.919282 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.8737 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.498723 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.71374 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.474694 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.15919 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.85725 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.845393 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.62833 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.778025 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.31555 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.620471 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.20668 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.937378 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.1541 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.941219 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.21453 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.955783 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.03185 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.22428 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.760861 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.948151 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.83349 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.906328 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.20611 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.38773 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.851244 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.988942 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.1676 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.808286 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.983759 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.08087 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.689013 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.818668 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.956494 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.12026 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.83041 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.00822 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.952038 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.22408 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.952283 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.15276 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.696745 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.800929 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.15827 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.905555 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.24861 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.840992 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.04168 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.06012 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15115 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.955555 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.15578 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.924728 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.23491 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.894859 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.920323 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.06263 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.736718 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.961244 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.970936 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.912703 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.06277 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.3466 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.988299 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.05264 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.05383 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.866075 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.01787 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11377 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.791917 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.935516 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.99682 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.911327 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.852549 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.887115 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.00564 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.02861 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.894566 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.13039 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.844084 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.960981 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.14681 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.00755 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.34805 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.902498 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.19339 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.932557 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.155999938446421 z_k1_s0000
                                 - 0.298999441747478 z_k1_s0001
                                 - 0.121999994965774 z_k1_s0002
                                 - 0.0739999523284375 z_k1_s0003
                                 - 0.16200032401344 z_k1_s0004
                                 - 0.324999671638454 z_k1_s0005
                                 - 0.208999261740011 z_k1_s0006
                                 - 0.254000152243433 z_k1_s0007
                                 - 0.165999303119477 z_k1_s0010
                                 - 0.174000400884237 z_k1_s0011
                                 - 0.171999316943071 z_k1_s0012
                                 - 0.226999983326004 z_k1_s0013
                                 - 0.17100039397244 z_k1_s0014
                                 - 0.107000030654106 z_k1_s0015
                                 - 0.494000598475954 z_k1_s0016
                                 - 0.107999925024709 z_k1_s0017
                                 - 0.0810002547201984 z_k1_s0020
                                 - 0.0679999025056968 z_k1_s0021
                                 - 0.0739999213738596 z_k1_s0022
                                 - 0.0269999915066882 z_k1_s0023
                                 - 0.0859999591101853 z_k1_s0024
                                 - 0.0460000356891964 z_k1_s0025
                                 - 0.0559998647693711 z_k1_s0026
                                 - 0.0290000133627499 z_k1_s0027
                                 - 0.269999526971683 z_k1_s0030
                                 - 0.186999529384199 z_k1_s0031
                                 - 0.1759999776554 z_k1_s0032
                                 - 0.111999946780739 z_k1_s0033
                                 - 0.232000165545608 z_k1_s0034
                                 - 0.25999954449125 z_k1_s0035
                                 - 0.137999901229864 z_k1_s0036
                                 - 0.0549999036423798 z_k1_s0037
                                 - 0.211999993993838 z_k1_s0040
                                 - 0.230000253635195 z_k1_s0041
                                 - 0.198000558051651 z_k1_s0042
                                 - 0.0580001089211131 z_k1_s0043
                                 - 0.167000493892083 z_k1_s0044
                                 - 0.0700000522150236 z_k1_s0045
                                 - 0.0800001581601673 z_k1_s0046
                                 - 0.134000017882928 z_k1_s0047
                                 - 0.197001356537901 z_k1_s0050
                                 - 0.198000076393531 z_k1_s0051
                                 - 0.324001108829576 z_k1_s0052
                                 - 0.0860002924103122 z_k1_s0053
                                 - 0.104000439901176 z_k1_s0054
                                 - 0.104000439901176 z_k1_s0055
                                 - 0.13500048777909 z_k1_s0056
                                 - 0.398002230516227 z_k1_s0057
                                 - 0.11799986800605 z_k1_s0060
                                 - 0.190999724345426 z_k1_s0061
                                 - 0.442999023199296 z_k1_s0062
                                 - 0.128999894084555 z_k1_s0063
                                 - 0.356000115523005 z_k1_s0064
                                 - 0.227999780391328 z_k1_s0065
                                 - 0.0969998974015797 z_k1_s0066
                                 - 0.180000743466244 z_k1_s0067
                                 - 0.0839998529017469 z_k1_s0070
                                 - 0.0450000303331799 z_k1_s0071
                                 - 0.0480001805155293 z_k1_s0072
                                 - 0.0400000423961742 z_k1_s0073
                                 - 0.0380000194413462 z_k1_s0074
                                 - 0.10900002408699 z_k1_s0075
                                 - 0.0600002256444116 z_k1_s0076
                                 - 0.0390000540687817 z_k1_s0077
                                 - 0.0870001017024068 z_k1_s0100
                                 - 0.0920001052278302 z_k1_s0101
                                 - 0.117999639239555 z_k1_s0102
                                 - 0.256000705182195 z_k1_s0103
                                 - 0.125000088135585 z_k1_s0104
                                 - 0.150000105762702 z_k1_s0105
                                 - 0.0600000423050807 z_k1_s0106
                                 - 0.121000408195564 z_k1_s0107
                                 - 0.502999817333553 z_k1_s0110
                                 - 0.40099960854324 z_k1_s0111
                                 - 0.6360013530925 z_k1_s0112
                                 - 0.577997409681341 z_k1_s0113
                                 - 0.222999952775287 z_k1_s0114
                                 - 0.836999254581222 z_k1_s0115
                                 - 0.907999878001563 z_k1_s0116
                                 - 0.267999814382955 z_k1_s0117
                                 - 0.395999552716889 z_k1_s0120
                                 - 0.201000401479765 z_k1_s0121
                                 - 0.281000264161909 z_k1_s0122
                                 - 0.486000426335508 z_k1_s0123
                                 - 0.57400231360826 z_k1_s0124
                                 - 0.697002553952587 z_k1_s0125
                                 - 0.54600220074932 z_k1_s0126
                                 - 0.395001847546053 z_k1_s0127
                                 - 0.177999335185099 z_k1_s0130
                                 - 0.336000099260061 z_k1_s0131
                                 - 0.255000355411767 z_k1_s0132
                                 - 0.159000114780235 z_k1_s0133
                                 - 0.190999314457055 z_k1_s0134
                                 - 0.190999314457055 z_k1_s0135
                                 - 0.0790000518373473 z_k1_s0136
                                 - 0.262000598250616 z_k1_s0137
                                 - 0.340000569871073 z_k1_s0140
                                 - 0.233000015470063 z_k1_s0141
                                 - 0.192000696868779 z_k1_s0142
                                 - 0.145000121139001 z_k1_s0143
                                 - 0.139000139211895 z_k1_s0144
                                 - 0.132000080596852 z_k1_s0145
                                 - 0.278000278423791 z_k1_s0146
                                 - 0.135000151260492 z_k1_s0147
                                 - 0.344998713699256 z_k1_s0150
                                 - 0.53599916349186 z_k1_s0151
                                 - 0.657000934417659 z_k1_s0152
                                 - 0.352000629785377 z_k1_s0153
                                 - 0.26100000645917 z_k1_s0154
                                 - 0.310998602992728 z_k1_s0155
                                 - 0.182000076252735 z_k1_s0156
                                 - 0.281999902719131 z_k1_s0157
                                 - 0.125000156219316 z_k1_s0160
                                 - 0.1040000708144 z_k1_s0161
                                 - 0.119000193960844 z_k1_s0162
                                 - 0.132000184687622 z_k1_s0163
                                 - 0.221999618564709 z_k1_s0164
                                 - 0.1040000708144 z_k1_s0165
                                 - 0.470000668584727 z_k1_s0166
                                 - 0.17400021049728 z_k1_s0167
                                 - 0.374000974569343 z_k1_s0170
                                 - 0.27300085998025 z_k1_s0171
                                 - 0.184000538905969 z_k1_s0172
                                 - 0.445001605371408 z_k1_s0173
                                 - 0.161000638268272 z_k1_s0174
                                 - 0.507001279533705 z_k1_s0175
                                 - 0.10300037657587 z_k1_s0176
                                 - 0.176000602462517 z_k1_s0177
                                 - 0.0999999421919068 z_k1_s1000
                                 - 0.171999549450356 z_k1_s1001
                                 - 0.0939999749203693 z_k1_s1002
                                 - 0.124999666490089 z_k1_s1003
                                 - 0.103999885539626 z_k1_s1004
                                 - 0.205999642655516 z_k1_s1005
                                 - 0.114000005158718 z_k1_s1006
                                 - 0.129999987549429 z_k1_s1007
                                 - 0.222000289412364 z_k1_s1010
                                 - 0.157999947383615 z_k1_s1011
                                 - 0.0779999982476035 z_k1_s1012
                                 - 0.131000251590166 z_k1_s1013
                                 - 0.0910000378221979 z_k1_s1014
                                 - 0.104999933241015 z_k1_s1015
                                 - 0.202999440372696 z_k1_s1016
                                 - 0.207999915593622 z_k1_s1017
                                 - 0.226000474483115 z_k1_s1020
                                 - 0.876003002189584 z_k1_s1021
                                 - 0.420998983831318 z_k1_s1022
                                 - 0.678002829051683 z_k1_s1023
                                 - 0.359999356430955 z_k1_s1024
                                 - 0.338000625662219 z_k1_s1025
                                 - 0.397001837940594 z_k1_s1026
                                 - 0.21800048878465 z_k1_s1027
                                 - 0.272000007580775 z_k1_s1030
                                 - 0.300000195890456 z_k1_s1031
                                 - 0.204000005685581 z_k1_s1032
                                 - 0.0859999180086747 z_k1_s1033
                                 - 0.328999821640087 z_k1_s1034
                                 - 0.139999985148939 z_k1_s1035
                                 - 0.17500037993595 z_k1_s1036
                                 - 0.0879999086879504 z_k1_s1037
                                 - 0.526001372407683 z_k1_s1040
                                 - 0.626000708727866 z_k1_s1041
                                 - 0.408000439068625 z_k1_s1042
                                 - 0.359999756354212 z_k1_s1043
                                 - 0.257000153864216 z_k1_s1044
                                 - 0.459001387952849 z_k1_s1045
                                 - 0.721000793432558 z_k1_s1046
                                 - 0.21900011998234 z_k1_s1047
                                 - 0.354000611249704 z_k1_s1050
                                 - 0.332000534854476 z_k1_s1051
                                 - 0.545999911824511 z_k1_s1052
                                 - 0.457000627381475 z_k1_s1053
                                 - 0.387000546942319 z_k1_s1054
                                 - 0.601999260574932 z_k1_s1055
                                 - 0.316000414239682 z_k1_s1056
                                 - 0.584001897836793 z_k1_s1057
                                 - 0.308999272319136 z_k1_s1060
                                 - 0.416998460031173 z_k1_s1061
                                 - 0.285999270313328 z_k1_s1062
                                 - 0.605998677747933 z_k1_s1063
                                 - 0.480997492980322 z_k1_s1064
                                 - 0.442998428759565 z_k1_s1065
                                 - 0.816999165695631 z_k1_s1066
                                 - 0.497999104336909 z_k1_s1067
                                 - 0.463000633254521 z_k1_s1070
                                 - 0.454999995334284 z_k1_s1071
                                 - 0.112999989897305 z_k1_s1072
                                 - 0.212000084284336 z_k1_s1073
                                 - 0.503001109855392 z_k1_s1074
                                 - 0.214000108114379 z_k1_s1075
                                 - 0.244999934879992 z_k1_s1076
                                 - 0.508001169430501 z_k1_s1077
                                 - 0.330000801626626 z_k1_s1100
                                 - 0.731002686517118 z_k1_s1101
                                 - 0.375000671257063 z_k1_s1102
                                 - 0.322000590447425 z_k1_s1103
                                 - 0.360000538946575 z_k1_s1104
                                 - 1.16800341250991 z_k1_s1105
                                 - 0.644999889039684 z_k1_s1106
                                 - 1.34200220534623 z_k1_s1107
                                 - 1.29099733444593 z_k1_s1110
                                 - 0.77299832064197 z_k1_s1111
                                 - 0.933000661204958 z_k1_s1112
                                 - 0.430999616658959 z_k1_s1113
                                 - 0.439999925252997 z_k1_s1114
                                 - 0.676999181102515 z_k1_s1115
                                 - 0.628000269066275 z_k1_s1116
                                 - 0.488999403488821 z_k1_s1117
                                 - 1.79399307613907 z_k1_s1120
                                 - 0.959998943920314 z_k1_s1121
                                 - 2.49299435187514 z_k1_s1122
                                 - 0.699998799479981 z_k1_s1123
                                 - 1.00099843248317 z_k1_s1124
                                 - 1.53199673631248 z_k1_s1125
                                 - 1.5389989167933 z_k1_s1126
                                 - 1.46000054618838 z_k1_s1127
                                 - 0.426000664047652 z_k1_s1130
                                 - 0.42499936354454 z_k1_s1131
                                 - 0.170000008917949 z_k1_s1132
                                 - 0.191000178969662 z_k1_s1133
                                 - 0.416000834023153 z_k1_s1134
                                 - 0.325999728801333 z_k1_s1135
                                 - 0.501000706732061 z_k1_s1136
                                 - 0.18000010244258 z_k1_s1137
                                 - 0.382001023856564 z_k1_s1140
                                 - 0.768004195988298 z_k1_s1141
                                 - 0.415000750889499 z_k1_s1142
                                 - 0.437000747978493 z_k1_s1143
                                 - 0.898001492781343 z_k1_s1144
                                 - 0.850002573534851 z_k1_s1145
                                 - 1.02300281924431 z_k1_s1146
                                 - 0.5990007265429 z_k1_s1147
                                 - 0.256000051758437 z_k1_s1150
                                 - 0.800998988809381 z_k1_s1151
                                 - 0.660999354363238 z_k1_s1152
                                 - 0.195999848324488 z_k1_s1153
                                 - 0.692000353718834 z_k1_s1154
                                 - 0.348999808926232 z_k1_s1155
                                 - 0.294999949925577 z_k1_s1156
                                 - 0.350999803704034 z_k1_s1157
                                 - 0.495001071322726 z_k1_s1160
                                 - 1.13500241170962 z_k1_s1161
                                 - 0.543000691825998 z_k1_s1162
                                 - 1.99100647536811 z_k1_s1163
                                 - 0.491000610613356 z_k1_s1164
                                 - 0.869998660380396 z_k1_s1165
                                 - 1.3450025976643 z_k1_s1166
                                 - 0.515001159366138 z_k1_s1167
                                 - 0.84899953589033 z_k1_s1170
                                 - 1.79099955092022 z_k1_s1171
                                 - 0.802999823971471 z_k1_s1172
                                 - 1.23499529868898 z_k1_s1173
                                 - 0.834999585654182 z_k1_s1174
                                 - 2.02899434493778 z_k1_s1175
                                 - 0.434999512622428 z_k1_s1176
                                 - 0.742999289817073 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.14652 w_k1_ADM_s000
                                 + 1.17916 w_k1_AXP_s000
                                 + 0.907681 w_k1_DIS_s000
                                 + 1.1025 w_k1_FDX_s000 + 1.13959 w_k1_IBM_s000
                                 + 0.949239 w_k1_INTC_s000
                                 + 0.846622 w_k1_KO_s000
                                 + 0.947921 w_k1_MAT_s000
                                 + 0.890063 w_k1_MSFT_s000
                                 + 1.02352 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.838583 w_k1_ADM_s000
                                 + 0.923249 w_k1_AXP_s000
                                 + 0.85998 w_k1_DIS_s000
                                 + 0.871311 w_k1_FDX_s000
                                 + 1.05127 w_k1_IBM_s000
                                 + 0.857262 w_k1_INTC_s000
                                 + 0.938417 w_k1_KO_s000
                                 + 0.89922 w_k1_MAT_s000
                                 + 0.854311 w_k1_MSFT_s000
                                 + 0.993743 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.899038 w_k1_ADM_s000
                                 + 0.823595 w_k1_AXP_s000
                                 + 1.13949 w_k1_DIS_s000
                                 + 0.994875 w_k1_FDX_s000
                                 + 0.794656 w_k1_IBM_s000
                                 + 1.06591 w_k1_INTC_s000
                                 + 0.915296 w_k1_KO_s000
                                 + 0.705393 w_k1_MAT_s000
                                 + 1.00812 w_k1_MSFT_s000
                                 + 0.916693 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.954506 w_k1_ADM_s000
                                 + 0.771182 w_k1_AXP_s000
                                 + 1.29443 w_k1_DIS_s000
                                 + 0.958665 w_k1_FDX_s000
                                 + 1.1628 w_k1_IBM_s000
                                 + 0.882974 w_k1_INTC_s000
                                 + 1.29494 w_k1_KO_s000
                                 + 0.899399 w_k1_MAT_s000
                                 + 1.07951 w_k1_MSFT_s000
                                 + 1.08972 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.93389 w_k1_ADM_s000
                                 + 1.09956 w_k1_AXP_s000
                                 + 1.15167 w_k1_DIS_s000 + 1.0415 w_k1_FDX_s000
                                 + 1.48122 w_k1_IBM_s000
                                 + 1.16816 w_k1_INTC_s000
                                 + 1.04942 w_k1_KO_s000
                                 + 0.968859 w_k1_MAT_s000
                                 + 1.00423 w_k1_MSFT_s000
                                 + 0.936987 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.897651 w_k1_ADM_s000
                                 + 0.596006 w_k1_AXP_s000
                                 + 0.660086 w_k1_DIS_s000
                                 + 0.776479 w_k1_FDX_s000
                                 + 0.731629 w_k1_IBM_s000
                                 + 0.853809 w_k1_INTC_s000
                                 + 0.685897 w_k1_KO_s000
                                 + 0.435662 w_k1_MAT_s000
                                 + 0.865453 w_k1_MSFT_s000
                                 + 0.817984 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.17619 w_k1_ADM_s000
                                 + 1.05885 w_k1_AXP_s000
                                 + 1.47315 w_k1_DIS_s000
                                 + 1.20215 w_k1_FDX_s000 + 1.1795 w_k1_IBM_s000
                                 + 1.19157 w_k1_INTC_s000
                                 + 1.21133 w_k1_KO_s000
                                 + 0.787637 w_k1_MAT_s000
                                 + 1.23604 w_k1_MSFT_s000
                                 + 1.24405 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.21524 w_k1_ADM_s000
                                 + 1.40365 w_k1_AXP_s000
                                 + 1.64983 w_k1_DIS_s000
                                 + 1.38653 w_k1_FDX_s000
                                 + 1.04514 w_k1_IBM_s000
                                 + 1.24958 w_k1_INTC_s000
                                 + 1.18806 w_k1_KO_s000
                                 + 0.723413 w_k1_MAT_s000
                                 + 1.27747 w_k1_MSFT_s000
                                 + 1.03951 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.04012 w_k1_ADM_s001
                                 + 1.21136 w_k1_AXP_s001
                                 + 0.988697 w_k1_DIS_s001
                                 + 1.06996 w_k1_FDX_s001
                                 + 0.820746 w_k1_IBM_s001
                                 + 1.04864 w_k1_INTC_s001
                                 + 0.97555 w_k1_KO_s001
                                 + 0.923739 w_k1_MAT_s001
                                 + 0.988303 w_k1_MSFT_s001
                                 + 1.14752 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.891756 w_k1_ADM_s001
                                 + 0.805456 w_k1_AXP_s001
                                 + 0.984135 w_k1_DIS_s001
                                 + 0.92913 w_k1_FDX_s001
                                 + 0.885447 w_k1_IBM_s001
                                 + 0.939889 w_k1_INTC_s001
                                 + 1.05893 w_k1_KO_s001
                                 + 0.716548 w_k1_MAT_s001
                                 + 0.962428 w_k1_MSFT_s001
                                 + 1.12068 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.01531 w_k1_ADM_s001
                                 + 1.30882 w_k1_AXP_s001
                                 + 1.55728 w_k1_DIS_s001
                                 + 1.09329 w_k1_FDX_s001
                                 + 1.53464 w_k1_IBM_s001
                                 + 1.29271 w_k1_INTC_s001
                                 + 1.42193 w_k1_KO_s001 + 1.46752 w_k1_MAT_s001
                                 + 1.21596 w_k1_MSFT_s001
                                 + 1.36124 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.769736 w_k1_ADM_s001
                                 + 0.810209 w_k1_AXP_s001
                                 + 1.10493 w_k1_DIS_s001
                                 + 0.96799 w_k1_FDX_s001
                                 + 1.20384 w_k1_IBM_s001
                                 + 0.975979 w_k1_INTC_s001
                                 + 1.30118 w_k1_KO_s001 + 1.28771 w_k1_MAT_s001
                                 + 1.06341 w_k1_MSFT_s001
                                 + 1.05123 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.23791 w_k1_ADM_s001
                                 + 1.2561 w_k1_AXP_s001 + 1.12626 w_k1_DIS_s001
                                 + 1.00021 w_k1_FDX_s001
                                 + 1.53478 w_k1_IBM_s001
                                 + 0.792506 w_k1_INTC_s001
                                 + 1.28975 w_k1_KO_s001 + 1.39058 w_k1_MAT_s001
                                 + 0.989491 w_k1_MSFT_s001
                                 + 1.23206 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.01691 w_k1_ADM_s001
                                 + 1.39939 w_k1_AXP_s001 + 1.1492 w_k1_DIS_s001
                                 + 1.27287 w_k1_FDX_s001 + 1.0358 w_k1_IBM_s001
                                 + 1.36287 w_k1_INTC_s001
                                 + 1.05492 w_k1_KO_s001 + 1.17142 w_k1_MAT_s001
                                 + 1.26908 w_k1_MSFT_s001
                                 + 1.18274 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.717142 w_k1_ADM_s001
                                 + 0.757331 w_k1_AXP_s001
                                 + 0.773541 w_k1_DIS_s001
                                 + 0.856454 w_k1_FDX_s001
                                 + 0.719237 w_k1_IBM_s001
                                 + 0.776917 w_k1_INTC_s001
                                 + 0.709484 w_k1_KO_s001
                                 + 0.808034 w_k1_MAT_s001
                                 + 0.829642 w_k1_MSFT_s001
                                 + 0.860674 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.1344 w_k1_ADM_s001
                                 + 1.00039 w_k1_AXP_s001
                                 + 1.64428 w_k1_DIS_s001
                                 + 1.26773 w_k1_FDX_s001
                                 + 1.05257 w_k1_IBM_s001
                                 + 1.10822 w_k1_INTC_s001
                                 + 1.28602 w_k1_KO_s001
                                 + 0.904891 w_k1_MAT_s001
                                 + 1.38539 w_k1_MSFT_s001
                                 + 1.36496 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.15075 w_k1_ADM_s002
                                 + 0.968875 w_k1_AXP_s002
                                 + 1.08319 w_k1_DIS_s002
                                 + 1.10042 w_k1_FDX_s002
                                 + 0.576993 w_k1_IBM_s002
                                 + 1.12704 w_k1_INTC_s002
                                 + 0.915745 w_k1_KO_s002
                                 + 0.463419 w_k1_MAT_s002
                                 + 1.16674 w_k1_MSFT_s002
                                 + 1.01958 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.966355 w_k1_ADM_s002
                                 + 0.819754 w_k1_AXP_s002
                                 + 0.711168 w_k1_DIS_s002
                                 + 0.825202 w_k1_FDX_s002
                                 + 1.18503 w_k1_IBM_s002
                                 + 0.820802 w_k1_INTC_s002
                                 + 1.08912 w_k1_KO_s002 + 1.01243 w_k1_MAT_s002
                                 + 0.804565 w_k1_MSFT_s002
                                 + 1.02656 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.856709 w_k1_ADM_s002
                                 + 0.761144 w_k1_AXP_s002
                                 + 1.20008 w_k1_DIS_s002
                                 + 0.945212 w_k1_FDX_s002
                                 + 0.962931 w_k1_IBM_s002
                                 + 0.999469 w_k1_INTC_s002
                                 + 1.1148 w_k1_KO_s002 + 0.766587 w_k1_MAT_s002
                                 + 1.07349 w_k1_MSFT_s002
                                 + 0.986727 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.32294 w_k1_ADM_s002
                                 + 1.1981 w_k1_AXP_s002
                                 + 0.912787 w_k1_DIS_s002
                                 + 1.01887 w_k1_FDX_s002
                                 + 1.05554 w_k1_IBM_s002
                                 + 0.904555 w_k1_INTC_s002
                                 + 0.971815 w_k1_KO_s002
                                 + 1.04087 w_k1_MAT_s002
                                 + 0.997319 w_k1_MSFT_s002
                                 + 1.17162 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.00763 w_k1_ADM_s002
                                 + 0.754823 w_k1_AXP_s002
                                 + 0.607969 w_k1_DIS_s002
                                 + 0.795037 w_k1_FDX_s002
                                 + 0.73385 w_k1_IBM_s002
                                 + 0.898255 w_k1_INTC_s002
                                 + 0.916719 w_k1_KO_s002
                                 + 0.702754 w_k1_MAT_s002
                                 + 0.804519 w_k1_MSFT_s002
                                 + 0.971693 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.18513 w_k1_ADM_s002
                                 + 1.47195 w_k1_AXP_s002
                                 + 1.21423 w_k1_DIS_s002 + 0.9766 w_k1_FDX_s002
                                 + 1.63794 w_k1_IBM_s002
                                 + 1.02467 w_k1_INTC_s002
                                 + 1.55589 w_k1_KO_s002 + 1.8472 w_k1_MAT_s002
                                 + 1.10728 w_k1_MSFT_s002
                                 + 1.16574 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.15088 w_k1_ADM_s002
                                 + 0.951248 w_k1_AXP_s002
                                 + 1.35594 w_k1_DIS_s002
                                 + 1.12245 w_k1_FDX_s002
                                 + 1.29718 w_k1_IBM_s002
                                 + 0.96423 w_k1_INTC_s002
                                 + 1.55973 w_k1_KO_s002 + 1.20304 w_k1_MAT_s002
                                 + 1.15778 w_k1_MSFT_s002
                                 + 1.25854 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.29164 w_k1_ADM_s002
                                 + 1.59588 w_k1_AXP_s002
                                 + 1.32914 w_k1_DIS_s002
                                 + 1.14076 w_k1_FDX_s002
                                 + 1.01631 w_k1_IBM_s002
                                 + 1.00075 w_k1_INTC_s002
                                 + 1.05832 w_k1_KO_s002
                                 + 0.908221 w_k1_MAT_s002
                                 + 0.942011 w_k1_MSFT_s002
                                 + 1.15448 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.09496 w_k1_ADM_s003
                                 + 1.32278 w_k1_AXP_s003
                                 + 1.38076 w_k1_DIS_s003
                                 + 1.21858 w_k1_FDX_s003
                                 + 0.876811 w_k1_IBM_s003
                                 + 1.10917 w_k1_INTC_s003
                                 + 1.04762 w_k1_KO_s003 + 1.10835 w_k1_MAT_s003
                                 + 1.23411 w_k1_MSFT_s003
                                 + 1.07164 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.94499 w_k1_ADM_s003
                                 + 0.910751 w_k1_AXP_s003
                                 + 0.893427 w_k1_DIS_s003
                                 + 0.839002 w_k1_FDX_s003
                                 + 1.15403 w_k1_IBM_s003
                                 + 0.853458 w_k1_INTC_s003
                                 + 0.919461 w_k1_KO_s003 + 1.2732 w_k1_MAT_s003
                                 + 0.906383 w_k1_MSFT_s003
                                 + 0.951291 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.792609 w_k1_ADM_s003
                                 + 0.859894 w_k1_AXP_s003
                                 + 1.01355 w_k1_DIS_s003
                                 + 0.955633 w_k1_FDX_s003
                                 + 0.747892 w_k1_IBM_s003
                                 + 1.0258 w_k1_INTC_s003
                                 + 0.826884 w_k1_KO_s003
                                 + 0.866988 w_k1_MAT_s003
                                 + 1.18448 w_k1_MSFT_s003
                                 + 0.974704 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.782625 w_k1_ADM_s003
                                 + 0.649524 w_k1_AXP_s003
                                 + 0.679255 w_k1_DIS_s003
                                 + 0.662072 w_k1_FDX_s003
                                 + 0.955967 w_k1_IBM_s003
                                 + 0.686368 w_k1_INTC_s003
                                 + 0.807379 w_k1_KO_s003
                                 + 0.909257 w_k1_MAT_s003
                                 + 0.811055 w_k1_MSFT_s003
                                 + 0.892514 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.887971 w_k1_ADM_s003
                                 + 1.09537 w_k1_AXP_s003
                                 + 1.04746 w_k1_DIS_s003
                                 + 0.959643 w_k1_FDX_s003
                                 + 0.942402 w_k1_IBM_s003
                                 + 0.873706 w_k1_INTC_s003
                                 + 0.97024 w_k1_KO_s003 + 1.33202 w_k1_MAT_s003
                                 + 1.12495 w_k1_MSFT_s003
                                 + 1.03976 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.822948 w_k1_ADM_s003
                                 + 1.11337 w_k1_AXP_s003
                                 + 1.22107 w_k1_DIS_s003
                                 + 1.04203 w_k1_FDX_s003
                                 + 1.42652 w_k1_IBM_s003
                                 + 1.11574 w_k1_INTC_s003
                                 + 1.27405 w_k1_KO_s003 + 1.81593 w_k1_MAT_s003
                                 + 1.15313 w_k1_MSFT_s003
                                 + 1.24655 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.04646 w_k1_ADM_s003
                                 + 0.714246 w_k1_AXP_s003
                                 + 0.74849 w_k1_DIS_s003
                                 + 0.894166 w_k1_FDX_s003
                                 + 0.682899 w_k1_IBM_s003
                                 + 1.11422 w_k1_INTC_s003
                                 + 0.71102 w_k1_KO_s003
                                 + 0.488082 w_k1_MAT_s003
                                 + 0.957222 w_k1_MSFT_s003
                                 + 0.86989 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.22477 w_k1_ADM_s003
                                 + 1.23167 w_k1_AXP_s003
                                 + 2.07155 w_k1_DIS_s003
                                 + 1.40626 w_k1_FDX_s003
                                 + 1.36902 w_k1_IBM_s003
                                 + 1.03854 w_k1_INTC_s003 + 1.4676 w_k1_KO_s003
                                 + 1.63122 w_k1_MAT_s003
                                 + 1.37365 w_k1_MSFT_s003
                                 + 1.26154 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.817684 w_k1_ADM_s004
                                 + 1.11046 w_k1_AXP_s004
                                 + 1.20297 w_k1_DIS_s004
                                 + 0.959786 w_k1_FDX_s004
                                 + 1.52924 w_k1_IBM_s004
                                 + 1.11352 w_k1_INTC_s004
                                 + 1.20479 w_k1_KO_s004 + 1.40997 w_k1_MAT_s004
                                 + 1.08001 w_k1_MSFT_s004
                                 + 0.993515 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.00076 w_k1_ADM_s004
                                 + 0.986779 w_k1_AXP_s004
                                 + 0.828932 w_k1_DIS_s004
                                 + 0.889925 w_k1_FDX_s004
                                 + 1.13796 w_k1_IBM_s004
                                 + 0.741569 w_k1_INTC_s004
                                 + 0.857397 w_k1_KO_s004
                                 + 1.05212 w_k1_MAT_s004
                                 + 0.810944 w_k1_MSFT_s004
                                 + 0.825427 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.04401 w_k1_ADM_s004
                                 + 1.14966 w_k1_AXP_s004
                                 + 1.17871 w_k1_DIS_s004
                                 + 1.20931 w_k1_FDX_s004
                                 + 0.872123 w_k1_IBM_s004
                                 + 0.719508 w_k1_INTC_s004
                                 + 0.931342 w_k1_KO_s004
                                 + 1.12341 w_k1_MAT_s004
                                 + 1.04569 w_k1_MSFT_s004
                                 + 1.04923 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.630029 w_k1_ADM_s004
                                 + 0.522913 w_k1_AXP_s004
                                 + 0.744969 w_k1_DIS_s004
                                 + 0.694327 w_k1_FDX_s004
                                 + 0.908304 w_k1_IBM_s004
                                 + 0.720292 w_k1_INTC_s004
                                 + 0.970112 w_k1_KO_s004
                                 + 0.906779 w_k1_MAT_s004
                                 + 0.794887 w_k1_MSFT_s004
                                 + 0.811748 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.890948 w_k1_ADM_s004
                                 + 0.77832 w_k1_AXP_s004
                                 + 0.819332 w_k1_DIS_s004
                                 + 0.83891 w_k1_FDX_s004
                                 + 0.857582 w_k1_IBM_s004
                                 + 0.781785 w_k1_INTC_s004
                                 + 0.909392 w_k1_KO_s004
                                 + 0.955659 w_k1_MAT_s004
                                 + 0.817433 w_k1_MSFT_s004
                                 + 0.857726 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.907421 w_k1_ADM_s004
                                 + 0.812509 w_k1_AXP_s004
                                 + 1.07622 w_k1_DIS_s004
                                 + 0.965591 w_k1_FDX_s004
                                 + 0.741831 w_k1_IBM_s004
                                 + 0.958497 w_k1_INTC_s004
                                 + 1.0649 w_k1_KO_s004 + 0.718751 w_k1_MAT_s004
                                 + 1.07304 w_k1_MSFT_s004
                                 + 0.888973 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.753519 w_k1_ADM_s004
                                 + 0.752355 w_k1_AXP_s004
                                 + 0.673778 w_k1_DIS_s004
                                 + 0.823839 w_k1_FDX_s004
                                 + 0.660402 w_k1_IBM_s004
                                 + 0.797757 w_k1_INTC_s004
                                 + 0.729865 w_k1_KO_s004
                                 + 0.560091 w_k1_MAT_s004
                                 + 0.839188 w_k1_MSFT_s004
                                 + 0.708767 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.05121 w_k1_ADM_s004
                                 + 1.27715 w_k1_AXP_s004
                                 + 1.67248 w_k1_DIS_s004
                                 + 1.25077 w_k1_FDX_s004
                                 + 1.20711 w_k1_IBM_s004
                                 + 0.929538 w_k1_INTC_s004
                                 + 1.23539 w_k1_KO_s004 + 1.52576 w_k1_MAT_s004
                                 + 1.34845 w_k1_MSFT_s004
                                 + 1.04458 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.873779 w_k1_ADM_s005
                                 + 1.04156 w_k1_AXP_s005
                                 + 0.976101 w_k1_DIS_s005
                                 + 1.0201 w_k1_FDX_s005 + 1.17914 w_k1_IBM_s005
                                 + 1.45952 w_k1_INTC_s005
                                 + 0.980026 w_k1_KO_s005
                                 + 0.975273 w_k1_MAT_s005
                                 + 1.01344 w_k1_MSFT_s005
                                 + 1.02119 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.14112 w_k1_ADM_s005
                                 + 1.19715 w_k1_AXP_s005
                                 + 1.60344 w_k1_DIS_s005
                                 + 1.28158 w_k1_FDX_s005
                                 + 1.20064 w_k1_IBM_s005
                                 + 1.33728 w_k1_INTC_s005
                                 + 1.24112 w_k1_KO_s005 + 0.82819 w_k1_MAT_s005
                                 + 1.26196 w_k1_MSFT_s005
                                 + 1.14589 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.923045 w_k1_ADM_s005
                                 + 0.736857 w_k1_AXP_s005
                                 + 0.695604 w_k1_DIS_s005
                                 + 0.752738 w_k1_FDX_s005
                                 + 1.05405 w_k1_IBM_s005
                                 + 0.826302 w_k1_INTC_s005
                                 + 0.860344 w_k1_KO_s005
                                 + 0.897153 w_k1_MAT_s005
                                 + 0.829708 w_k1_MSFT_s005
                                 + 0.845733 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.23175 w_k1_ADM_s005
                                 + 1.21966 w_k1_AXP_s005
                                 + 1.14773 w_k1_DIS_s005
                                 + 1.11041 w_k1_FDX_s005
                                 + 1.08867 w_k1_IBM_s005
                                 + 1.23806 w_k1_INTC_s005
                                 + 1.27222 w_k1_KO_s005 + 1.50905 w_k1_MAT_s005
                                 + 1.16154 w_k1_MSFT_s005 + 1.162 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.37658 w_k1_ADM_s005
                                 + 1.66124 w_k1_AXP_s005
                                 + 1.24409 w_k1_DIS_s005 + 1.2464 w_k1_FDX_s005
                                 + 1.34776 w_k1_IBM_s005
                                 + 1.35996 w_k1_INTC_s005
                                 + 1.14074 w_k1_KO_s005 + 1.37226 w_k1_MAT_s005
                                 + 1.10266 w_k1_MSFT_s005
                                 + 1.03334 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.44654 w_k1_ADM_s005
                                 + 1.09679 w_k1_AXP_s005
                                 + 1.19259 w_k1_DIS_s005
                                 + 0.923991 w_k1_FDX_s005
                                 + 1.58152 w_k1_IBM_s005
                                 + 1.14125 w_k1_INTC_s005
                                 + 1.27756 w_k1_KO_s005
                                 + 0.842116 w_k1_MAT_s005
                                 + 0.956165 w_k1_MSFT_s005
                                 + 1.05831 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.16803 w_k1_ADM_s005
                                 + 0.940394 w_k1_AXP_s005
                                 + 1.08947 w_k1_DIS_s005
                                 + 0.938343 w_k1_FDX_s005
                                 + 0.780671 w_k1_IBM_s005
                                 + 1.16892 w_k1_INTC_s005
                                 + 0.946912 w_k1_KO_s005
                                 + 0.923411 w_k1_MAT_s005
                                 + 0.968874 w_k1_MSFT_s005
                                 + 0.984886 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.993756 w_k1_ADM_s005
                                 + 0.870893 w_k1_AXP_s005
                                 + 0.822304 w_k1_DIS_s005
                                 + 0.972293 w_k1_FDX_s005
                                 + 0.758082 w_k1_IBM_s005
                                 + 1.23565 w_k1_INTC_s005
                                 + 0.736823 w_k1_KO_s005
                                 + 0.896938 w_k1_MAT_s005
                                 + 0.997411 w_k1_MSFT_s005
                                 + 0.945825 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.0868 w_k1_ADM_s006
                                 + 1.89685 w_k1_AXP_s006
                                 + 1.82871 w_k1_DIS_s006
                                 + 1.47601 w_k1_FDX_s006
                                 + 1.66789 w_k1_IBM_s006
                                 + 1.69434 w_k1_INTC_s006
                                 + 1.56704 w_k1_KO_s006 + 1.52479 w_k1_MAT_s006
                                 + 1.24248 w_k1_MSFT_s006
                                 + 1.46723 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.761982 w_k1_ADM_s006
                                 + 0.751885 w_k1_AXP_s006
                                 + 0.644975 w_k1_DIS_s006
                                 + 0.762051 w_k1_FDX_s006
                                 + 1.06776 w_k1_IBM_s006
                                 + 0.873311 w_k1_INTC_s006
                                 + 0.673893 w_k1_KO_s006
                                 + 0.692399 w_k1_MAT_s006
                                 + 0.574231 w_k1_MSFT_s006
                                 + 0.779826 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.87241 w_k1_ADM_s006
                                 + 0.808263 w_k1_AXP_s006
                                 + 0.865596 w_k1_DIS_s006
                                 + 0.954613 w_k1_FDX_s006
                                 + 0.790833 w_k1_IBM_s006
                                 + 0.881361 w_k1_INTC_s006
                                 + 0.830974 w_k1_KO_s006
                                 + 0.645421 w_k1_MAT_s006
                                 + 0.904395 w_k1_MSFT_s006
                                 + 1.07138 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.860956 w_k1_ADM_s006
                                 + 1.37695 w_k1_AXP_s006 + 1.0729 w_k1_DIS_s006
                                 + 1.11278 w_k1_FDX_s006
                                 + 1.55519 w_k1_IBM_s006
                                 + 1.10121 w_k1_INTC_s006
                                 + 1.29509 w_k1_KO_s006 + 1.29646 w_k1_MAT_s006
                                 + 0.886389 w_k1_MSFT_s006
                                 + 1.23871 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.2902 w_k1_ADM_s006
                                 + 1.15304 w_k1_AXP_s006
                                 + 1.08602 w_k1_DIS_s006 + 1.1432 w_k1_FDX_s006
                                 + 0.823931 w_k1_IBM_s006
                                 + 1.13354 w_k1_INTC_s006
                                 + 0.973972 w_k1_KO_s006
                                 + 0.871621 w_k1_MAT_s006 + 0.99 w_k1_MSFT_s006
                                 + 1.05131 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.01155 w_k1_ADM_s006
                                 + 1.04517 w_k1_AXP_s006
                                 + 0.784888 w_k1_DIS_s006
                                 + 0.909249 w_k1_FDX_s006
                                 + 0.85426 w_k1_IBM_s006
                                 + 0.897786 w_k1_INTC_s006
                                 + 0.859278 w_k1_KO_s006
                                 + 1.00376 w_k1_MAT_s006
                                 + 0.810196 w_k1_MSFT_s006
                                 + 1.08149 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.05734 w_k1_ADM_s006
                                 + 1.00564 w_k1_AXP_s006
                                 + 0.784578 w_k1_DIS_s006
                                 + 0.858882 w_k1_FDX_s006
                                 + 1.34168 w_k1_IBM_s006
                                 + 0.772329 w_k1_INTC_s006
                                 + 1.0772 w_k1_KO_s006 + 1.31268 w_k1_MAT_s006
                                 + 0.765642 w_k1_MSFT_s006
                                 + 1.07388 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.23492 w_k1_ADM_s006
                                 + 1.3591 w_k1_AXP_s006 + 1.56855 w_k1_DIS_s006
                                 + 1.43433 w_k1_FDX_s006
                                 + 0.963386 w_k1_IBM_s006
                                 + 1.346 w_k1_INTC_s006 + 1.23166 w_k1_KO_s006
                                 + 0.950359 w_k1_MAT_s006
                                 + 1.18958 w_k1_MSFT_s006
                                 + 1.34363 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.05569 w_k1_ADM_s007
                                 + 0.735076 w_k1_AXP_s007
                                 + 0.928216 w_k1_DIS_s007
                                 + 0.981041 w_k1_FDX_s007
                                 + 0.655151 w_k1_IBM_s007
                                 + 1.04153 w_k1_INTC_s007
                                 + 0.731709 w_k1_KO_s007
                                 + 0.532438 w_k1_MAT_s007
                                 + 1.11452 w_k1_MSFT_s007
                                 + 0.924201 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.12965 w_k1_ADM_s007
                                 + 1.12112 w_k1_AXP_s007 + 1.0339 w_k1_DIS_s007
                                 + 0.953245 w_k1_FDX_s007
                                 + 1.08485 w_k1_IBM_s007
                                 + 0.812105 w_k1_INTC_s007
                                 + 0.929277 w_k1_KO_s007
                                 + 1.12878 w_k1_MAT_s007
                                 + 0.910046 w_k1_MSFT_s007
                                 + 0.887909 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.36457 w_k1_ADM_s007
                                 + 1.51171 w_k1_AXP_s007
                                 + 1.81403 w_k1_DIS_s007
                                 + 1.33611 w_k1_FDX_s007 + 1.3871 w_k1_IBM_s007
                                 + 1.09891 w_k1_INTC_s007
                                 + 1.26864 w_k1_KO_s007 + 1.08017 w_k1_MAT_s007
                                 + 1.47424 w_k1_MSFT_s007
                                 + 1.04519 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.919621 w_k1_ADM_s007
                                 + 0.602618 w_k1_AXP_s007
                                 + 0.613968 w_k1_DIS_s007
                                 + 0.81733 w_k1_FDX_s007
                                 + 0.483071 w_k1_IBM_s007
                                 + 0.797675 w_k1_INTC_s007
                                 + 0.569857 w_k1_KO_s007
                                 + 0.355822 w_k1_MAT_s007
                                 + 0.94254 w_k1_MSFT_s007
                                 + 0.725661 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.882089 w_k1_ADM_s007
                                 + 0.558898 w_k1_AXP_s007
                                 + 0.867768 w_k1_DIS_s007
                                 + 0.875304 w_k1_FDX_s007
                                 + 1.01511 w_k1_IBM_s007
                                 + 0.846863 w_k1_INTC_s007
                                 + 0.912432 w_k1_KO_s007
                                 + 0.704308 w_k1_MAT_s007
                                 + 1.01979 w_k1_MSFT_s007
                                 + 0.805347 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.18753 w_k1_ADM_s007
                                 + 1.09415 w_k1_AXP_s007
                                 + 1.25667 w_k1_DIS_s007
                                 + 1.11811 w_k1_FDX_s007
                                 + 1.30914 w_k1_IBM_s007
                                 + 0.918432 w_k1_INTC_s007
                                 + 1.08371 w_k1_KO_s007 + 1.382 w_k1_MAT_s007
                                 + 1.14532 w_k1_MSFT_s007
                                 + 0.953706 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.31671 w_k1_ADM_s007
                                 + 1.08074 w_k1_AXP_s007
                                 + 1.04286 w_k1_DIS_s007
                                 + 1.19737 w_k1_FDX_s007
                                 + 0.928591 w_k1_IBM_s007
                                 + 0.866841 w_k1_INTC_s007
                                 + 0.86177 w_k1_KO_s007
                                 + 0.735382 w_k1_MAT_s007
                                 + 1.03393 w_k1_MSFT_s007
                                 + 1.01092 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.681498 w_k1_ADM_s007
                                 + 0.884919 w_k1_AXP_s007
                                 + 0.666265 w_k1_DIS_s007
                                 + 0.749879 w_k1_FDX_s007
                                 + 0.88065 w_k1_IBM_s007
                                 + 0.787788 w_k1_INTC_s007
                                 + 0.701373 w_k1_KO_s007
                                 + 0.844039 w_k1_MAT_s007
                                 + 0.895534 w_k1_MSFT_s007
                                 + 0.901809 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.00924 w_k1_ADM_s010
                                 + 1.03139 w_k1_AXP_s010
                                 + 1.04846 w_k1_DIS_s010
                                 + 1.14646 w_k1_FDX_s010
                                 + 0.821166 w_k1_IBM_s010
                                 + 1.05063 w_k1_INTC_s010
                                 + 0.87549 w_k1_KO_s010
                                 + 0.816434 w_k1_MAT_s010
                                 + 1.12361 w_k1_MSFT_s010
                                 + 0.943828 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.06607 w_k1_ADM_s010
                                 + 1.1488 w_k1_AXP_s010 + 1.35348 w_k1_DIS_s010
                                 + 1.17346 w_k1_FDX_s010 + 1.5475 w_k1_IBM_s010
                                 + 1.31362 w_k1_INTC_s010
                                 + 1.37854 w_k1_KO_s010 + 1.28509 w_k1_MAT_s010
                                 + 1.1441 w_k1_MSFT_s010 + 1.09487 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.798226 w_k1_ADM_s010
                                 + 0.88036 w_k1_AXP_s010
                                 + 1.01621 w_k1_DIS_s010
                                 + 0.971575 w_k1_FDX_s010
                                 + 0.90038 w_k1_IBM_s010
                                 + 1.33978 w_k1_INTC_s010
                                 + 1.00567 w_k1_KO_s010
                                 + 0.620748 w_k1_MAT_s010
                                 + 1.04516 w_k1_MSFT_s010
                                 + 0.956603 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.990684 w_k1_ADM_s010
                                 + 1.49074 w_k1_AXP_s010 + 1.3775 w_k1_DIS_s010
                                 + 1.19936 w_k1_FDX_s010
                                 + 1.22774 w_k1_IBM_s010
                                 + 1.18387 w_k1_INTC_s010
                                 + 1.23563 w_k1_KO_s010 + 1.20148 w_k1_MAT_s010
                                 + 1.16353 w_k1_MSFT_s010
                                 + 1.11649 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.786272 w_k1_ADM_s010
                                 + 0.916245 w_k1_AXP_s010
                                 + 1.04425 w_k1_DIS_s010
                                 + 0.933583 w_k1_FDX_s010
                                 + 1.31871 w_k1_IBM_s010
                                 + 1.17414 w_k1_INTC_s010
                                 + 1.16437 w_k1_KO_s010 + 1.29934 w_k1_MAT_s010
                                 + 1.03768 w_k1_MSFT_s010
                                 + 0.901751 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.805065 w_k1_ADM_s010
                                 + 0.966134 w_k1_AXP_s010
                                 + 0.829356 w_k1_DIS_s010
                                 + 0.887152 w_k1_FDX_s010
                                 + 0.802423 w_k1_IBM_s010
                                 + 1.08877 w_k1_INTC_s010
                                 + 0.920978 w_k1_KO_s010
                                 + 0.659131 w_k1_MAT_s010
                                 + 0.930262 w_k1_MSFT_s010
                                 + 0.861415 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.986262 w_k1_ADM_s010
                                 + 0.868066 w_k1_AXP_s010
                                 + 0.823416 w_k1_DIS_s010
                                 + 0.981619 w_k1_FDX_s010
                                 + 0.980704 w_k1_IBM_s010
                                 + 1.35907 w_k1_INTC_s010 + 1.025 w_k1_KO_s010
                                 + 0.587179 w_k1_MAT_s010
                                 + 0.969373 w_k1_MSFT_s010
                                 + 0.87477 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.780626 w_k1_ADM_s010
                                 + 0.897725 w_k1_AXP_s010
                                 + 0.636924 w_k1_DIS_s010
                                 + 0.759652 w_k1_FDX_s010
                                 + 1.02034 w_k1_IBM_s010
                                 + 0.880126 w_k1_INTC_s010
                                 + 0.85255 w_k1_KO_s010
                                 + 0.725767 w_k1_MAT_s010
                                 + 0.774911 w_k1_MSFT_s010
                                 + 0.804916 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.77982 w_k1_ADM_s011
                                 + 0.855582 w_k1_AXP_s011
                                 + 0.912899 w_k1_DIS_s011
                                 + 0.823848 w_k1_FDX_s011
                                 + 0.858862 w_k1_IBM_s011
                                 + 0.971394 w_k1_INTC_s011
                                 + 0.90791 w_k1_KO_s011 + 0.88176 w_k1_MAT_s011
                                 + 0.860004 w_k1_MSFT_s011
                                 + 0.967148 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.607148 w_k1_ADM_s011
                                 + 0.827512 w_k1_AXP_s011
                                 + 0.6416 w_k1_DIS_s011
                                 + 0.719725 w_k1_FDX_s011
                                 + 1.05992 w_k1_IBM_s011
                                 + 0.880539 w_k1_INTC_s011
                                 + 0.989556 w_k1_KO_s011
                                 + 1.05313 w_k1_MAT_s011
                                 + 0.790662 w_k1_MSFT_s011
                                 + 0.92311 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.959647 w_k1_ADM_s011
                                 + 1.01245 w_k1_AXP_s011
                                 + 1.11693 w_k1_DIS_s011
                                 + 1.08792 w_k1_FDX_s011
                                 + 0.739406 w_k1_IBM_s011
                                 + 0.908387 w_k1_INTC_s011
                                 + 0.850782 w_k1_KO_s011
                                 + 0.848806 w_k1_MAT_s011
                                 + 1.20831 w_k1_MSFT_s011
                                 + 1.03941 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.00259 w_k1_ADM_s011
                                 + 1.34396 w_k1_AXP_s011
                                 + 1.12729 w_k1_DIS_s011
                                 + 1.17424 w_k1_FDX_s011
                                 + 0.949527 w_k1_IBM_s011
                                 + 1.09386 w_k1_INTC_s011
                                 + 1.10109 w_k1_KO_s011 + 1.12586 w_k1_MAT_s011
                                 + 1.12667 w_k1_MSFT_s011
                                 + 1.09048 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.698869 w_k1_ADM_s011
                                 + 0.745327 w_k1_AXP_s011
                                 + 0.547284 w_k1_DIS_s011
                                 + 0.728217 w_k1_FDX_s011
                                 + 0.794784 w_k1_IBM_s011
                                 + 0.776855 w_k1_INTC_s011
                                 + 0.690047 w_k1_KO_s011
                                 + 0.874732 w_k1_MAT_s011
                                 + 0.671381 w_k1_MSFT_s011
                                 + 0.839945 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.13867 w_k1_ADM_s011
                                 + 1.68935 w_k1_AXP_s011
                                 + 1.76116 w_k1_DIS_s011
                                 + 1.43438 w_k1_FDX_s011
                                 + 1.24195 w_k1_IBM_s011
                                 + 1.15212 w_k1_INTC_s011
                                 + 1.28515 w_k1_KO_s011 + 1.82555 w_k1_MAT_s011
                                 + 1.41886 w_k1_MSFT_s011
                                 + 1.40852 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.958382 w_k1_ADM_s011
                                 + 1.11012 w_k1_AXP_s011
                                 + 1.07889 w_k1_DIS_s011
                                 + 0.929013 w_k1_FDX_s011
                                 + 1.45378 w_k1_IBM_s011
                                 + 0.910629 w_k1_INTC_s011
                                 + 1.17204 w_k1_KO_s011 + 1.46289 w_k1_MAT_s011
                                 + 0.940969 w_k1_MSFT_s011
                                 + 1.19894 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.681807 w_k1_ADM_s011
                                 + 0.71059 w_k1_AXP_s011
                                 + 0.641903 w_k1_DIS_s011
                                 + 0.725173 w_k1_FDX_s011
                                 + 0.583222 w_k1_IBM_s011
                                 + 0.982991 w_k1_INTC_s011
                                 + 0.707053 w_k1_KO_s011
                                 + 0.397743 w_k1_MAT_s011
                                 + 0.874418 w_k1_MSFT_s011
                                 + 0.859644 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.09034 w_k1_ADM_s012
                                 + 1.14128 w_k1_AXP_s012
                                 + 0.990863 w_k1_DIS_s012
                                 + 1.05718 w_k1_FDX_s012
                                 + 1.15272 w_k1_IBM_s012
                                 + 0.900837 w_k1_INTC_s012
                                 + 1.03744 w_k1_KO_s012 + 1.75541 w_k1_MAT_s012
                                 + 1.01233 w_k1_MSFT_s012
                                 + 0.994307 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.753028 w_k1_ADM_s012
                                 + 0.821795 w_k1_AXP_s012
                                 + 0.723363 w_k1_DIS_s012
                                 + 0.904612 w_k1_FDX_s012
                                 + 0.720159 w_k1_IBM_s012
                                 + 0.795834 w_k1_INTC_s012
                                 + 0.709592 w_k1_KO_s012
                                 + 0.612873 w_k1_MAT_s012
                                 + 0.821694 w_k1_MSFT_s012
                                 + 0.851374 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.37174 w_k1_ADM_s012
                                 + 1.55477 w_k1_AXP_s012
                                 + 1.61827 w_k1_DIS_s012
                                 + 1.47327 w_k1_FDX_s012
                                 + 1.24072 w_k1_IBM_s012 + 1.439 w_k1_INTC_s012
                                 + 1.4134 w_k1_KO_s012 + 1.24481 w_k1_MAT_s012
                                 + 1.36599 w_k1_MSFT_s012
                                 + 1.34126 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.895867 w_k1_ADM_s012
                                 + 0.830985 w_k1_AXP_s012
                                 + 0.834465 w_k1_DIS_s012
                                 + 0.814142 w_k1_FDX_s012
                                 + 1.13652 w_k1_IBM_s012
                                 + 0.792888 w_k1_INTC_s012
                                 + 0.948485 w_k1_KO_s012
                                 + 1.00918 w_k1_MAT_s012
                                 + 0.769418 w_k1_MSFT_s012
                                 + 0.924825 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.996557 w_k1_ADM_s012
                                 + 0.990232 w_k1_AXP_s012
                                 + 1.07561 w_k1_DIS_s012
                                 + 1.07551 w_k1_FDX_s012
                                 + 0.860482 w_k1_IBM_s012
                                 + 0.905871 w_k1_INTC_s012
                                 + 0.95179 w_k1_KO_s012 + 1.03448 w_k1_MAT_s012
                                 + 0.975751 w_k1_MSFT_s012
                                 + 1.09586 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.776649 w_k1_ADM_s012
                                 + 0.734319 w_k1_AXP_s012
                                 + 0.735555 w_k1_DIS_s012
                                 + 0.765111 w_k1_FDX_s012
                                 + 0.833877 w_k1_IBM_s012
                                 + 0.76938 w_k1_INTC_s012
                                 + 0.911721 w_k1_KO_s012
                                 + 0.986256 w_k1_MAT_s012
                                 + 0.825274 w_k1_MSFT_s012
                                 + 0.908866 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.01093 w_k1_ADM_s012
                                 + 1.113 w_k1_AXP_s012 + 1.46535 w_k1_DIS_s012
                                 + 1.11398 w_k1_FDX_s012 + 1.4089 w_k1_IBM_s012
                                 + 0.900592 w_k1_INTC_s012
                                 + 1.3779 w_k1_KO_s012 + 1.66947 w_k1_MAT_s012
                                 + 1.14511 w_k1_MSFT_s012
                                 + 1.21509 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.24235 w_k1_ADM_s012
                                 + 1.39359 w_k1_AXP_s012
                                 + 1.16573 w_k1_DIS_s012
                                 + 1.20678 w_k1_FDX_s012
                                 + 1.26162 w_k1_IBM_s012
                                 + 1.0511 w_k1_INTC_s012 + 1.04245 w_k1_KO_s012
                                 + 1.42506 w_k1_MAT_s012
                                 + 1.07894 w_k1_MSFT_s012
                                 + 1.17954 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.849456 w_k1_ADM_s013
                                 + 0.730062 w_k1_AXP_s013
                                 + 0.849284 w_k1_DIS_s013
                                 + 0.946601 w_k1_FDX_s013
                                 + 0.774083 w_k1_IBM_s013
                                 + 0.900487 w_k1_INTC_s013
                                 + 0.906838 w_k1_KO_s013
                                 + 0.594549 w_k1_MAT_s013
                                 + 0.914878 w_k1_MSFT_s013
                                 + 0.941722 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.22623 w_k1_ADM_s013
                                 + 1.25446 w_k1_AXP_s013
                                 + 1.42398 w_k1_DIS_s013
                                 + 1.26787 w_k1_FDX_s013
                                 + 1.20884 w_k1_IBM_s013
                                 + 1.27808 w_k1_INTC_s013
                                 + 1.30668 w_k1_KO_s013 + 1.35356 w_k1_MAT_s013
                                 + 1.1957 w_k1_MSFT_s013 + 1.27126 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.906436 w_k1_ADM_s013
                                 + 0.770367 w_k1_AXP_s013
                                 + 0.519257 w_k1_DIS_s013
                                 + 0.784484 w_k1_FDX_s013
                                 + 0.741084 w_k1_IBM_s013
                                 + 0.831198 w_k1_INTC_s013
                                 + 0.751971 w_k1_KO_s013
                                 + 0.706432 w_k1_MAT_s013
                                 + 0.78496 w_k1_MSFT_s013
                                 + 0.880714 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.18073 w_k1_ADM_s013
                                 + 0.904666 w_k1_AXP_s013
                                 + 0.835012 w_k1_DIS_s013 + 1.055 w_k1_FDX_s013
                                 + 0.604948 w_k1_IBM_s013
                                 + 0.869274 w_k1_INTC_s013
                                 + 0.735665 w_k1_KO_s013
                                 + 0.735739 w_k1_MAT_s013
                                 + 0.897071 w_k1_MSFT_s013
                                 + 0.871489 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.10693 w_k1_ADM_s013
                                 + 1.21831 w_k1_AXP_s013
                                 + 0.834584 w_k1_DIS_s013
                                 + 1.1674 w_k1_FDX_s013
                                 + 0.892937 w_k1_IBM_s013
                                 + 1.29601 w_k1_INTC_s013
                                 + 0.977318 w_k1_KO_s013
                                 + 0.996582 w_k1_MAT_s013
                                 + 1.06815 w_k1_MSFT_s013
                                 + 1.08237 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.89194 w_k1_ADM_s013
                                 + 0.744107 w_k1_AXP_s013
                                 + 0.906105 w_k1_DIS_s013
                                 + 0.879417 w_k1_FDX_s013
                                 + 1.18917 w_k1_IBM_s013
                                 + 0.917479 w_k1_INTC_s013
                                 + 1.05251 w_k1_KO_s013
                                 + 0.894911 w_k1_MAT_s013
                                 + 1.04026 w_k1_MSFT_s013
                                 + 0.92882 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.933993 w_k1_ADM_s013
                                 + 1.01382 w_k1_AXP_s013
                                 + 0.941217 w_k1_DIS_s013
                                 + 0.84535 w_k1_FDX_s013
                                 + 1.55961 w_k1_IBM_s013
                                 + 0.82033 w_k1_INTC_s013
                                 + 1.05004 w_k1_KO_s013 + 1.22316 w_k1_MAT_s013
                                 + 0.874907 w_k1_MSFT_s013
                                 + 1.06531 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.08523 w_k1_ADM_s013
                                 + 0.898047 w_k1_AXP_s013
                                 + 0.900554 w_k1_DIS_s013
                                 + 1.0372 w_k1_FDX_s013 + 1.01759 w_k1_IBM_s013
                                 + 1.08062 w_k1_INTC_s013
                                 + 1.05427 w_k1_KO_s013 + 1.06301 w_k1_MAT_s013
                                 + 1.10692 w_k1_MSFT_s013
                                 + 0.981205 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.03565 w_k1_ADM_s014
                                 + 0.849742 w_k1_AXP_s014
                                 + 0.889178 w_k1_DIS_s014
                                 + 0.926905 w_k1_FDX_s014
                                 + 1.09699 w_k1_IBM_s014
                                 + 0.851282 w_k1_INTC_s014
                                 + 1.02817 w_k1_KO_s014
                                 + 0.943048 w_k1_MAT_s014
                                 + 0.913189 w_k1_MSFT_s014
                                 + 1.02843 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.10011 w_k1_ADM_s014
                                 + 1.0045 w_k1_AXP_s014
                                 + 0.977117 w_k1_DIS_s014
                                 + 1.01834 w_k1_FDX_s014
                                 + 0.754127 w_k1_IBM_s014
                                 + 0.753253 w_k1_INTC_s014
                                 + 0.930155 w_k1_KO_s014
                                 + 0.778945 w_k1_MAT_s014
                                 + 0.995567 w_k1_MSFT_s014
                                 + 1.05857 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.18728 w_k1_ADM_s014
                                 + 1.52942 w_k1_AXP_s014
                                 + 1.67958 w_k1_DIS_s014
                                 + 1.38988 w_k1_FDX_s014
                                 + 1.28969 w_k1_IBM_s014
                                 + 1.39545 w_k1_INTC_s014
                                 + 1.37809 w_k1_KO_s014 + 1.10056 w_k1_MAT_s014
                                 + 1.42099 w_k1_MSFT_s014 + 1.255 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.08884 w_k1_ADM_s014
                                 + 1.55562 w_k1_AXP_s014
                                 + 1.28584 w_k1_DIS_s014
                                 + 1.35669 w_k1_FDX_s014
                                 + 1.17643 w_k1_IBM_s014
                                 + 1.14139 w_k1_INTC_s014
                                 + 1.16234 w_k1_KO_s014 + 1.27249 w_k1_MAT_s014
                                 + 1.24371 w_k1_MSFT_s014
                                 + 1.15568 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.687622 w_k1_ADM_s014
                                 + 0.7566 w_k1_AXP_s014
                                 + 0.527967 w_k1_DIS_s014
                                 + 0.665257 w_k1_FDX_s014
                                 + 0.826084 w_k1_IBM_s014
                                 + 0.723319 w_k1_INTC_s014
                                 + 0.765388 w_k1_KO_s014
                                 + 0.826964 w_k1_MAT_s014
                                 + 0.675009 w_k1_MSFT_s014
                                 + 0.768391 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.805729 w_k1_ADM_s014
                                 + 0.776733 w_k1_AXP_s014
                                 + 0.557749 w_k1_DIS_s014
                                 + 0.803282 w_k1_FDX_s014
                                 + 0.701693 w_k1_IBM_s014
                                 + 0.863186 w_k1_INTC_s014
                                 + 0.674714 w_k1_KO_s014
                                 + 0.656577 w_k1_MAT_s014
                                 + 0.802335 w_k1_MSFT_s014
                                 + 0.868652 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.14992 w_k1_ADM_s014
                                 + 1.29373 w_k1_AXP_s014
                                 + 1.07168 w_k1_DIS_s014
                                 + 1.12064 w_k1_FDX_s014
                                 + 1.37655 w_k1_IBM_s014
                                 + 1.11183 w_k1_INTC_s014
                                 + 1.13818 w_k1_KO_s014 + 1.39533 w_k1_MAT_s014
                                 + 1.10234 w_k1_MSFT_s014
                                 + 1.18321 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.911581 w_k1_ADM_s014
                                 + 1.19081 w_k1_AXP_s014
                                 + 0.730567 w_k1_DIS_s014
                                 + 0.885414 w_k1_FDX_s014
                                 + 0.918243 w_k1_IBM_s014
                                 + 0.974472 w_k1_INTC_s014
                                 + 1.03019 w_k1_KO_s014 + 1.10332 w_k1_MAT_s014
                                 + 0.999531 w_k1_MSFT_s014
                                 + 1.01858 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.889658 w_k1_ADM_s015
                                 + 0.844725 w_k1_AXP_s015
                                 + 0.769924 w_k1_DIS_s015
                                 + 0.91608 w_k1_FDX_s015
                                 + 0.703485 w_k1_IBM_s015
                                 + 0.952086 w_k1_INTC_s015
                                 + 0.822649 w_k1_KO_s015
                                 + 0.491242 w_k1_MAT_s015
                                 + 0.952855 w_k1_MSFT_s015
                                 + 0.792747 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.15384 w_k1_ADM_s015
                                 + 1.23512 w_k1_AXP_s015
                                 + 1.39061 w_k1_DIS_s015
                                 + 1.15157 w_k1_FDX_s015 + 1.3379 w_k1_IBM_s015
                                 + 1.0476 w_k1_INTC_s015 + 1.29256 w_k1_KO_s015
                                 + 1.29943 w_k1_MAT_s015
                                 + 1.28224 w_k1_MSFT_s015
                                 + 1.10339 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.21984 w_k1_ADM_s015
                                 + 1.20227 w_k1_AXP_s015
                                 + 0.970579 w_k1_DIS_s015
                                 + 1.06218 w_k1_FDX_s015
                                 + 0.99304 w_k1_IBM_s015
                                 + 0.913679 w_k1_INTC_s015
                                 + 0.855516 w_k1_KO_s015
                                 + 0.967149 w_k1_MAT_s015
                                 + 1.03165 w_k1_MSFT_s015
                                 + 0.989892 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.978297 w_k1_ADM_s015
                                 + 0.821497 w_k1_AXP_s015
                                 + 0.978101 w_k1_DIS_s015
                                 + 0.915355 w_k1_FDX_s015
                                 + 1.25585 w_k1_IBM_s015
                                 + 0.916485 w_k1_INTC_s015
                                 + 1.16275 w_k1_KO_s015 + 1.11234 w_k1_MAT_s015
                                 + 0.997231 w_k1_MSFT_s015
                                 + 0.897139 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.894189 w_k1_ADM_s015
                                 + 0.587979 w_k1_AXP_s015
                                 + 0.770544 w_k1_DIS_s015
                                 + 0.789812 w_k1_FDX_s015
                                 + 0.706796 w_k1_IBM_s015
                                 + 0.697061 w_k1_INTC_s015
                                 + 0.623144 w_k1_KO_s015
                                 + 0.575431 w_k1_MAT_s015
                                 + 0.845794 w_k1_MSFT_s015
                                 + 0.770265 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.41794 w_k1_ADM_s015
                                 + 1.67611 w_k1_AXP_s015
                                 + 1.75854 w_k1_DIS_s015
                                 + 1.50585 w_k1_FDX_s015
                                 + 1.37941 w_k1_IBM_s015
                                 + 1.18366 w_k1_INTC_s015 + 1.2691 w_k1_KO_s015
                                 + 1.49095 w_k1_MAT_s015
                                 + 1.41453 w_k1_MSFT_s015 + 1.2353 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.854303 w_k1_ADM_s015
                                 + 1.065 w_k1_AXP_s015 + 0.661676 w_k1_DIS_s015
                                 + 0.785252 w_k1_FDX_s015
                                 + 1.21128 w_k1_IBM_s015
                                 + 0.91361 w_k1_INTC_s015
                                 + 0.761806 w_k1_KO_s015
                                 + 1.16952 w_k1_MAT_s015
                                 + 0.82257 w_k1_MSFT_s015
                                 + 0.801069 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.812286 w_k1_ADM_s015
                                 + 0.693811 w_k1_AXP_s015
                                 + 0.582509 w_k1_DIS_s015
                                 + 0.696479 w_k1_FDX_s015
                                 + 0.888722 w_k1_IBM_s015
                                 + 0.772293 w_k1_INTC_s015
                                 + 0.738168 w_k1_KO_s015
                                 + 0.544214 w_k1_MAT_s015
                                 + 0.757279 w_k1_MSFT_s015
                                 + 0.714255 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.14781 w_k1_ADM_s016
                                 + 1.0961 w_k1_AXP_s016 + 1.77268 w_k1_DIS_s016
                                 + 1.41435 w_k1_FDX_s016
                                 + 0.914686 w_k1_IBM_s016
                                 + 1.3127 w_k1_INTC_s016
                                 + 0.981216 w_k1_KO_s016
                                 + 0.913714 w_k1_MAT_s016
                                 + 1.48476 w_k1_MSFT_s016
                                 + 1.04694 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.99982 w_k1_ADM_s016
                                 + 1.23313 w_k1_AXP_s016
                                 + 1.64326 w_k1_DIS_s016 + 1.3442 w_k1_FDX_s016
                                 + 1.61846 w_k1_IBM_s016
                                 + 1.39412 w_k1_INTC_s016
                                 + 1.48644 w_k1_KO_s016 + 1.23903 w_k1_MAT_s016
                                 + 1.22342 w_k1_MSFT_s016
                                 + 1.23687 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.932884 w_k1_ADM_s016
                                 + 0.823826 w_k1_AXP_s016
                                 + 1.12476 w_k1_DIS_s016
                                 + 1.02873 w_k1_FDX_s016
                                 + 1.27639 w_k1_IBM_s016
                                 + 1.11783 w_k1_INTC_s016
                                 + 1.21023 w_k1_KO_s016 + 1.10988 w_k1_MAT_s016
                                 + 1.10957 w_k1_MSFT_s016
                                 + 1.01567 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.592881 w_k1_ADM_s016
                                 + 0.624481 w_k1_AXP_s016
                                 + 0.654081 w_k1_DIS_s016
                                 + 0.686509 w_k1_FDX_s016
                                 + 1.03174 w_k1_IBM_s016
                                 + 0.84406 w_k1_INTC_s016
                                 + 0.778278 w_k1_KO_s016
                                 + 0.963141 w_k1_MAT_s016
                                 + 0.862359 w_k1_MSFT_s016
                                 + 0.886517 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.848614 w_k1_ADM_s016
                                 + 0.564265 w_k1_AXP_s016
                                 + 0.71373 w_k1_DIS_s016
                                 + 0.867466 w_k1_FDX_s016
                                 + 0.605218 w_k1_IBM_s016
                                 + 0.786694 w_k1_INTC_s016
                                 + 0.670359 w_k1_KO_s016
                                 + 0.632269 w_k1_MAT_s016
                                 + 0.893516 w_k1_MSFT_s016
                                 + 0.834507 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.25671 w_k1_ADM_s016
                                 + 1.23695 w_k1_AXP_s016
                                 + 1.28896 w_k1_DIS_s016
                                 + 1.04334 w_k1_FDX_s016
                                 + 1.57491 w_k1_IBM_s016
                                 + 1.08883 w_k1_INTC_s016
                                 + 1.27612 w_k1_KO_s016 + 1.60424 w_k1_MAT_s016
                                 + 1.05956 w_k1_MSFT_s016
                                 + 1.19376 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.02059 w_k1_ADM_s016
                                 + 0.771573 w_k1_AXP_s016
                                 + 1.10506 w_k1_DIS_s016 + 1.0855 w_k1_FDX_s016
                                 + 0.895752 w_k1_IBM_s016
                                 + 1.07666 w_k1_INTC_s016
                                 + 0.963489 w_k1_KO_s016
                                 + 0.553051 w_k1_MAT_s016
                                 + 1.12155 w_k1_MSFT_s016
                                 + 0.944371 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.83251 w_k1_ADM_s016
                                 + 0.877131 w_k1_AXP_s016
                                 + 0.775854 w_k1_DIS_s016
                                 + 0.804821 w_k1_FDX_s016
                                 + 1.25376 w_k1_IBM_s016
                                 + 0.926433 w_k1_INTC_s016
                                 + 0.896821 w_k1_KO_s016
                                 + 1.17444 w_k1_MAT_s016
                                 + 0.907663 w_k1_MSFT_s016
                                 + 0.951236 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.748972 w_k1_ADM_s017
                                 + 0.760516 w_k1_AXP_s017
                                 + 0.904624 w_k1_DIS_s017
                                 + 0.882056 w_k1_FDX_s017
                                 + 0.930115 w_k1_IBM_s017
                                 + 1.00484 w_k1_INTC_s017
                                 + 0.924318 w_k1_KO_s017
                                 + 0.911434 w_k1_MAT_s017
                                 + 0.891029 w_k1_MSFT_s017
                                 + 0.912209 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.79775 w_k1_ADM_s017
                                 + 0.768685 w_k1_AXP_s017
                                 + 0.747176 w_k1_DIS_s017
                                 + 0.766115 w_k1_FDX_s017
                                 + 1.19517 w_k1_IBM_s017
                                 + 0.96323 w_k1_INTC_s017
                                 + 1.05885 w_k1_KO_s017 + 1.24824 w_k1_MAT_s017
                                 + 0.855476 w_k1_MSFT_s017
                                 + 0.912397 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.753674 w_k1_ADM_s017
                                 + 0.667466 w_k1_AXP_s017
                                 + 0.657036 w_k1_DIS_s017
                                 + 0.857189 w_k1_FDX_s017
                                 + 0.713198 w_k1_IBM_s017
                                 + 0.906136 w_k1_INTC_s017
                                 + 0.793226 w_k1_KO_s017
                                 + 0.672598 w_k1_MAT_s017
                                 + 0.748971 w_k1_MSFT_s017
                                 + 0.819947 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.09945 w_k1_ADM_s017
                                 + 1.27534 w_k1_AXP_s017
                                 + 1.36673 w_k1_DIS_s017 + 1.1932 w_k1_FDX_s017
                                 + 1.46449 w_k1_IBM_s017
                                 + 1.17191 w_k1_INTC_s017
                                 + 1.36199 w_k1_KO_s017 + 1.53944 w_k1_MAT_s017
                                 + 1.08699 w_k1_MSFT_s017
                                 + 1.06514 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.80425 w_k1_ADM_s017
                                 + 0.737715 w_k1_AXP_s017
                                 + 0.59921 w_k1_DIS_s017
                                 + 0.715329 w_k1_FDX_s017
                                 + 0.864318 w_k1_IBM_s017
                                 + 0.875707 w_k1_INTC_s017
                                 + 0.726896 w_k1_KO_s017
                                 + 1.04772 w_k1_MAT_s017
                                 + 0.760101 w_k1_MSFT_s017
                                 + 0.911687 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.25713 w_k1_ADM_s017
                                 + 1.10986 w_k1_AXP_s017
                                 + 1.00532 w_k1_DIS_s017
                                 + 1.10925 w_k1_FDX_s017
                                 + 0.892597 w_k1_IBM_s017
                                 + 0.957691 w_k1_INTC_s017
                                 + 1.00399 w_k1_KO_s017 + 1.30443 w_k1_MAT_s017
                                 + 1.02168 w_k1_MSFT_s017
                                 + 1.02089 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.02538 w_k1_ADM_s017
                                 + 0.838548 w_k1_AXP_s017
                                 + 0.930791 w_k1_DIS_s017
                                 + 0.991551 w_k1_FDX_s017
                                 + 0.655307 w_k1_IBM_s017
                                 + 1.05239 w_k1_INTC_s017
                                 + 0.739138 w_k1_KO_s017
                                 + 0.642038 w_k1_MAT_s017
                                 + 1.03747 w_k1_MSFT_s017
                                 + 0.861245 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.35171 w_k1_ADM_s017
                                 + 1.33755 w_k1_AXP_s017 + 1.5509 w_k1_DIS_s017
                                 + 1.35611 w_k1_FDX_s017
                                 + 0.973185 w_k1_IBM_s017
                                 + 1.48249 w_k1_INTC_s017
                                 + 1.19613 w_k1_KO_s017 + 1.40994 w_k1_MAT_s017
                                 + 1.47925 w_k1_MSFT_s017
                                 + 1.14705 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.832762 w_k1_ADM_s100
                                 + 1.18781 w_k1_AXP_s100
                                 + 1.11261 w_k1_DIS_s100
                                 + 1.15516 w_k1_FDX_s100
                                 + 0.815947 w_k1_IBM_s100
                                 + 1.17419 w_k1_INTC_s100
                                 + 1.06227 w_k1_KO_s100 + 1.06282 w_k1_MAT_s100
                                 + 1.05817 w_k1_MSFT_s100
                                 + 1.15604 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.965076 w_k1_ADM_s100
                                 + 0.937094 w_k1_AXP_s100
                                 + 1.02398 w_k1_DIS_s100
                                 + 0.941252 w_k1_FDX_s100
                                 + 0.979086 w_k1_IBM_s100
                                 + 0.883432 w_k1_INTC_s100
                                 + 1.07861 w_k1_KO_s100
                                 + 0.892193 w_k1_MAT_s100
                                 + 0.932251 w_k1_MSFT_s100
                                 + 0.877323 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.831866 w_k1_ADM_s100
                                 + 0.73599 w_k1_AXP_s100
                                 + 1.20735 w_k1_DIS_s100
                                 + 1.03936 w_k1_FDX_s100
                                 + 0.770713 w_k1_IBM_s100
                                 + 0.957661 w_k1_INTC_s100
                                 + 1.01932 w_k1_KO_s100
                                 + 0.867044 w_k1_MAT_s100
                                 + 1.04568 w_k1_MSFT_s100
                                 + 1.11275 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.87618 w_k1_ADM_s100
                                 + 0.970907 w_k1_AXP_s100
                                 + 1.25627 w_k1_DIS_s100
                                 + 0.976131 w_k1_FDX_s100
                                 + 1.43705 w_k1_IBM_s100
                                 + 1.15011 w_k1_INTC_s100
                                 + 1.27139 w_k1_KO_s100 + 1.26547 w_k1_MAT_s100
                                 + 1.03313 w_k1_MSFT_s100
                                 + 1.03849 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.37188 w_k1_ADM_s100
                                 + 1.43769 w_k1_AXP_s100
                                 + 2.10728 w_k1_DIS_s100
                                 + 1.43616 w_k1_FDX_s100 + 1.5064 w_k1_IBM_s100
                                 + 1.22815 w_k1_INTC_s100
                                 + 1.96408 w_k1_KO_s100 + 1.57871 w_k1_MAT_s100
                                 + 1.3505 w_k1_MSFT_s100 + 1.34239 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.08045 w_k1_ADM_s100
                                 + 1.04488 w_k1_AXP_s100
                                 + 1.12106 w_k1_DIS_s100
                                 + 1.01208 w_k1_FDX_s100
                                 + 1.14253 w_k1_IBM_s100
                                 + 1.00575 w_k1_INTC_s100 + 1.2745 w_k1_KO_s100
                                 + 0.957156 w_k1_MAT_s100
                                 + 0.952682 w_k1_MSFT_s100
                                 + 1.08052 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.901063 w_k1_ADM_s100
                                 + 0.737115 w_k1_AXP_s100
                                 + 0.656145 w_k1_DIS_s100
                                 + 0.812564 w_k1_FDX_s100
                                 + 0.650607 w_k1_IBM_s100
                                 + 1.04753 w_k1_INTC_s100 + 0.8208 w_k1_KO_s100
                                 + 0.525494 w_k1_MAT_s100
                                 + 0.871822 w_k1_MSFT_s100
                                 + 0.817658 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.781838 w_k1_ADM_s100
                                 + 0.581524 w_k1_AXP_s100
                                 + 0.681061 w_k1_DIS_s100
                                 + 0.674783 w_k1_FDX_s100
                                 + 0.86945 w_k1_IBM_s100
                                 + 0.817058 w_k1_INTC_s100
                                 + 0.931164 w_k1_KO_s100
                                 + 0.802072 w_k1_MAT_s100
                                 + 0.788874 w_k1_MSFT_s100
                                 + 0.866971 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.2171 w_k1_ADM_s101
                                 + 1.40381 w_k1_AXP_s101
                                 + 1.67829 w_k1_DIS_s101
                                 + 1.41051 w_k1_FDX_s101
                                 + 1.31968 w_k1_IBM_s101
                                 + 1.61994 w_k1_INTC_s101
                                 + 1.19457 w_k1_KO_s101 + 1.18526 w_k1_MAT_s101
                                 + 1.2402 w_k1_MSFT_s101 + 1.2202 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.989018 w_k1_ADM_s101
                                 + 0.72951 w_k1_AXP_s101
                                 + 1.05254 w_k1_DIS_s101
                                 + 0.969081 w_k1_FDX_s101
                                 + 0.837659 w_k1_IBM_s101
                                 + 0.922449 w_k1_INTC_s101
                                 + 0.914863 w_k1_KO_s101
                                 + 0.933874 w_k1_MAT_s101
                                 + 1.00008 w_k1_MSFT_s101
                                 + 0.974193 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.793182 w_k1_ADM_s101
                                 + 0.793441 w_k1_AXP_s101
                                 + 0.738658 w_k1_DIS_s101
                                 + 0.842676 w_k1_FDX_s101
                                 + 0.612864 w_k1_IBM_s101
                                 + 0.985779 w_k1_INTC_s101
                                 + 0.7641 w_k1_KO_s101 + 0.776666 w_k1_MAT_s101
                                 + 0.780519 w_k1_MSFT_s101
                                 + 0.856092 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.99025 w_k1_ADM_s101
                                 + 1.31156 w_k1_AXP_s101
                                 + 1.19543 w_k1_DIS_s101
                                 + 1.23334 w_k1_FDX_s101
                                 + 1.11285 w_k1_IBM_s101
                                 + 1.58097 w_k1_INTC_s101
                                 + 1.07922 w_k1_KO_s101 + 1.35945 w_k1_MAT_s101
                                 + 1.13648 w_k1_MSFT_s101
                                 + 1.13305 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.08803 w_k1_ADM_s101
                                 + 1.26444 w_k1_AXP_s101
                                 + 0.940584 w_k1_DIS_s101
                                 + 1.13313 w_k1_FDX_s101
                                 + 1.12309 w_k1_IBM_s101
                                 + 1.06397 w_k1_INTC_s101
                                 + 1.04951 w_k1_KO_s101 + 1.64446 w_k1_MAT_s101
                                 + 1.06367 w_k1_MSFT_s101
                                 + 0.999768 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.827016 w_k1_ADM_s101
                                 + 0.961648 w_k1_AXP_s101
                                 + 0.842955 w_k1_DIS_s101
                                 + 1.00396 w_k1_FDX_s101
                                 + 0.746228 w_k1_IBM_s101
                                 + 1.03445 w_k1_INTC_s101
                                 + 0.886379 w_k1_KO_s101
                                 + 1.22145 w_k1_MAT_s101
                                 + 0.968983 w_k1_MSFT_s101
                                 + 0.920555 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.771814 w_k1_ADM_s101
                                 + 0.715766 w_k1_AXP_s101
                                 + 0.715117 w_k1_DIS_s101
                                 + 0.701138 w_k1_FDX_s101
                                 + 0.877472 w_k1_IBM_s101
                                 + 0.926077 w_k1_INTC_s101
                                 + 0.826292 w_k1_KO_s101
                                 + 0.906739 w_k1_MAT_s101
                                 + 0.826827 w_k1_MSFT_s101
                                 + 0.807998 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.862236 w_k1_ADM_s101
                                 + 1.04914 w_k1_AXP_s101
                                 + 1.06586 w_k1_DIS_s101
                                 + 0.950056 w_k1_FDX_s101
                                 + 1.17949 w_k1_IBM_s101
                                 + 1.12039 w_k1_INTC_s101
                                 + 1.16959 w_k1_KO_s101 + 1.82057 w_k1_MAT_s101
                                 + 0.998259 w_k1_MSFT_s101
                                 + 1.10123 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.755443 w_k1_ADM_s102
                                 + 0.897271 w_k1_AXP_s102
                                 + 0.723018 w_k1_DIS_s102
                                 + 0.868675 w_k1_FDX_s102
                                 + 0.638465 w_k1_IBM_s102
                                 + 0.816608 w_k1_INTC_s102
                                 + 0.730309 w_k1_KO_s102
                                 + 0.723236 w_k1_MAT_s102
                                 + 0.852172 w_k1_MSFT_s102
                                 + 0.876161 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.985861 w_k1_ADM_s102
                                 + 1.03196 w_k1_AXP_s102
                                 + 1.10404 w_k1_DIS_s102
                                 + 1.02903 w_k1_FDX_s102
                                 + 1.00512 w_k1_IBM_s102
                                 + 0.821461 w_k1_INTC_s102
                                 + 1.05555 w_k1_KO_s102 + 1.15155 w_k1_MAT_s102
                                 + 1.01197 w_k1_MSFT_s102
                                 + 1.00116 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.32245 w_k1_ADM_s102
                                 + 1.35747 w_k1_AXP_s102
                                 + 1.22822 w_k1_DIS_s102
                                 + 1.26947 w_k1_FDX_s102
                                 + 0.827896 w_k1_IBM_s102
                                 + 1.05609 w_k1_INTC_s102
                                 + 1.04399 w_k1_KO_s102 + 1.29238 w_k1_MAT_s102
                                 + 1.19902 w_k1_MSFT_s102
                                 + 1.11608 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.1487 w_k1_ADM_s102
                                 + 1.33764 w_k1_AXP_s102 + 1.3497 w_k1_DIS_s102
                                 + 1.10095 w_k1_FDX_s102 + 1.3 w_k1_IBM_s102
                                 + 0.882862 w_k1_INTC_s102
                                 + 1.18664 w_k1_KO_s102 + 1.53302 w_k1_MAT_s102
                                 + 1.21721 w_k1_MSFT_s102
                                 + 1.15799 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.819068 w_k1_ADM_s102
                                 + 0.656222 w_k1_AXP_s102
                                 + 0.65391 w_k1_DIS_s102
                                 + 0.672991 w_k1_FDX_s102
                                 + 0.778501 w_k1_IBM_s102
                                 + 0.672809 w_k1_INTC_s102
                                 + 0.757673 w_k1_KO_s102
                                 + 0.71468 w_k1_MAT_s102
                                 + 0.709089 w_k1_MSFT_s102
                                 + 0.785051 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.832083 w_k1_ADM_s102
                                 + 0.880678 w_k1_AXP_s102
                                 + 0.820598 w_k1_DIS_s102
                                 + 0.811147 w_k1_FDX_s102
                                 + 1.20396 w_k1_IBM_s102
                                 + 0.752091 w_k1_INTC_s102
                                 + 0.935127 w_k1_KO_s102
                                 + 0.965113 w_k1_MAT_s102
                                 + 0.837406 w_k1_MSFT_s102
                                 + 1.03319 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.845767 w_k1_ADM_s102
                                 + 0.851444 w_k1_AXP_s102
                                 + 1.00862 w_k1_DIS_s102
                                 + 0.978958 w_k1_FDX_s102
                                 + 0.723578 w_k1_IBM_s102
                                 + 0.8144 w_k1_INTC_s102
                                 + 0.877639 w_k1_KO_s102
                                 + 0.798431 w_k1_MAT_s102
                                 + 1.02276 w_k1_MSFT_s102
                                 + 0.858413 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.05103 w_k1_ADM_s102
                                 + 2.10121 w_k1_AXP_s102
                                 + 1.83718 w_k1_DIS_s102
                                 + 1.44893 w_k1_FDX_s102
                                 + 1.70977 w_k1_IBM_s102
                                 + 1.03532 w_k1_INTC_s102
                                 + 1.52693 w_k1_KO_s102 + 2.56272 w_k1_MAT_s102
                                 + 1.38448 w_k1_MSFT_s102
                                 + 1.32313 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.30038 w_k1_ADM_s103
                                 + 1.07548 w_k1_AXP_s103
                                 + 0.956832 w_k1_DIS_s103
                                 + 0.909257 w_k1_FDX_s103
                                 + 1.2273 w_k1_IBM_s103
                                 + 0.77686 w_k1_INTC_s103
                                 + 1.08146 w_k1_KO_s103 + 1.31263 w_k1_MAT_s103
                                 + 1.02584 w_k1_MSFT_s103
                                 + 1.08405 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.13927 w_k1_ADM_s103
                                 + 1.0384 w_k1_AXP_s103 + 1.34947 w_k1_DIS_s103
                                 + 1.17517 w_k1_FDX_s103
                                 + 0.915825 w_k1_IBM_s103
                                 + 0.944246 w_k1_INTC_s103
                                 + 1.06443 w_k1_KO_s103 + 1.00679 w_k1_MAT_s103
                                 + 1.25603 w_k1_MSFT_s103
                                 + 1.16718 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.945813 w_k1_ADM_s103
                                 + 0.808837 w_k1_AXP_s103
                                 + 0.792943 w_k1_DIS_s103
                                 + 0.893329 w_k1_FDX_s103
                                 + 0.70626 w_k1_IBM_s103
                                 + 0.806361 w_k1_INTC_s103
                                 + 0.733272 w_k1_KO_s103
                                 + 0.586549 w_k1_MAT_s103
                                 + 0.857415 w_k1_MSFT_s103
                                 + 0.817617 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.2077 w_k1_ADM_s103
                                 + 1.6239 w_k1_AXP_s103 + 1.36972 w_k1_DIS_s103
                                 + 1.01635 w_k1_FDX_s103
                                 + 1.69511 w_k1_IBM_s103
                                 + 1.19835 w_k1_INTC_s103
                                 + 1.46164 w_k1_KO_s103 + 1.20742 w_k1_MAT_s103
                                 + 1.16647 w_k1_MSFT_s103
                                 + 1.22796 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.15792 w_k1_ADM_s103
                                 + 0.918858 w_k1_AXP_s103
                                 + 1.01719 w_k1_DIS_s103
                                 + 0.986506 w_k1_FDX_s103
                                 + 0.876561 w_k1_IBM_s103
                                 + 0.852791 w_k1_INTC_s103
                                 + 0.929935 w_k1_KO_s103
                                 + 0.730938 w_k1_MAT_s103
                                 + 0.921023 w_k1_MSFT_s103
                                 + 0.939218 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.81868 w_k1_ADM_s103
                                 + 1.3939 w_k1_AXP_s103 + 1.69499 w_k1_DIS_s103
                                 + 1.29258 w_k1_FDX_s103 + 1.308 w_k1_IBM_s103
                                 + 0.997507 w_k1_INTC_s103
                                 + 1.28845 w_k1_KO_s103
                                 + 0.898534 w_k1_MAT_s103
                                 + 1.31589 w_k1_MSFT_s103
                                 + 1.27465 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.783272 w_k1_ADM_s103
                                 + 0.636397 w_k1_AXP_s103
                                 + 0.564425 w_k1_DIS_s103
                                 + 0.677087 w_k1_FDX_s103
                                 + 0.805768 w_k1_IBM_s103
                                 + 0.774421 w_k1_INTC_s103
                                 + 0.786441 w_k1_KO_s103
                                 + 0.456275 w_k1_MAT_s103
                                 + 0.787568 w_k1_MSFT_s103
                                 + 0.849245 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.35527 w_k1_ADM_s103
                                 + 1.18815 w_k1_AXP_s103
                                 + 1.32239 w_k1_DIS_s103
                                 + 1.15009 w_k1_FDX_s103
                                 + 1.25343 w_k1_IBM_s103
                                 + 0.907415 w_k1_INTC_s103
                                 + 1.34919 w_k1_KO_s103 + 1.18326 w_k1_MAT_s103
                                 + 1.27037 w_k1_MSFT_s103
                                 + 1.17788 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.0896 w_k1_ADM_s104
                                 + 1.13184 w_k1_AXP_s104
                                 + 1.01845 w_k1_DIS_s104
                                 + 1.05063 w_k1_FDX_s104
                                 + 1.18229 w_k1_IBM_s104
                                 + 0.95013 w_k1_INTC_s104
                                 + 1.06993 w_k1_KO_s104 + 1.45755 w_k1_MAT_s104
                                 + 0.956984 w_k1_MSFT_s104
                                 + 1.04513 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.991933 w_k1_ADM_s104
                                 + 0.901559 w_k1_AXP_s104
                                 + 0.833379 w_k1_DIS_s104
                                 + 0.854738 w_k1_FDX_s104
                                 + 1.0927 w_k1_IBM_s104
                                 + 0.984755 w_k1_INTC_s104
                                 + 0.808712 w_k1_KO_s104
                                 + 1.09102 w_k1_MAT_s104
                                 + 0.920509 w_k1_MSFT_s104
                                 + 0.749458 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.779225 w_k1_ADM_s104
                                 + 0.686322 w_k1_AXP_s104
                                 + 0.715834 w_k1_DIS_s104
                                 + 0.795746 w_k1_FDX_s104
                                 + 0.732082 w_k1_IBM_s104
                                 + 0.863412 w_k1_INTC_s104
                                 + 0.759676 w_k1_KO_s104
                                 + 0.764916 w_k1_MAT_s104
                                 + 0.842745 w_k1_MSFT_s104
                                 + 0.773481 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.19267 w_k1_ADM_s104
                                 + 1.2008 w_k1_AXP_s104 + 1.62153 w_k1_DIS_s104
                                 + 1.35967 w_k1_FDX_s104
                                 + 1.00274 w_k1_IBM_s104
                                 + 1.53679 w_k1_INTC_s104
                                 + 1.20086 w_k1_KO_s104
                                 + 0.805353 w_k1_MAT_s104
                                 + 1.39688 w_k1_MSFT_s104
                                 + 0.964205 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.66357 w_k1_ADM_s104
                                 + 0.560342 w_k1_AXP_s104
                                 + 0.503546 w_k1_DIS_s104
                                 + 0.708297 w_k1_FDX_s104
                                 + 0.848288 w_k1_IBM_s104
                                 + 0.796122 w_k1_INTC_s104
                                 + 0.675327 w_k1_KO_s104
                                 + 0.759478 w_k1_MAT_s104
                                 + 0.69111 w_k1_MSFT_s104
                                 + 0.694133 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.1278 w_k1_ADM_s104
                                 + 1.4186 w_k1_AXP_s104 + 1.43919 w_k1_DIS_s104
                                 + 1.26402 w_k1_FDX_s104
                                 + 1.33979 w_k1_IBM_s104
                                 + 1.16996 w_k1_INTC_s104
                                 + 1.28578 w_k1_KO_s104 + 1.34624 w_k1_MAT_s104
                                 + 1.16661 w_k1_MSFT_s104
                                 + 1.17594 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.988431 w_k1_ADM_s104
                                 + 0.909611 w_k1_AXP_s104
                                 + 1.0274 w_k1_DIS_s104 + 1.05677 w_k1_FDX_s104
                                 + 0.82651 w_k1_IBM_s104
                                 + 1.0823 w_k1_INTC_s104 + 0.95219 w_k1_KO_s104
                                 + 0.908437 w_k1_MAT_s104
                                 + 1.08226 w_k1_MSFT_s104
                                 + 0.896349 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.05056 w_k1_ADM_s104
                                 + 1.04653 w_k1_AXP_s104
                                 + 0.915895 w_k1_DIS_s104
                                 + 1.01979 w_k1_FDX_s104
                                 + 1.32253 w_k1_IBM_s104
                                 + 1.04444 w_k1_INTC_s104 + 1.0538 w_k1_KO_s104
                                 + 1.37683 w_k1_MAT_s104
                                 + 0.965067 w_k1_MSFT_s104
                                 + 0.855366 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.16586 w_k1_ADM_s105
                                 + 1.36086 w_k1_AXP_s105
                                 + 1.26261 w_k1_DIS_s105 + 1.2416 w_k1_FDX_s105
                                 + 0.924363 w_k1_IBM_s105
                                 + 1.10075 w_k1_INTC_s105
                                 + 0.997569 w_k1_KO_s105
                                 + 1.41251 w_k1_MAT_s105
                                 + 1.16552 w_k1_MSFT_s105
                                 + 1.11927 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.02004 w_k1_ADM_s105
                                 + 1.3972 w_k1_AXP_s105 + 1.78504 w_k1_DIS_s105
                                 + 1.50485 w_k1_FDX_s105
                                 + 1.29339 w_k1_IBM_s105
                                 + 1.39424 w_k1_INTC_s105
                                 + 1.60211 w_k1_KO_s105 + 1.95606 w_k1_MAT_s105
                                 + 1.51267 w_k1_MSFT_s105
                                 + 1.43472 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.632333 w_k1_ADM_s105
                                 + 0.7583 w_k1_AXP_s105 + 0.67825 w_k1_DIS_s105
                                 + 0.891409 w_k1_FDX_s105
                                 + 0.692931 w_k1_IBM_s105
                                 + 1.06972 w_k1_INTC_s105
                                 + 0.753973 w_k1_KO_s105
                                 + 0.675005 w_k1_MAT_s105
                                 + 0.859027 w_k1_MSFT_s105
                                 + 1.0277 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.987683 w_k1_ADM_s105
                                 + 1.06594 w_k1_AXP_s105
                                 + 0.964304 w_k1_DIS_s105
                                 + 1.14924 w_k1_FDX_s105
                                 + 0.916692 w_k1_IBM_s105
                                 + 1.24189 w_k1_INTC_s105
                                 + 0.995666 w_k1_KO_s105
                                 + 0.918372 w_k1_MAT_s105
                                 + 1.11559 w_k1_MSFT_s105
                                 + 1.01228 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.764656 w_k1_ADM_s105
                                 + 1.03868 w_k1_AXP_s105
                                 + 1.25796 w_k1_DIS_s105 + 1.051 w_k1_FDX_s105
                                 + 1.59946 w_k1_IBM_s105
                                 + 1.02812 w_k1_INTC_s105
                                 + 1.40804 w_k1_KO_s105 + 1.8483 w_k1_MAT_s105
                                 + 1.14854 w_k1_MSFT_s105
                                 + 1.17429 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.85433 w_k1_ADM_s105
                                 + 0.910325 w_k1_AXP_s105
                                 + 0.734272 w_k1_DIS_s105
                                 + 0.849461 w_k1_FDX_s105
                                 + 1.16055 w_k1_IBM_s105
                                 + 1.13222 w_k1_INTC_s105
                                 + 0.994052 w_k1_KO_s105
                                 + 1.30663 w_k1_MAT_s105
                                 + 0.914496 w_k1_MSFT_s105
                                 + 0.985686 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.07591 w_k1_ADM_s105
                                 + 0.963889 w_k1_AXP_s105
                                 + 1.15068 w_k1_DIS_s105
                                 + 1.07989 w_k1_FDX_s105
                                 + 0.91455 w_k1_IBM_s105
                                 + 1.07107 w_k1_INTC_s105 + 1.0801 w_k1_KO_s105
                                 + 0.947261 w_k1_MAT_s105
                                 + 1.07126 w_k1_MSFT_s105 + 1.0598 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.795196 w_k1_ADM_s105
                                 + 0.840813 w_k1_AXP_s105
                                 + 0.837521 w_k1_DIS_s105
                                 + 0.863836 w_k1_FDX_s105
                                 + 1.00082 w_k1_IBM_s105
                                 + 0.918246 w_k1_INTC_s105
                                 + 0.999685 w_k1_KO_s105
                                 + 1.12459 w_k1_MAT_s105
                                 + 0.865992 w_k1_MSFT_s105
                                 + 0.920853 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.717545 w_k1_ADM_s106
                                 + 0.777484 w_k1_AXP_s106
                                 + 0.62778 w_k1_DIS_s106
                                 + 0.707505 w_k1_FDX_s106
                                 + 1.03169 w_k1_IBM_s106
                                 + 0.799335 w_k1_INTC_s106
                                 + 0.909727 w_k1_KO_s106
                                 + 1.42609 w_k1_MAT_s106
                                 + 0.697724 w_k1_MSFT_s106
                                 + 0.737371 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.05715 w_k1_ADM_s106
                                 + 0.780166 w_k1_AXP_s106
                                 + 0.905692 w_k1_DIS_s106
                                 + 0.857473 w_k1_FDX_s106
                                 + 0.972856 w_k1_IBM_s106
                                 + 0.955398 w_k1_INTC_s106
                                 + 1.06461 w_k1_KO_s106 + 1.08624 w_k1_MAT_s106
                                 + 0.914659 w_k1_MSFT_s106
                                 + 0.908597 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.684248 w_k1_ADM_s106
                                 + 0.527697 w_k1_AXP_s106
                                 + 0.483129 w_k1_DIS_s106
                                 + 0.711177 w_k1_FDX_s106
                                 + 0.679521 w_k1_IBM_s106
                                 + 0.698647 w_k1_INTC_s106
                                 + 0.659739 w_k1_KO_s106
                                 + 0.857952 w_k1_MAT_s106
                                 + 0.74468 w_k1_MSFT_s106
                                 + 0.706866 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.41577 w_k1_ADM_s106
                                 + 1.56558 w_k1_AXP_s106
                                 + 1.54018 w_k1_DIS_s106 + 1.3864 w_k1_FDX_s106
                                 + 1.31943 w_k1_IBM_s106
                                 + 1.17656 w_k1_INTC_s106
                                 + 1.45689 w_k1_KO_s106 + 1.90286 w_k1_MAT_s106
                                 + 1.37074 w_k1_MSFT_s106
                                 + 1.17529 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.09721 w_k1_ADM_s106
                                 + 1.10856 w_k1_AXP_s106
                                 + 0.876183 w_k1_DIS_s106
                                 + 1.03379 w_k1_FDX_s106
                                 + 0.769731 w_k1_IBM_s106
                                 + 0.912647 w_k1_INTC_s106
                                 + 0.851464 w_k1_KO_s106
                                 + 1.03621 w_k1_MAT_s106
                                 + 0.909486 w_k1_MSFT_s106
                                 + 0.965656 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.869262 w_k1_ADM_s106
                                 + 0.605217 w_k1_AXP_s106
                                 + 0.802495 w_k1_DIS_s106
                                 + 0.859346 w_k1_FDX_s106
                                 + 0.978635 w_k1_IBM_s106
                                 + 0.76255 w_k1_INTC_s106
                                 + 0.866391 w_k1_KO_s106
                                 + 0.999024 w_k1_MAT_s106
                                 + 0.787415 w_k1_MSFT_s106
                                 + 0.831633 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.16232 w_k1_ADM_s106
                                 + 1.21261 w_k1_AXP_s106
                                 + 1.17435 w_k1_DIS_s106
                                 + 1.11825 w_k1_FDX_s106
                                 + 1.19612 w_k1_IBM_s106
                                 + 0.874448 w_k1_INTC_s106
                                 + 1.19771 w_k1_KO_s106 + 1.46077 w_k1_MAT_s106
                                 + 0.959793 w_k1_MSFT_s106
                                 + 0.969011 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.978857 w_k1_ADM_s106
                                 + 0.934602 w_k1_AXP_s106
                                 + 1.21104 w_k1_DIS_s106
                                 + 1.12272 w_k1_FDX_s106
                                 + 0.83024 w_k1_IBM_s106
                                 + 1.20537 w_k1_INTC_s106
                                 + 1.03186 w_k1_KO_s106
                                 + 0.856335 w_k1_MAT_s106
                                 + 1.23234 w_k1_MSFT_s106
                                 + 0.98357 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.17283 w_k1_ADM_s107
                                 + 1.40229 w_k1_AXP_s107
                                 + 1.41184 w_k1_DIS_s107
                                 + 1.16326 w_k1_FDX_s107
                                 + 1.51212 w_k1_IBM_s107
                                 + 0.925337 w_k1_INTC_s107
                                 + 1.25705 w_k1_KO_s107 + 1.39956 w_k1_MAT_s107
                                 + 0.901517 w_k1_MSFT_s107
                                 + 1.18321 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.883842 w_k1_ADM_s107
                                 + 0.749005 w_k1_AXP_s107
                                 + 0.734911 w_k1_DIS_s107
                                 + 0.931287 w_k1_FDX_s107
                                 + 0.807508 w_k1_IBM_s107
                                 + 0.991161 w_k1_INTC_s107
                                 + 0.743065 w_k1_KO_s107
                                 + 0.679229 w_k1_MAT_s107
                                 + 1.03432 w_k1_MSFT_s107
                                 + 0.877128 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.21375 w_k1_ADM_s107
                                 + 1.3471 w_k1_AXP_s107 + 2.11117 w_k1_DIS_s107
                                 + 1.43382 w_k1_FDX_s107
                                 + 1.20606 w_k1_IBM_s107
                                 + 1.26356 w_k1_INTC_s107
                                 + 1.43537 w_k1_KO_s107 + 1.48803 w_k1_MAT_s107
                                 + 1.47558 w_k1_MSFT_s107
                                 + 1.28374 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.844269 w_k1_ADM_s107
                                 + 0.754934 w_k1_AXP_s107
                                 + 1.01715 w_k1_DIS_s107
                                 + 0.995835 w_k1_FDX_s107
                                 + 0.723821 w_k1_IBM_s107
                                 + 1.03114 w_k1_INTC_s107
                                 + 0.96011 w_k1_KO_s107
                                 + 0.462083 w_k1_MAT_s107
                                 + 0.907986 w_k1_MSFT_s107
                                 + 0.913154 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.867976 w_k1_ADM_s107
                                 + 0.844092 w_k1_AXP_s107
                                 + 1.15947 w_k1_DIS_s107 + 0.9322 w_k1_FDX_s107
                                 + 1.17262 w_k1_IBM_s107
                                 + 1.14782 w_k1_INTC_s107
                                 + 1.06364 w_k1_KO_s107
                                 + 0.929273 w_k1_MAT_s107
                                 + 1.13218 w_k1_MSFT_s107
                                 + 0.897036 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.25712 w_k1_ADM_s107
                                 + 1.20643 w_k1_AXP_s107
                                 + 1.23407 w_k1_DIS_s107
                                 + 1.24157 w_k1_FDX_s107
                                 + 0.82978 w_k1_IBM_s107
                                 + 0.966948 w_k1_INTC_s107
                                 + 0.901321 w_k1_KO_s107
                                 + 1.09681 w_k1_MAT_s107
                                 + 1.01934 w_k1_MSFT_s107
                                 + 1.06352 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.598268 w_k1_ADM_s107
                                 + 0.56321 w_k1_AXP_s107
                                 + 0.585469 w_k1_DIS_s107
                                 + 0.635051 w_k1_FDX_s107
                                 + 0.751596 w_k1_IBM_s107
                                 + 0.811676 w_k1_INTC_s107
                                 + 0.669408 w_k1_KO_s107
                                 + 0.612827 w_k1_MAT_s107
                                 + 0.758845 w_k1_MSFT_s107
                                 + 0.763067 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.01183 w_k1_ADM_s107
                                 + 1.11951 w_k1_AXP_s107
                                 + 1.03449 w_k1_DIS_s107
                                 + 1.00279 w_k1_FDX_s107
                                 + 1.11938 w_k1_IBM_s107
                                 + 1.06704 w_k1_INTC_s107
                                 + 0.95917 w_k1_KO_s107 + 1.02431 w_k1_MAT_s107
                                 + 0.917674 w_k1_MSFT_s107
                                 + 0.999692 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.649908 w_k1_ADM_s110
                                 + 0.613511 w_k1_AXP_s110
                                 + 0.546095 w_k1_DIS_s110
                                 + 0.730405 w_k1_FDX_s110
                                 + 0.644306 w_k1_IBM_s110
                                 + 0.94711 w_k1_INTC_s110
                                 + 0.616466 w_k1_KO_s110
                                 + 0.915616 w_k1_MAT_s110
                                 + 0.73386 w_k1_MSFT_s110
                                 + 0.726569 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.01948 w_k1_ADM_s110
                                 + 0.862423 w_k1_AXP_s110
                                 + 0.979113 w_k1_DIS_s110
                                 + 1.0284 w_k1_FDX_s110 + 0.83283 w_k1_IBM_s110
                                 + 1.01763 w_k1_INTC_s110
                                 + 0.955855 w_k1_KO_s110
                                 + 0.919746 w_k1_MAT_s110
                                 + 0.888284 w_k1_MSFT_s110
                                 + 0.922154 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.862269 w_k1_ADM_s110
                                 + 0.781735 w_k1_AXP_s110
                                 + 0.75437 w_k1_DIS_s110
                                 + 0.948793 w_k1_FDX_s110
                                 + 0.499077 w_k1_IBM_s110
                                 + 0.934705 w_k1_INTC_s110
                                 + 0.783329 w_k1_KO_s110
                                 + 0.702862 w_k1_MAT_s110
                                 + 0.848319 w_k1_MSFT_s110
                                 + 0.778202 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.767856 w_k1_ADM_s110
                                 + 0.793366 w_k1_AXP_s110
                                 + 0.70282 w_k1_DIS_s110
                                 + 0.833248 w_k1_FDX_s110
                                 + 0.84405 w_k1_IBM_s110
                                 + 1.04317 w_k1_INTC_s110
                                 + 0.851513 w_k1_KO_s110
                                 + 1.00425 w_k1_MAT_s110
                                 + 0.844964 w_k1_MSFT_s110
                                 + 0.85015 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.822988 w_k1_ADM_s110
                                 + 0.844317 w_k1_AXP_s110
                                 + 0.643684 w_k1_DIS_s110
                                 + 0.697974 w_k1_FDX_s110
                                 + 0.993885 w_k1_IBM_s110
                                 + 0.857637 w_k1_INTC_s110
                                 + 0.918862 w_k1_KO_s110
                                 + 1.25827 w_k1_MAT_s110
                                 + 0.731411 w_k1_MSFT_s110
                                 + 0.796543 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.97762 w_k1_ADM_s110
                                 + 1.11999 w_k1_AXP_s110
                                 + 1.25581 w_k1_DIS_s110
                                 + 1.29575 w_k1_FDX_s110
                                 + 0.910364 w_k1_IBM_s110
                                 + 1.35098 w_k1_INTC_s110
                                 + 1.11295 w_k1_KO_s110 + 1.1053 w_k1_MAT_s110
                                 + 1.19153 w_k1_MSFT_s110 + 1.079 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.11407 w_k1_ADM_s110
                                 + 1.32822 w_k1_AXP_s110
                                 + 1.51445 w_k1_DIS_s110
                                 + 1.13311 w_k1_FDX_s110
                                 + 1.80616 w_k1_IBM_s110
                                 + 1.16617 w_k1_INTC_s110
                                 + 1.53688 w_k1_KO_s110 + 2.26321 w_k1_MAT_s110
                                 + 1.11372 w_k1_MSFT_s110
                                 + 1.00307 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.00519 w_k1_ADM_s110
                                 + 1.01273 w_k1_AXP_s110
                                 + 1.02944 w_k1_DIS_s110
                                 + 0.972553 w_k1_FDX_s110
                                 + 1.32583 w_k1_IBM_s110
                                 + 1.10398 w_k1_INTC_s110
                                 + 1.23921 w_k1_KO_s110 + 1.46152 w_k1_MAT_s110
                                 + 1.09993 w_k1_MSFT_s110
                                 + 1.01356 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.866364 w_k1_ADM_s111
                                 + 1.06505 w_k1_AXP_s111
                                 + 0.926981 w_k1_DIS_s111
                                 + 0.907347 w_k1_FDX_s111
                                 + 0.956935 w_k1_IBM_s111
                                 + 0.870769 w_k1_INTC_s111
                                 + 1.08639 w_k1_KO_s111 + 1.1694 w_k1_MAT_s111
                                 + 0.963329 w_k1_MSFT_s111
                                 + 1.04838 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.86356 w_k1_ADM_s111
                                 + 0.9203 w_k1_AXP_s111
                                 + 0.994099 w_k1_DIS_s111
                                 + 0.830262 w_k1_FDX_s111
                                 + 1.29501 w_k1_IBM_s111
                                 + 0.83926 w_k1_INTC_s111
                                 + 1.17812 w_k1_KO_s111 + 1.19328 w_k1_MAT_s111
                                 + 0.896182 w_k1_MSFT_s111
                                 + 1.12286 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.928501 w_k1_ADM_s111
                                 + 1.1742 w_k1_AXP_s111 + 1.32364 w_k1_DIS_s111
                                 + 1.19934 w_k1_FDX_s111
                                 + 0.971528 w_k1_IBM_s111
                                 + 1.16609 w_k1_INTC_s111
                                 + 1.09314 w_k1_KO_s111
                                 + 0.824881 w_k1_MAT_s111
                                 + 1.15013 w_k1_MSFT_s111
                                 + 1.03758 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.09426 w_k1_ADM_s111
                                 + 1.79324 w_k1_AXP_s111
                                 + 1.78204 w_k1_DIS_s111
                                 + 1.37548 w_k1_FDX_s111
                                 + 1.48061 w_k1_IBM_s111
                                 + 1.31296 w_k1_INTC_s111
                                 + 1.53014 w_k1_KO_s111 + 1.41569 w_k1_MAT_s111
                                 + 1.33542 w_k1_MSFT_s111
                                 + 1.38276 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.02444 w_k1_ADM_s111
                                 + 1.01821 w_k1_AXP_s111
                                 + 1.35309 w_k1_DIS_s111
                                 + 1.14101 w_k1_FDX_s111 + 1.3493 w_k1_IBM_s111
                                 + 1.22485 w_k1_INTC_s111
                                 + 1.22439 w_k1_KO_s111
                                 + 0.744743 w_k1_MAT_s111
                                 + 1.13211 w_k1_MSFT_s111
                                 + 1.16113 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.668176 w_k1_ADM_s111
                                 + 0.661769 w_k1_AXP_s111
                                 + 0.637948 w_k1_DIS_s111
                                 + 0.672246 w_k1_FDX_s111
                                 + 0.800312 w_k1_IBM_s111
                                 + 0.98354 w_k1_INTC_s111
                                 + 0.82382 w_k1_KO_s111
                                 + 0.608615 w_k1_MAT_s111
                                 + 0.90467 w_k1_MSFT_s111
                                 + 0.870709 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.773479 w_k1_ADM_s111
                                 + 0.864627 w_k1_AXP_s111
                                 + 0.897869 w_k1_DIS_s111
                                 + 0.952363 w_k1_FDX_s111
                                 + 0.665524 w_k1_IBM_s111
                                 + 1.01067 w_k1_INTC_s111
                                 + 0.792645 w_k1_KO_s111
                                 + 0.577334 w_k1_MAT_s111
                                 + 0.948103 w_k1_MSFT_s111
                                 + 0.858661 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.990606 w_k1_ADM_s111
                                 + 1.27754 w_k1_AXP_s111
                                 + 1.10849 w_k1_DIS_s111
                                 + 1.27697 w_k1_FDX_s111
                                 + 0.77394 w_k1_IBM_s111
                                 + 0.977822 w_k1_INTC_s111
                                 + 0.988801 w_k1_KO_s111
                                 + 0.998663 w_k1_MAT_s111
                                 + 1.10355 w_k1_MSFT_s111
                                 + 1.13662 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.707362 w_k1_ADM_s112
                                 + 0.603493 w_k1_AXP_s112
                                 + 0.608812 w_k1_DIS_s112
                                 + 0.718962 w_k1_FDX_s112
                                 + 0.81405 w_k1_IBM_s112
                                 + 0.725431 w_k1_INTC_s112
                                 + 0.719304 w_k1_KO_s112
                                 + 0.799476 w_k1_MAT_s112
                                 + 0.745584 w_k1_MSFT_s112
                                 + 0.811952 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.39063 w_k1_ADM_s112
                                 + 0.936821 w_k1_AXP_s112
                                 + 1.14713 w_k1_DIS_s112
                                 + 1.11092 w_k1_FDX_s112
                                 + 0.93024 w_k1_IBM_s112
                                 + 0.897343 w_k1_INTC_s112
                                 + 0.823571 w_k1_KO_s112
                                 + 0.734291 w_k1_MAT_s112
                                 + 1.05343 w_k1_MSFT_s112
                                 + 1.05482 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.20107 w_k1_ADM_s112
                                 + 0.928048 w_k1_AXP_s112
                                 + 0.794934 w_k1_DIS_s112
                                 + 0.907997 w_k1_FDX_s112
                                 + 0.974655 w_k1_IBM_s112
                                 + 0.766869 w_k1_INTC_s112
                                 + 0.838841 w_k1_KO_s112
                                 + 1.11105 w_k1_MAT_s112
                                 + 0.872262 w_k1_MSFT_s112
                                 + 0.880581 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.14221 w_k1_ADM_s112
                                 + 1.11851 w_k1_AXP_s112
                                 + 0.945474 w_k1_DIS_s112 + 1.144 w_k1_FDX_s112
                                 + 0.694155 w_k1_IBM_s112
                                 + 0.789227 w_k1_INTC_s112
                                 + 0.774232 w_k1_KO_s112
                                 + 0.759599 w_k1_MAT_s112
                                 + 0.968306 w_k1_MSFT_s112
                                 + 0.9588 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.982269 w_k1_ADM_s112
                                 + 0.602549 w_k1_AXP_s112
                                 + 0.922235 w_k1_DIS_s112
                                 + 0.83766 w_k1_FDX_s112
                                 + 1.07259 w_k1_IBM_s112
                                 + 0.867035 w_k1_INTC_s112
                                 + 0.968133 w_k1_KO_s112
                                 + 0.830535 w_k1_MAT_s112
                                 + 0.922248 w_k1_MSFT_s112
                                 + 0.984929 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.1915 w_k1_ADM_s112
                                 + 1.21377 w_k1_AXP_s112 + 1.0347 w_k1_DIS_s112
                                 + 1.01878 w_k1_FDX_s112
                                 + 1.35402 w_k1_IBM_s112
                                 + 0.865842 w_k1_INTC_s112
                                 + 1.04885 w_k1_KO_s112 + 1.60241 w_k1_MAT_s112
                                 + 0.856801 w_k1_MSFT_s112
                                 + 1.0977 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.22928 w_k1_ADM_s112
                                 + 1.0558 w_k1_AXP_s112 + 1.20801 w_k1_DIS_s112
                                 + 1.19297 w_k1_FDX_s112
                                 + 1.02641 w_k1_IBM_s112
                                 + 1.02954 w_k1_INTC_s112
                                 + 1.10999 w_k1_KO_s112
                                 + 0.968631 w_k1_MAT_s112
                                 + 1.15745 w_k1_MSFT_s112
                                 + 1.10076 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.44796 w_k1_ADM_s112
                                 + 1.36164 w_k1_AXP_s112
                                 + 1.61595 w_k1_DIS_s112 + 1.3999 w_k1_FDX_s112
                                 + 1.29603 w_k1_IBM_s112
                                 + 1.1859 w_k1_INTC_s112 + 1.2937 w_k1_KO_s112
                                 + 1.40438 w_k1_MAT_s112
                                 + 1.29958 w_k1_MSFT_s112
                                 + 1.20983 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.805071 w_k1_ADM_s113
                                 + 1.08131 w_k1_AXP_s113
                                 + 0.906788 w_k1_DIS_s113
                                 + 0.977536 w_k1_FDX_s113
                                 + 0.947923 w_k1_IBM_s113
                                 + 0.910987 w_k1_INTC_s113
                                 + 0.952723 w_k1_KO_s113
                                 + 0.992351 w_k1_MAT_s113
                                 + 0.926606 w_k1_MSFT_s113
                                 + 0.992136 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.993064 w_k1_ADM_s113
                                 + 1.02632 w_k1_AXP_s113
                                 + 1.14716 w_k1_DIS_s113
                                 + 1.03263 w_k1_FDX_s113
                                 + 1.15484 w_k1_IBM_s113
                                 + 1.19326 w_k1_INTC_s113
                                 + 1.17082 w_k1_KO_s113
                                 + 0.751726 w_k1_MAT_s113
                                 + 0.985993 w_k1_MSFT_s113
                                 + 0.94863 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.81305 w_k1_ADM_s113
                                 + 0.80206 w_k1_AXP_s113
                                 + 0.697707 w_k1_DIS_s113
                                 + 0.773774 w_k1_FDX_s113
                                 + 0.879322 w_k1_IBM_s113
                                 + 0.780995 w_k1_INTC_s113
                                 + 0.823546 w_k1_KO_s113
                                 + 0.818119 w_k1_MAT_s113
                                 + 0.747515 w_k1_MSFT_s113
                                 + 0.863779 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.778642 w_k1_ADM_s113
                                 + 0.687766 w_k1_AXP_s113
                                 + 0.582453 w_k1_DIS_s113
                                 + 0.72355 w_k1_FDX_s113
                                 + 0.699921 w_k1_IBM_s113
                                 + 0.790519 w_k1_INTC_s113
                                 + 0.647086 w_k1_KO_s113
                                 + 0.623259 w_k1_MAT_s113
                                 + 0.827289 w_k1_MSFT_s113
                                 + 0.743472 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.930135 w_k1_ADM_s113
                                 + 1.10423 w_k1_AXP_s113
                                 + 1.24524 w_k1_DIS_s113
                                 + 1.17101 w_k1_FDX_s113
                                 + 0.866222 w_k1_IBM_s113
                                 + 1.14721 w_k1_INTC_s113 + 1.0478 w_k1_KO_s113
                                 + 1.07092 w_k1_MAT_s113
                                 + 1.16841 w_k1_MSFT_s113
                                 + 1.08499 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.949954 w_k1_ADM_s113
                                 + 1.0571 w_k1_AXP_s113
                                 + 0.875714 w_k1_DIS_s113
                                 + 0.979244 w_k1_FDX_s113
                                 + 0.694999 w_k1_IBM_s113
                                 + 0.886133 w_k1_INTC_s113
                                 + 0.787355 w_k1_KO_s113
                                 + 0.675686 w_k1_MAT_s113
                                 + 0.906938 w_k1_MSFT_s113
                                 + 0.86393 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.27851 w_k1_ADM_s113
                                 + 1.36823 w_k1_AXP_s113
                                 + 1.29154 w_k1_DIS_s113
                                 + 1.11136 w_k1_FDX_s113
                                 + 1.34403 w_k1_IBM_s113
                                 + 0.99907 w_k1_INTC_s113
                                 + 1.26487 w_k1_KO_s113 + 1.49116 w_k1_MAT_s113
                                 + 1.18202 w_k1_MSFT_s113
                                 + 1.12535 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.08945 w_k1_ADM_s113
                                 + 0.970883 w_k1_AXP_s113
                                 + 1.1769 w_k1_DIS_s113
                                 + 0.916153 w_k1_FDX_s113
                                 + 1.07296 w_k1_IBM_s113
                                 + 0.926177 w_k1_INTC_s113
                                 + 0.970441 w_k1_KO_s113
                                 + 0.912497 w_k1_MAT_s113
                                 + 0.937921 w_k1_MSFT_s113
                                 + 0.911786 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.820671 w_k1_ADM_s114
                                 + 0.783395 w_k1_AXP_s114
                                 + 0.856244 w_k1_DIS_s114
                                 + 0.756076 w_k1_FDX_s114
                                 + 1.75428 w_k1_IBM_s114
                                 + 0.690606 w_k1_INTC_s114
                                 + 1.07193 w_k1_KO_s114 + 1.21165 w_k1_MAT_s114
                                 + 0.702147 w_k1_MSFT_s114
                                 + 0.8871 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.13532 w_k1_ADM_s114
                                 + 0.925182 w_k1_AXP_s114
                                 + 0.939978 w_k1_DIS_s114
                                 + 1.0604 w_k1_FDX_s114
                                 + 0.736881 w_k1_IBM_s114
                                 + 1.07746 w_k1_INTC_s114
                                 + 0.810287 w_k1_KO_s114 + 0.6592 w_k1_MAT_s114
                                 + 0.967961 w_k1_MSFT_s114
                                 + 0.800338 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.886742 w_k1_ADM_s114
                                 + 0.678119 w_k1_AXP_s114
                                 + 0.653948 w_k1_DIS_s114
                                 + 0.824671 w_k1_FDX_s114
                                 + 0.595878 w_k1_IBM_s114
                                 + 0.70827 w_k1_INTC_s114 + 0.7492 w_k1_KO_s114
                                 + 0.727261 w_k1_MAT_s114
                                 + 0.795484 w_k1_MSFT_s114
                                 + 0.79419 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.49693 w_k1_ADM_s114
                                 + 1.1676 w_k1_AXP_s114 + 1.51424 w_k1_DIS_s114
                                 + 1.30096 w_k1_FDX_s114
                                 + 0.839701 w_k1_IBM_s114
                                 + 1.05163 w_k1_INTC_s114
                                 + 1.09112 w_k1_KO_s114
                                 + 0.781217 w_k1_MAT_s114
                                 + 1.33928 w_k1_MSFT_s114
                                 + 1.10139 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.31517 w_k1_ADM_s114
                                 + 1.02336 w_k1_AXP_s114
                                 + 1.09118 w_k1_DIS_s114
                                 + 1.12295 w_k1_FDX_s114
                                 + 0.954006 w_k1_IBM_s114
                                 + 0.853707 w_k1_INTC_s114
                                 + 1.11183 w_k1_KO_s114 + 1.18139 w_k1_MAT_s114
                                 + 0.987265 w_k1_MSFT_s114
                                 + 1.03931 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.803396 w_k1_ADM_s114
                                 + 0.714289 w_k1_AXP_s114
                                 + 0.67475 w_k1_DIS_s114
                                 + 0.732666 w_k1_FDX_s114
                                 + 0.903044 w_k1_IBM_s114
                                 + 0.75211 w_k1_INTC_s114
                                 + 0.796085 w_k1_KO_s114
                                 + 0.780536 w_k1_MAT_s114
                                 + 0.805274 w_k1_MSFT_s114
                                 + 0.738357 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.985638 w_k1_ADM_s114
                                 + 0.854593 w_k1_AXP_s114
                                 + 0.97962 w_k1_DIS_s114
                                 + 0.862699 w_k1_FDX_s114
                                 + 1.31025 w_k1_IBM_s114
                                 + 0.984356 w_k1_INTC_s114
                                 + 1.05192 w_k1_KO_s114 + 1.31007 w_k1_MAT_s114
                                 + 0.989118 w_k1_MSFT_s114
                                 + 1.00569 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.83284 w_k1_ADM_s114
                                 + 1.51943 w_k1_AXP_s114 + 1.658 w_k1_DIS_s114
                                 + 1.41945 w_k1_FDX_s114
                                 + 1.17456 w_k1_IBM_s114
                                 + 1.18456 w_k1_INTC_s114
                                 + 1.37795 w_k1_KO_s114 + 1.22767 w_k1_MAT_s114
                                 + 1.34803 w_k1_MSFT_s114
                                 + 1.20433 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.06665 w_k1_ADM_s115
                                 + 1.13968 w_k1_AXP_s115
                                 + 1.21393 w_k1_DIS_s115
                                 + 1.10606 w_k1_FDX_s115
                                 + 0.962659 w_k1_IBM_s115
                                 + 1.07411 w_k1_INTC_s115
                                 + 0.985463 w_k1_KO_s115
                                 + 0.987164 w_k1_MAT_s115
                                 + 1.13594 w_k1_MSFT_s115
                                 + 1.11798 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.850326 w_k1_ADM_s115
                                 + 0.741098 w_k1_AXP_s115
                                 + 0.829425 w_k1_DIS_s115
                                 + 0.860959 w_k1_FDX_s115
                                 + 0.679877 w_k1_IBM_s115
                                 + 0.789928 w_k1_INTC_s115
                                 + 0.675553 w_k1_KO_s115
                                 + 0.721607 w_k1_MAT_s115
                                 + 0.818047 w_k1_MSFT_s115
                                 + 0.837338 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.937745 w_k1_ADM_s115
                                 + 0.704294 w_k1_AXP_s115
                                 + 1.00613 w_k1_DIS_s115
                                 + 0.997844 w_k1_FDX_s115
                                 + 0.956795 w_k1_IBM_s115
                                 + 0.872869 w_k1_INTC_s115
                                 + 0.940482 w_k1_KO_s115
                                 + 0.868987 w_k1_MAT_s115
                                 + 0.876369 w_k1_MSFT_s115
                                 + 0.912128 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.36344 w_k1_ADM_s115
                                 + 1.07687 w_k1_AXP_s115
                                 + 1.45088 w_k1_DIS_s115
                                 + 1.26581 w_k1_FDX_s115
                                 + 1.23303 w_k1_IBM_s115
                                 + 1.06756 w_k1_INTC_s115
                                 + 1.15572 w_k1_KO_s115
                                 + 0.956917 w_k1_MAT_s115
                                 + 1.04205 w_k1_MSFT_s115
                                 + 1.06456 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.00021 w_k1_ADM_s115
                                 + 0.876128 w_k1_AXP_s115
                                 + 0.942823 w_k1_DIS_s115
                                 + 0.919899 w_k1_FDX_s115
                                 + 1.27043 w_k1_IBM_s115
                                 + 1.03453 w_k1_INTC_s115
                                 + 1.00125 w_k1_KO_s115 + 1.00312 w_k1_MAT_s115
                                 + 0.961985 w_k1_MSFT_s115
                                 + 0.90034 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.30635 w_k1_ADM_s115
                                 + 1.4865 w_k1_AXP_s115 + 1.44749 w_k1_DIS_s115
                                 + 1.33489 w_k1_FDX_s115
                                 + 0.936211 w_k1_IBM_s115
                                 + 1.24205 w_k1_INTC_s115
                                 + 1.17662 w_k1_KO_s115
                                 + 0.941357 w_k1_MAT_s115
                                 + 1.12817 w_k1_MSFT_s115
                                 + 1.22267 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.962588 w_k1_ADM_s115
                                 + 1.07717 w_k1_AXP_s115
                                 + 1.02133 w_k1_DIS_s115
                                 + 1.10762 w_k1_FDX_s115
                                 + 1.10482 w_k1_IBM_s115
                                 + 1.10021 w_k1_INTC_s115
                                 + 0.959296 w_k1_KO_s115
                                 + 1.19603 w_k1_MAT_s115
                                 + 1.06433 w_k1_MSFT_s115
                                 + 0.978394 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.3353 w_k1_ADM_s115
                                 + 1.38116 w_k1_AXP_s115
                                 + 1.38938 w_k1_DIS_s115 + 1.3148 w_k1_FDX_s115
                                 + 1.58358 w_k1_IBM_s115
                                 + 1.10034 w_k1_INTC_s115
                                 + 1.20044 w_k1_KO_s115 + 1.65471 w_k1_MAT_s115
                                 + 1.14056 w_k1_MSFT_s115
                                 + 1.26036 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.918331 w_k1_ADM_s116
                                 + 1.11146 w_k1_AXP_s116
                                 + 0.989131 w_k1_DIS_s116
                                 + 0.865394 w_k1_FDX_s116
                                 + 1.58993 w_k1_IBM_s116
                                 + 1.23599 w_k1_INTC_s116
                                 + 1.31106 w_k1_KO_s116 + 1.42979 w_k1_MAT_s116
                                 + 1.12638 w_k1_MSFT_s116
                                 + 0.956522 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.774149 w_k1_ADM_s116
                                 + 0.847882 w_k1_AXP_s116
                                 + 0.687884 w_k1_DIS_s116
                                 + 0.77088 w_k1_FDX_s116
                                 + 1.16549 w_k1_IBM_s116
                                 + 0.915752 w_k1_INTC_s116
                                 + 0.952369 w_k1_KO_s116
                                 + 1.14682 w_k1_MAT_s116
                                 + 0.705952 w_k1_MSFT_s116
                                 + 0.988653 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.14193 w_k1_ADM_s116
                                 + 1.22342 w_k1_AXP_s116 + 1.1123 w_k1_DIS_s116
                                 + 1.20476 w_k1_FDX_s116
                                 + 0.952259 w_k1_IBM_s116
                                 + 1.12769 w_k1_INTC_s116
                                 + 1.06835 w_k1_KO_s116
                                 + 0.823547 w_k1_MAT_s116
                                 + 0.933162 w_k1_MSFT_s116
                                 + 1.05044 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.726835 w_k1_ADM_s116
                                 + 0.92775 w_k1_AXP_s116
                                 + 0.752774 w_k1_DIS_s116
                                 + 0.862442 w_k1_FDX_s116
                                 + 0.802221 w_k1_IBM_s116
                                 + 0.936999 w_k1_INTC_s116
                                 + 0.848763 w_k1_KO_s116
                                 + 0.932649 w_k1_MAT_s116
                                 + 0.923233 w_k1_MSFT_s116
                                 + 0.854349 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.00257 w_k1_ADM_s116
                                 + 1.22701 w_k1_AXP_s116
                                 + 0.852245 w_k1_DIS_s116
                                 + 0.926101 w_k1_FDX_s116
                                 + 1.05662 w_k1_IBM_s116
                                 + 1.00994 w_k1_INTC_s116
                                 + 0.943024 w_k1_KO_s116
                                 + 1.14541 w_k1_MAT_s116
                                 + 0.913542 w_k1_MSFT_s116
                                 + 0.959904 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.9123 w_k1_ADM_s116
                                 + 1.39601 w_k1_AXP_s116
                                 + 1.34528 w_k1_DIS_s116
                                 + 1.24716 w_k1_FDX_s116
                                 + 1.31233 w_k1_IBM_s116
                                 + 1.32678 w_k1_INTC_s116
                                 + 1.23867 w_k1_KO_s116 + 1.7177 w_k1_MAT_s116
                                 + 1.15115 w_k1_MSFT_s116
                                 + 1.10596 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.913022 w_k1_ADM_s116
                                 + 0.918093 w_k1_AXP_s116
                                 + 1.09484 w_k1_DIS_s116
                                 + 1.07236 w_k1_FDX_s116
                                 + 0.823522 w_k1_IBM_s116
                                 + 1.25639 w_k1_INTC_s116
                                 + 0.903424 w_k1_KO_s116
                                 + 0.690367 w_k1_MAT_s116
                                 + 1.07106 w_k1_MSFT_s116
                                 + 0.927878 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.28063 w_k1_ADM_s116
                                 + 1.74552 w_k1_AXP_s116
                                 + 1.28688 w_k1_DIS_s116
                                 + 1.44081 w_k1_FDX_s116
                                 + 0.868475 w_k1_IBM_s116
                                 + 1.54235 w_k1_INTC_s116
                                 + 1.01591 w_k1_KO_s116 + 1.07599 w_k1_MAT_s116
                                 + 1.29477 w_k1_MSFT_s116
                                 + 1.07251 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.762248 w_k1_ADM_s117
                                 + 0.592048 w_k1_AXP_s117
                                 + 0.595675 w_k1_DIS_s117
                                 + 0.722873 w_k1_FDX_s117
                                 + 0.612899 w_k1_IBM_s117
                                 + 0.815868 w_k1_INTC_s117
                                 + 0.683198 w_k1_KO_s117
                                 + 0.492377 w_k1_MAT_s117
                                 + 0.681144 w_k1_MSFT_s117
                                 + 0.868397 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.953537 w_k1_ADM_s117
                                 + 0.762022 w_k1_AXP_s117
                                 + 0.768331 w_k1_DIS_s117
                                 + 0.858933 w_k1_FDX_s117
                                 + 0.905191 w_k1_IBM_s117
                                 + 1.00015 w_k1_INTC_s117
                                 + 0.887555 w_k1_KO_s117
                                 + 0.959547 w_k1_MAT_s117
                                 + 0.882238 w_k1_MSFT_s117
                                 + 0.902697 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.28295 w_k1_ADM_s117
                                 + 1.28857 w_k1_AXP_s117
                                 + 1.26728 w_k1_DIS_s117
                                 + 1.07193 w_k1_FDX_s117
                                 + 1.58486 w_k1_IBM_s117
                                 + 1.33177 w_k1_INTC_s117
                                 + 1.25833 w_k1_KO_s117 + 1.62604 w_k1_MAT_s117
                                 + 1.03411 w_k1_MSFT_s117 + 1.1918 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.49323 w_k1_ADM_s117
                                 + 1.39458 w_k1_AXP_s117
                                 + 1.48707 w_k1_DIS_s117
                                 + 1.39323 w_k1_FDX_s117
                                 + 1.18646 w_k1_IBM_s117
                                 + 1.20387 w_k1_INTC_s117
                                 + 1.31985 w_k1_KO_s117 + 1.40537 w_k1_MAT_s117
                                 + 1.28037 w_k1_MSFT_s117 + 1.3894 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.23514 w_k1_ADM_s117
                                 + 1.11761 w_k1_AXP_s117
                                 + 0.976381 w_k1_DIS_s117
                                 + 0.987071 w_k1_FDX_s117
                                 + 0.839891 w_k1_IBM_s117
                                 + 1.02695 w_k1_INTC_s117
                                 + 1.05729 w_k1_KO_s117 + 0.81339 w_k1_MAT_s117
                                 + 0.863846 w_k1_MSFT_s117
                                 + 1.0202 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.01272 w_k1_ADM_s117
                                 + 1.13595 w_k1_AXP_s117
                                 + 1.29436 w_k1_DIS_s117
                                 + 1.08549 w_k1_FDX_s117
                                 + 0.995396 w_k1_IBM_s117
                                 + 1.18605 w_k1_INTC_s117
                                 + 1.15934 w_k1_KO_s117
                                 + 0.935102 w_k1_MAT_s117
                                 + 1.10384 w_k1_MSFT_s117
                                 + 1.15227 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.991227 w_k1_ADM_s117
                                 + 0.851723 w_k1_AXP_s117
                                 + 0.980145 w_k1_DIS_s117
                                 + 0.850004 w_k1_FDX_s117
                                 + 1.51096 w_k1_IBM_s117
                                 + 0.923659 w_k1_INTC_s117
                                 + 1.23424 w_k1_KO_s117
                                 + 0.873418 w_k1_MAT_s117
                                 + 0.85161 w_k1_MSFT_s117
                                 + 1.04933 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.889371 w_k1_ADM_s117
                                 + 0.854861 w_k1_AXP_s117
                                 + 0.845674 w_k1_DIS_s117
                                 + 0.827881 w_k1_FDX_s117
                                 + 1.20304 w_k1_IBM_s117
                                 + 0.835383 w_k1_INTC_s117
                                 + 0.949363 w_k1_KO_s117
                                 + 0.916672 w_k1_MAT_s117
                                 + 0.840834 w_k1_MSFT_s117
                                 + 1.09552 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.016074296777749 w_k1_ADM_s000
                                 + 0.0147446397331761 w_k1_ADM_s001
                                 + 0.00501117719029467 w_k1_ADM_s002
                                 + 0.013312536247292 w_k1_ADM_s003
                                 + 0.0106023244154036 w_k1_ADM_s004
                                 + 0.0164988340538782 w_k1_ADM_s005
                                 + 0.0178611353568846 w_k1_ADM_s006
                                 + 0.00510338127999251 w_k1_ADM_s007
                                 + 0.00906364448283077 w_k1_ADM_s010
                                 + 0.0398739522507845 w_k1_ADM_s011
                                 + 0.0356003703920579 w_k1_ADM_s012
                                 + 0.0172199291701116 w_k1_ADM_s013
                                 + 0.0163896525620076 w_k1_ADM_s014
                                 + 0.0313009379841917 w_k1_ADM_s015
                                 + 0.0138035636944631 w_k1_ADM_s016
                                 + 0.0223620371351485 w_k1_ADM_s017
                                 + 0.0100659417376262 w_k1_ADM_s100
                                 + 0.0113992275918696 w_k1_ADM_s101
                                 + 0.0351092426275889 w_k1_ADM_s102
                                 + 0.0188419973169145 w_k1_ADM_s103
                                 + 0.0357229856666508 w_k1_ADM_s104
                                 + 0.0316260678844884 w_k1_ADM_s105
                                 + 0.0406612132058533 w_k1_ADM_s106
                                 + 0.0262751112701889 w_k1_ADM_s107
                                 + 0.050359961618757 w_k1_ADM_s110
                                 + 0.0499718062335629 w_k1_ADM_s111
                                 + 0.132123190365562 w_k1_ADM_s112
                                 + 0.0258520882446648 w_k1_ADM_s113
                                 + 0.0617768204677559 w_k1_ADM_s114
                                 + 0.0374197061799222 w_k1_ADM_s115
                                 + 0.0657394293888388 w_k1_ADM_s116
                                 + 0.0940740301555708 w_k1_ADM_s117
                                 + 0.0156720285265946 w_k1_AXP_s000
                                 + 0.0159696053821913 w_k1_AXP_s001
                                 + 0.00455070400204656 w_k1_AXP_s002
                                 + 0.0146145728384957 w_k1_AXP_s003
                                 + 0.0113852140480125 w_k1_AXP_s004
                                 + 0.0154626546456627 w_k1_AXP_s005
                                 + 0.0189409089195675 w_k1_AXP_s006
                                 + 0.00448720852244511 w_k1_AXP_s007
                                 + 0.0110109340553051 w_k1_AXP_s010
                                 + 0.0496153393563096 w_k1_AXP_s011
                                 + 0.0369626609102876 w_k1_AXP_s012
                                 + 0.0158193607728674 w_k1_AXP_s013
                                 + 0.0177028922829204 w_k1_AXP_s014
                                 + 0.0309672740700541 w_k1_AXP_s015
                                 + 0.0121489564653815 w_k1_AXP_s016
                                 + 0.0218788123388932 w_k1_AXP_s017
                                 + 0.0099490049848376 w_k1_AXP_s100
                                 + 0.0124017004698992 w_k1_AXP_s101
                                 + 0.0391519977733917 w_k1_AXP_s102
                                 + 0.0162208569672798 w_k1_AXP_s103
                                 + 0.0355220114852509 w_k1_AXP_s104
                                 + 0.0359239381699933 w_k1_AXP_s105
                                 + 0.039226874557661 w_k1_AXP_s106
                                 + 0.02691780953297 w_k1_AXP_s107
                                 + 0.0520940248496626 w_k1_AXP_s110
                                 + 0.0601851387604878 w_k1_AXP_s111
                                 + 0.111540959968068 w_k1_AXP_s112
                                 + 0.0282875591177742 w_k1_AXP_s113
                                 + 0.0511198033580697 w_k1_AXP_s114
                                 + 0.0348960221036082 w_k1_AXP_s115
                                 + 0.0797476795219458 w_k1_AXP_s116
                                 + 0.0886815098267339 w_k1_AXP_s117
                                 + 0.0176158023261681 w_k1_DIS_s000
                                 + 0.0172930025799445 w_k1_DIS_s001
                                 + 0.004721665543368 w_k1_DIS_s002
                                 + 0.0157205141025999 w_k1_DIS_s003
                                 + 0.0121245672527513 w_k1_DIS_s004
                                 + 0.0156162964600632 w_k1_DIS_s005
                                 + 0.0178486144343917 w_k1_DIS_s006
                                 + 0.00494636519792646 w_k1_DIS_s007
                                 + 0.0106969770240952 w_k1_DIS_s010
                                 + 0.0482620485693652 w_k1_DIS_s011
                                 + 0.0378869837218733 w_k1_DIS_s012
                                 + 0.0153757941635072 w_k1_DIS_s013
                                 + 0.0158248118238435 w_k1_DIS_s014
                                 + 0.03025660100381 w_k1_DIS_s015
                                 + 0.0155954732142991 w_k1_DIS_s016
                                 + 0.0224640484703541 w_k1_DIS_s017
                                 + 0.0117134252817053 w_k1_DIS_s100
                                 + 0.0129406962293679 w_k1_DIS_s101
                                 + 0.0387642450423954 w_k1_DIS_s102
                                 + 0.0173175445190189 w_k1_DIS_s103
                                 + 0.0366455187065823 w_k1_DIS_s104
                                 + 0.0363220253891599 w_k1_DIS_s105
                                 + 0.039826629734303 w_k1_DIS_s106
                                 + 0.0295853319692964 w_k1_DIS_s107
                                 + 0.054619915171257 w_k1_DIS_s110
                                 + 0.0610134510450831 w_k1_DIS_s111
                                 + 0.11563771514795 w_k1_DIS_s112
                                 + 0.0276610136190165 w_k1_DIS_s113
                                 + 0.055308237882514 w_k1_DIS_s114
                                 + 0.0387113025851154 w_k1_DIS_s115
                                 + 0.0709728043088238 w_k1_DIS_s116
                                 + 0.0923218297971389 w_k1_DIS_s117
                                 + 0.0164933339510113 w_k1_FDX_s000
                                 + 0.0161429742051228 w_k1_FDX_s001
                                 + 0.0045193893867398 w_k1_FDX_s002
                                 + 0.0142255606253147 w_k1_FDX_s003
                                 + 0.0112907295585129 w_k1_FDX_s004
                                 + 0.0153347071723919 w_k1_FDX_s005
                                 + 0.0184194148142847 w_k1_FDX_s006
                                 + 0.00478353487119689 w_k1_FDX_s007
                                 + 0.010299693200634 w_k1_FDX_s010
                                 + 0.0447448837462378 w_k1_FDX_s011
                                 + 0.0364567821114251 w_k1_FDX_s012
                                 + 0.0169175797258768 w_k1_FDX_s013
                                 + 0.0164557296717686 w_k1_FDX_s014
                                 + 0.0296712688488654 w_k1_FDX_s015
                                 + 0.0148093855462326 w_k1_FDX_s016
                                 + 0.0224610490923613 w_k1_FDX_s017
                                 + 0.0103532939227126 w_k1_FDX_s100
                                 + 0.0124201774731637 w_k1_FDX_s101
                                 + 0.0359960722003281 w_k1_FDX_s102
                                 + 0.0159473405492798 w_k1_FDX_s103
                                 + 0.036493305149928 w_k1_FDX_s104
                                 + 0.0371489081969151 w_k1_FDX_s105
                                 + 0.0397039831806304 w_k1_FDX_s106
                                 + 0.0273506414191983 w_k1_FDX_s107
                                 + 0.0541763729121569 w_k1_FDX_s110
                                 + 0.0570466615057623 w_k1_FDX_s111
                                 + 0.11699823180262 w_k1_FDX_s112
                                 + 0.0265311203866807 w_k1_FDX_s113
                                 + 0.0537794688517713 w_k1_FDX_s114
                                 + 0.037711406881869 w_k1_FDX_s115
                                 + 0.0739872422819595 w_k1_FDX_s116
                                 + 0.0874498282835832 w_k1_FDX_s117
                                 + 0.0166481793858146 w_k1_IBM_s000
                                 + 0.0166979969462239 w_k1_IBM_s001
                                 + 0.0046764617037352 w_k1_IBM_s002
                                 + 0.0145030678340424 w_k1_IBM_s003
                                 + 0.0122102238338864 w_k1_IBM_s004
                                 + 0.0161691399967229 w_k1_IBM_s005
                                 + 0.0174335389948162 w_k1_IBM_s006
                                 + 0.00461086339824225 w_k1_IBM_s007
                                 + 0.0110186474839225 w_k1_IBM_s010
                                 + 0.0456920576618796 w_k1_IBM_s011
                                 + 0.0384495849386423 w_k1_IBM_s012
                                 + 0.0161661957343979 w_k1_IBM_s013
                                 + 0.0168098624977864 w_k1_IBM_s014
                                 + 0.0313884381319984 w_k1_IBM_s015
                                 + 0.0150804383509516 w_k1_IBM_s016
                                 + 0.0228755715653619 w_k1_IBM_s017
                                 + 0.0108129866693645 w_k1_IBM_s100
                                 + 0.012229217088118 w_k1_IBM_s101
                                 + 0.0360192016684271 w_k1_IBM_s102
                                 + 0.0162124947865488 w_k1_IBM_s103
                                 + 0.0368411728067838 w_k1_IBM_s104
                                 + 0.0374502141771661 w_k1_IBM_s105
                                 + 0.0391284652568944 w_k1_IBM_s106
                                 + 0.028774530258545 w_k1_IBM_s107
                                 + 0.056457084500602 w_k1_IBM_s110
                                 + 0.0571292761542972 w_k1_IBM_s111
                                 + 0.118890100735646 w_k1_IBM_s112
                                 + 0.0263120332632048 w_k1_IBM_s113
                                 + 0.0551854451201262 w_k1_IBM_s114
                                 + 0.0375276969689407 w_k1_IBM_s115
                                 + 0.0743958678390289 w_k1_IBM_s116
                                 + 0.091515486853285 w_k1_IBM_s117
                                 + 0.0163294383723688 w_k1_INTC_s000
                                 + 0.0156633813562201 w_k1_INTC_s001
                                 + 0.00452891850608787 w_k1_INTC_s002
                                 + 0.0142015925451723 w_k1_INTC_s003
                                 + 0.00976899407339258 w_k1_INTC_s004
                                 + 0.0183622841490068 w_k1_INTC_s005
                                 + 0.0182466430833019 w_k1_INTC_s006
                                 + 0.00423712379394156 w_k1_INTC_s007
                                 + 0.0117154628797927 w_k1_INTC_s010
                                 + 0.042795450424978 w_k1_INTC_s011
                                 + 0.0326954081947093 w_k1_INTC_s012
                                 + 0.0171059562596169 w_k1_INTC_s013
                                 + 0.0155349836670996 w_k1_INTC_s014
                                 + 0.0274698559540852 w_k1_INTC_s015
                                 + 0.015086289683776 w_k1_INTC_s016
                                 + 0.0232286074910119 w_k1_INTC_s017
                                 + 0.0106369986975883 w_k1_INTC_s100
                                 + 0.0141584489968058 w_k1_INTC_s101
                                 + 0.0299278844656823 w_k1_INTC_s102
                                 + 0.01397731019643 w_k1_INTC_s103
                                 + 0.0377243189694415 w_k1_INTC_s104
                                 + 0.0395836070885554 w_k1_INTC_s105
                                 + 0.036496856022192 w_k1_INTC_s106
                                 + 0.0276599374692093 w_k1_INTC_s107
                                 + 0.0586327582918608 w_k1_INTC_s110
                                 + 0.0574440011589633 w_k1_INTC_s111
                                 + 0.101373679482867 w_k1_INTC_s112
                                 + 0.026123404003621 w_k1_INTC_s113
                                 + 0.0496727853827306 w_k1_INTC_s114
                                 + 0.0355406254587234 w_k1_INTC_s115
                                 + 0.0826344246038095 w_k1_INTC_s116
                                 + 0.0932659637785941 w_k1_INTC_s117
                                 + 0.0156800792107595 w_k1_KO_s000
                                 + 0.0170893350636452 w_k1_KO_s001
                                 + 0.00525414496432194 w_k1_KO_s002
                                 + 0.0142594045792476 w_k1_KO_s003
                                 + 0.0114363340390576 w_k1_KO_s004
                                 + 0.0149956705143362 w_k1_KO_s005
                                 + 0.0171764837852238 w_k1_KO_s006
                                 + 0.00418827023941975 w_k1_KO_s007
                                 + 0.0108633648403592 w_k1_KO_s010
                                 + 0.0451426809400935 w_k1_KO_s011
                                 + 0.03757491576545 w_k1_KO_s012
                                 + 0.0165605320413557 w_k1_KO_s013
                                 + 0.0165037970012087 w_k1_KO_s014
                                 + 0.0285212843864446 w_k1_KO_s015
                                 + 0.014144159926433 w_k1_KO_s016
                                 + 0.0229951056893483 w_k1_KO_s017
                                 + 0.0122791990543539 w_k1_KO_s100
                                 + 0.0121030770215327 w_k1_KO_s101
                                 + 0.0360391024152349 w_k1_KO_s102
                                 + 0.0163146224963919 w_k1_KO_s103
                                 + 0.0349234240194891 w_k1_KO_s104
                                 + 0.0382019139458888 w_k1_KO_s105
                                 + 0.0408235932536511 w_k1_KO_s106
                                 + 0.0266500900400994 w_k1_KO_s107
                                 + 0.0575513363221 w_k1_KO_s110
                                 + 0.0607036281820714 w_k1_KO_s111
                                 + 0.108872515380947 w_k1_KO_s112
                                 + 0.0266799818833029 w_k1_KO_s113
                                 + 0.0539613046738888 w_k1_KO_s114
                                 + 0.0344942677405104 w_k1_KO_s115
                                 + 0.0727889368965138 w_k1_KO_s116
                                 + 0.0928749355382506 w_k1_KO_s117
                                 + 0.0121626336375606 w_k1_MAT_s000
                                 + 0.0168277139914959 w_k1_MAT_s001
                                 + 0.00430329636859889 w_k1_MAT_s002
                                 + 0.017300104247182 w_k1_MAT_s003
                                 + 0.0127509795446549 w_k1_MAT_s004
                                 + 0.0148850943347666 w_k1_MAT_s005
                                 + 0.0160288650024999 w_k1_MAT_s006
                                 + 0.00416043501543653 w_k1_MAT_s007
                                 + 0.00954422225906605 w_k1_MAT_s010
                                 + 0.0521436394044766 w_k1_MAT_s011
                                 + 0.04414235634153 w_k1_MAT_s012
                                 + 0.0159416144326403 w_k1_MAT_s013
                                 + 0.0163641736898323 w_k1_MAT_s014
                                 + 0.0287312531447415 w_k1_MAT_s015
                                 + 0.0127377602304588 w_k1_MAT_s016
                                 + 0.0263477117655803 w_k1_MAT_s017
                                 + 0.010249592967817 w_k1_MAT_s100
                                 + 0.0148999235110576 w_k1_MAT_s101
                                 + 0.0421484142502718 w_k1_MAT_s102
                                 + 0.0143281487610337 w_k1_MAT_s103
                                 + 0.0382128726557176 w_k1_MAT_s104
                                 + 0.0439567258769434 w_k1_MAT_s105
                                 + 0.0485300787816894 w_k1_MAT_s106
                                 + 0.0259579142991349 w_k1_MAT_s107
                                 + 0.0672653893767207 w_k1_MAT_s110
                                 + 0.0540250232424692 w_k1_MAT_s111
                                 + 0.12268101505992 w_k1_MAT_s112
                                 + 0.0257744690762469 w_k1_MAT_s113
                                 + 0.0541225429863332 w_k1_MAT_s114
                                 + 0.0354900042847141 w_k1_MAT_s115
                                 + 0.0785296305070629 w_k1_MAT_s116
                                 + 0.0881543129948513 w_k1_MAT_s117
                                 + 0.0162392915692521 w_k1_MSFT_s000
                                 + 0.0164651941917737 w_k1_MSFT_s001
                                 + 0.00467859734518664 w_k1_MSFT_s002
                                 + 0.0157045790110552 w_k1_MSFT_s003
                                 + 0.0112808224291908 w_k1_MSFT_s004
                                 + 0.0156012529384383 w_k1_MSFT_s005
                                 + 0.0159683773171542 w_k1_MSFT_s006
                                 + 0.00503590894878309 w_k1_MSFT_s007
                                 + 0.0104538096628145 w_k1_MSFT_s010
                                 + 0.0459538243646272 w_k1_MSFT_s011
                                 + 0.0351054136867578 w_k1_MSFT_s012
                                 + 0.016692387305736 w_k1_MSFT_s013
                                 + 0.0163674337044902 w_k1_MSFT_s014
                                 + 0.0306876377118007 w_k1_MSFT_s015
                                 + 0.0155231892018948 w_k1_MSFT_s016
                                 + 0.0219589326601082 w_k1_MSFT_s017
                                 + 0.0103224380968065 w_k1_MSFT_s100
                                 + 0.0121711735943566 w_k1_MSFT_s101
                                 + 0.0365530831168071 w_k1_MSFT_s102
                                 + 0.016679248494648 w_k1_MSFT_s103
                                 + 0.0363108206646962 w_k1_MSFT_s104
                                 + 0.037329271368912 w_k1_MSFT_s105
                                 + 0.0382479055704785 w_k1_MSFT_s106
                                 + 0.026869752781869 w_k1_MSFT_s107
                                 + 0.0533118870246596 w_k1_MSFT_s110
                                 + 0.0583067330610321 w_k1_MSFT_s111
                                 + 0.111157084421465 w_k1_MSFT_s112
                                 + 0.0264161088047094 w_k1_MSFT_s113
                                 + 0.0531740366966246 w_k1_MSFT_s114
                                 + 0.03503315211249 w_k1_MSFT_s115
                                 + 0.0726112219321163 w_k1_MSFT_s116
                                 + 0.0852620504981236 w_k1_MSFT_s117
                                 + 0.0159095138750377 w_k1_PG_s000
                                 + 0.0176807284265393 w_k1_PG_s001
                                 + 0.00498191315196248 w_k1_PG_s002
                                 + 0.0149349890194818 w_k1_PG_s003
                                 + 0.0105744722415184 w_k1_PG_s004
                                 + 0.0152894596258017 w_k1_PG_s005
                                 + 0.0192335986235972 w_k1_PG_s006
                                 + 0.00427167665920126 w_k1_PG_s007
                                 + 0.00973355020577004 w_k1_PG_s010
                                 + 0.0483326248926725 w_k1_PG_s011
                                 + 0.0378310179362263 w_k1_PG_s012
                                 + 0.0168329036363258 w_k1_PG_s013
                                 + 0.016927579805014 w_k1_PG_s014
                                 + 0.0276349647812436 w_k1_PG_s015
                                 + 0.0141616919171898 w_k1_PG_s016
                                 + 0.0217007669601673 w_k1_PG_s017
                                 + 0.0106902684529536 w_k1_PG_s100
                                 + 0.0122072823367601 w_k1_PG_s101
                                 + 0.035910922559149 w_k1_PG_s102
                                 + 0.0165713923447925 w_k1_PG_s103
                                 + 0.0323333918296576 w_k1_PG_s104
                                 + 0.0381679703005552 w_k1_PG_s105
                                 + 0.036355063833864 w_k1_PG_s106
                                 + 0.0265917111611743 w_k1_PG_s107
                                 + 0.050336518635041 w_k1_PG_s110
                                 + 0.0598086877000464 w_k1_PG_s111
                                 + 0.114637107714952 w_k1_PG_s112
                                 + 0.0257558387316366 w_k1_PG_s113
                                 + 0.0507555928257028 w_k1_PG_s114
                                 + 0.0354923893523647 w_k1_PG_s115
                                 + 0.0710084637434551 w_k1_PG_s116
                                 + 0.094871607414796 w_k1_PG_s117 >= 1.19
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
