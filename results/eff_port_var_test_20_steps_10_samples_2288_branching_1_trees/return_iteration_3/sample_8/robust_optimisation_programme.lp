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
 w_k1_ADM_s0_return_constraint:  1.02471 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.998858 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.19493 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.875483 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.05321 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.39243 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.858453 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.900413 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.19465 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.04121 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.804193 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.778248 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.08935 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.922179 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.01667 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.1319 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.36329 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.910039 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.0653 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.50975 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.01202 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.797272 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.898046 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.739982 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.17365 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.19839 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.01864 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.26507 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.727843 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.8063 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.978181 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.21215 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.05418 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.759889 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.692749 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.986873 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.55965 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.19419 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.01444 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.02748 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.41656 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.922932 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.938784 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.54905 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.716947 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.09947 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.35419 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.919034 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.876068 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.663452 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.44941 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.07788 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.986247 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.34217 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.38493 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.911909 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.04647 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.53082 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.83441 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.658233 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.963956 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.694176 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.49361 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.10426 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.25387 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.37627 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.729116 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.97848 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.785808 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.05469 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.05749 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.20095 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.66887 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.985847 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.71395 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.34931 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.02901 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.08139 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.44174 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.869256 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.852972 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.57833 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.642764 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.874457 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.50335 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.21167 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.876282 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.584723 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.66287 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.989701 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.873594 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.24422 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.30583 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.992163 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.33933 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.81885 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.77023 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.660931 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.17209 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.73246 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.40147 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.22217 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.08898 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.83403 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.731011 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.06948 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.833764 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.50479 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.01824 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.37556 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.620534 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.07604 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.51527 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.61799 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.966471 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.07651 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.15358 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.953018 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.920305 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.32925 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.764964 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.05677 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.51278 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.19868 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.983244 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.756683 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.43758 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.02571 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.937053 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.07985 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.36324 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.855615 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.16133 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.41403 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.996329 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.757179 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.02761 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.812692 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.23182 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.278 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.13614 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.44715 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.735684 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.924497 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.82983 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.27889 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.09754 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.05156 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.762382 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.02243 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.51508 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.30676 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.19475 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.871341 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.43249 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.848965 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.958163 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.47684 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.20624 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.806687 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.95184 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.799084 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.01623 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.748703 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.40403 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.50536 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.01238 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.77772 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.903383 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.30875 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.867129 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.39798 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.566372 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.773812 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.15811 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.661586 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.66038 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.736879 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.85718 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.10369 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.03852 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.72532 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.04737 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.815471 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.752072 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.8315 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.705062 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.05821 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.997241 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.33327 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.972966 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.00488 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.09184 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.994593 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.935051 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.35569 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.964734 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.01732 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.30206 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.12051 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.03262 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.837377 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.5207 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.06813 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.97937 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.00018 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.1224 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.807436 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.984076 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.26717 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.782709 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.830864 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.0966 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.91262 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.20067 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.17613 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.19202 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.39839 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.852734 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.01836 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.944494 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.14551 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.962467 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.11841 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.838258 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.930058 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.28843 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.04889 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.09539 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.978711 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.27666 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.856412 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.9087 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.37873 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.873758 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.93705 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.34498 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.16304 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.06701 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.685253 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.68944 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.12924 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.931707 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.35099 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.14132 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.15669 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.00101 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.5411 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.738013 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.816291 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.20328 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.734097 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.33923 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.02967 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.971678 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.28814 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.90913 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.18195 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.00984 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.00296 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.967381 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.41728 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.726126 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.02762 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.19325 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.48383 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.14629 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.0329 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.72371 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.924186 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.769057 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.36262 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.03182 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.949725 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.871885 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.877441 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.844387 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.534266 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.63858 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.28702 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.932727 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.75142 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.13883 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.1192 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.837527 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.21277 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.639406 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.699124 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.07938 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.712379 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.80027 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.921871 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.836458 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.08981 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.09087 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.68407 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.05205 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.794553 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.897929 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.64616 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.487621 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.21676 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.21629 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.5729 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.02029 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.08863 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.11214 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.958225 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.852945 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.31524 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.785727 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.893018 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.28558 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.269 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.964434 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.785097 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.53192 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.02573 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.938369 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.996642 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.11343 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.840386 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.16571 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.31549 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.874436 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.694088 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.13482 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.877868 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.29089 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.26445 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.08536 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.52267 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.857037 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.0409 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.891684 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.24092 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.995993 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.06307 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.727762 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.919673 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.12942 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.12478 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.04552 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.01317 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.12094 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.909285 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.917197 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.22844 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.821957 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.999065 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.30572 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.0714 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.998017 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.821031 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.31619 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.16369 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.00197 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.21352 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.17406 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.0275 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.04277 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.20686 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.895706 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.822802 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.06217 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.862748 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.34283 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.07505 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.06425 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.31855 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.905168 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.08777 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.886993 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.14218 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.01545 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.989449 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.823522 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.09912 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.23 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.20196 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.19399958472572 z_k1_s0000
                                 - 0.116999809071612 z_k1_s0001
                                 - 0.215999154055993 z_k1_s0002
                                 - 0.139000019900961 z_k1_s0003
                                 - 0.226999580220206 z_k1_s0004
                                 - 0.245000044307436 z_k1_s0005
                                 - 0.0949998548418927 z_k1_s0006
                                 - 0.049999977621971 z_k1_s0007
                                 - 0.155000137231419 z_k1_s0010
                                 - 0.335000431703456 z_k1_s0011
                                 - 0.120000135614661 z_k1_s0012
                                 - 0.252001031401161 z_k1_s0013
                                 - 0.437000155460724 z_k1_s0014
                                 - 0.107000010145518 z_k1_s0015
                                 - 0.252001031401161 z_k1_s0016
                                 - 0.163000158412402 z_k1_s0017
                                 - 0.295999489540966 z_k1_s0020
                                 - 0.0649997279328576 z_k1_s0021
                                 - 0.291998401321211 z_k1_s0022
                                 - 0.100999670144582 z_k1_s0023
                                 - 0.293998945431088 z_k1_s0024
                                 - 0.115999619582892 z_k1_s0025
                                 - 0.174999145281192 z_k1_s0026
                                 - 0.249999686970004 z_k1_s0027
                                 - 0.101999860128608 z_k1_s0030
                                 - 0.103999788993982 z_k1_s0031
                                 - 0.138000323702607 z_k1_s0032
                                 - 0.063999903688552 z_k1_s0033
                                 - 0.15999975922138 z_k1_s0034
                                 - 0.173999697278315 z_k1_s0035
                                 - 0.138999852135978 z_k1_s0036
                                 - 0.208999542420652 z_k1_s0037
                                 - 0.339000143365157 z_k1_s0040
                                 - 0.461001436530497 z_k1_s0041
                                 - 0.184000272535892 z_k1_s0042
                                 - 0.303999449718979 z_k1_s0043
                                 - 0.158000319599401 z_k1_s0044
                                 - 0.451999181819009 z_k1_s0045
                                 - 0.722000207084377 z_k1_s0046
                                 - 0.407999261464946 z_k1_s0047
                                 - 0.855997660821252 z_k1_s0050
                                 - 0.173999601419102 z_k1_s0051
                                 - 0.474998975583612 z_k1_s0052
                                 - 0.578997478630021 z_k1_s0053
                                 - 0.242999612524575 z_k1_s0054
                                 - 0.249999310912762 z_k1_s0055
                                 - 0.258999284839624 z_k1_s0056
                                 - 0.328997851217965 z_k1_s0057
                                 - 0.106999791577545 z_k1_s0060
                                 - 0.267999890286729 z_k1_s0061
                                 - 0.338998660111737 z_k1_s0062
                                 - 0.0979997403889382 z_k1_s0063
                                 - 0.381998904679523 z_k1_s0064
                                 - 0.13399961834412 z_k1_s0065
                                 - 0.085999835537085 z_k1_s0066
                                 - 0.219999617280827 z_k1_s0067
                                 - 0.0669997606017359 z_k1_s0070
                                 - 0.280999560286014 z_k1_s0071
                                 - 0.163998895561185 z_k1_s0072
                                 - 0.326998744420701 z_k1_s0073
                                 - 0.159999634764658 z_k1_s0074
                                 - 0.120999618122371 z_k1_s0075
                                 - 0.264999443110056 z_k1_s0076
                                 - 0.151999422477186 z_k1_s0077
                                 - 1.23299740504356 z_k1_s0100
                                 - 0.666999298668399 z_k1_s0101
                                 - 0.723998412421501 z_k1_s0102
                                 - 0.684996058807514 z_k1_s0103
                                 - 0.668995814246332 z_k1_s0104
                                 - 0.815000154862444 z_k1_s0105
                                 - 0.462998711307726 z_k1_s0106
                                 - 0.367999438520883 z_k1_s0107
                                 - 0.0879999794586926 z_k1_s0110
                                 - 0.377000957611045 z_k1_s0111
                                 - 0.221999627120636 z_k1_s0112
                                 - 0.104000079851233 z_k1_s0113
                                 - 0.348999731973588 z_k1_s0114
                                 - 0.356000038382952 z_k1_s0115
                                 - 0.317999465875506 z_k1_s0116
                                 - 0.0950000949679638 z_k1_s0117
                                 - 0.50100141010445 z_k1_s0120
                                 - 0.209000147735456 z_k1_s0121
                                 - 0.220000191753123 z_k1_s0122
                                 - 0.205000069129013 z_k1_s0123
                                 - 0.564999913406708 z_k1_s0124
                                 - 0.695999216916732 z_k1_s0125
                                 - 0.479999103770053 z_k1_s0126
                                 - 0.567000297010034 z_k1_s0127
                                 - 0.844002452544458 z_k1_s0130
                                 - 0.815002039162699 z_k1_s0131
                                 - 0.555000656226264 z_k1_s0132
                                 - 0.388000354410277 z_k1_s0133
                                 - 0.819003115147914 z_k1_s0134
                                 - 0.787000417275033 z_k1_s0135
                                 - 0.630999823913526 z_k1_s0136
                                 - 1.07099998211038 z_k1_s0137
                                 - 0.222999944318574 z_k1_s0140
                                 - 0.551001075061913 z_k1_s0141
                                 - 0.344999926080308 z_k1_s0142
                                 - 0.153999959102512 z_k1_s0143
                                 - 0.261000121860803 z_k1_s0144
                                 - 0.304000875137682 z_k1_s0145
                                 - 0.575000785467423 z_k1_s0146
                                 - 0.312999403539385 z_k1_s0147
                                 - 0.806999157920308 z_k1_s0150
                                 - 1.5069933840818 z_k1_s0151
                                 - 0.332998835034975 z_k1_s0152
                                 - 0.664999128239693 z_k1_s0153
                                 - 0.388998680686979 z_k1_s0154
                                 - 0.565995858683292 z_k1_s0155
                                 - 0.418998506465111 z_k1_s0156
                                 - 0.439998640809041 z_k1_s0157
                                 - 0.221999939971207 z_k1_s0160
                                 - 0.324000094531004 z_k1_s0161
                                 - 0.34199894778239 z_k1_s0162
                                 - 0.312999643321487 z_k1_s0163
                                 - 0.169000004507825 z_k1_s0164
                                 - 0.466000263961296 z_k1_s0165
                                 - 0.230000150305367 z_k1_s0166
                                 - 0.525999897466926 z_k1_s0167
                                 - 0.832999782753929 z_k1_s0170
                                 - 1.75199728424882 z_k1_s0171
                                 - 0.835999228947567 z_k1_s0172
                                 - 0.904999467991701 z_k1_s0173
                                 - 0.920999604022166 z_k1_s0174
                                 - 2.06699474853993 z_k1_s0175
                                 - 1.18999968964847 z_k1_s0176
                                 - 0.764999668074114 z_k1_s0177
                                 - 1.15199641414455 z_k1_s1000
                                 - 0.721999594742614 z_k1_s1001
                                 - 0.851998633166455 z_k1_s1002
                                 - 0.424999323979967 z_k1_s1003
                                 - 1.73199829302136 z_k1_s1004
                                 - 0.470998983729942 z_k1_s1005
                                 - 0.440999205632132 z_k1_s1006
                                 - 0.373999084315108 z_k1_s1007
                                 - 1.08899760154797 z_k1_s1010
                                 - 1.19399803551775 z_k1_s1011
                                 - 1.77100172016039 z_k1_s1012
                                 - 0.582000155648161 z_k1_s1013
                                 - 1.67099880709548 z_k1_s1014
                                 - 1.60299682621134 z_k1_s1015
                                 - 2.03999643256311 z_k1_s1016
                                 - 0.548999719286417 z_k1_s1017
                                 - 0.249000095944571 z_k1_s1020
                                 - 0.38600046945949 z_k1_s1021
                                 - 0.562000756363307 z_k1_s1022
                                 - 0.222000112644561 z_k1_s1023
                                 - 0.51899964774465 z_k1_s1024
                                 - 0.316001345941033 z_k1_s1025
                                 - 0.2039999363112 z_k1_s1026
                                 - 0.353999906022378 z_k1_s1027
                                 - 0.326998721254357 z_k1_s1030
                                 - 0.670001387940479 z_k1_s1031
                                 - 0.160999871403566 z_k1_s1032
                                 - 0.516998980252392 z_k1_s1033
                                 - 0.175000017056064 z_k1_s1034
                                 - 0.329998709522746 z_k1_s1035
                                 - 0.538999896220664 z_k1_s1036
                                 - 0.286999979875941 z_k1_s1037
                                 - 0.204999759648835 z_k1_s1040
                                 - 0.360000062934112 z_k1_s1041
                                 - 0.500999495750494 z_k1_s1042
                                 - 0.950999574418134 z_k1_s1043
                                 - 0.272999929358473 z_k1_s1044
                                 - 0.27899969367425 z_k1_s1045
                                 - 0.216999643380076 z_k1_s1046
                                 - 0.764998358236552 z_k1_s1047
                                 - 0.209999970255039 z_k1_s1050
                                 - 0.442998560895251 z_k1_s1051
                                 - 0.524998855137673 z_k1_s1052
                                 - 0.778998513303314 z_k1_s1053
                                 - 0.987998636628556 z_k1_s1054
                                 - 0.490999777524888 z_k1_s1055
                                 - 0.500000540892908 z_k1_s1056
                                 - 0.346000134505909 z_k1_s1057
                                 - 0.662998710438191 z_k1_s1060
                                 - 0.446998708626818 z_k1_s1061
                                 - 0.72299831244236 z_k1_s1062
                                 - 0.31399919235192 z_k1_s1063
                                 - 0.568999174565397 z_k1_s1064
                                 - 0.567000303629098 z_k1_s1065
                                 - 0.955995890194103 z_k1_s1066
                                 - 0.542999506429854 z_k1_s1067
                                 - 0.309999542306736 z_k1_s1070
                                 - 0.411000584863823 z_k1_s1071
                                 - 0.226999955946967 z_k1_s1072
                                 - 0.226999955946967 z_k1_s1073
                                 - 0.238999892745972 z_k1_s1074
                                 - 0.452998812661531 z_k1_s1075
                                 - 0.437999244162504 z_k1_s1076
                                 - 0.514999285122445 z_k1_s1077
                                 - 0.29900131627739 z_k1_s1100
                                 - 0.243000200550171 z_k1_s1101
                                 - 0.412000117732711 z_k1_s1102
                                 - 0.378999760457864 z_k1_s1103
                                 - 0.186000034421071 z_k1_s1104
                                 - 0.764999763731672 z_k1_s1105
                                 - 0.275000129422583 z_k1_s1106
                                 - 0.197000061645983 z_k1_s1107
                                 - 0.041000033647071 z_k1_s1110
                                 - 0.0460000667260115 z_k1_s1111
                                 - 0.0749998361834375 z_k1_s1112
                                 - 0.0360000665681975 z_k1_s1113
                                 - 0.0689997700886821 z_k1_s1114
                                 - 0.0869999683729484 z_k1_s1115
                                 - 0.160000002525024 z_k1_s1116
                                 - 0.146000068304152 z_k1_s1117
                                 - 0.346001064607841 z_k1_s1120
                                 - 0.15900043535666 z_k1_s1121
                                 - 0.394001165455065 z_k1_s1122
                                 - 0.290000271617881 z_k1_s1123
                                 - 0.127000300591691 z_k1_s1124
                                 - 0.148000243412122 z_k1_s1125
                                 - 0.180000378177091 z_k1_s1126
                                 - 0.382001140243259 z_k1_s1127
                                 - 0.0820001398804728 z_k1_s1130
                                 - 0.0430000270787304 z_k1_s1131
                                 - 0.0290000211089145 z_k1_s1132
                                 - 0.0910002224040179 z_k1_s1133
                                 - 0.139000050528876 z_k1_s1134
                                 - 0.0370000332631053 z_k1_s1135
                                 - 0.0840001582190384 z_k1_s1136
                                 - 0.107000369112542 z_k1_s1137
                                 - 0.117000178554931 z_k1_s1140
                                 - 0.0880000435360984 z_k1_s1141
                                 - 0.197000230051396 z_k1_s1142
                                 - 0.208000841831502 z_k1_s1143
                                 - 0.118000079234789 z_k1_s1144
                                 - 0.324999840485277 z_k1_s1145
                                 - 0.107000122955325 z_k1_s1146
                                 - 0.150999816973053 z_k1_s1147
                                 - 0.30900040570909 z_k1_s1150
                                 - 0.087999916459646 z_k1_s1151
                                 - 0.203999738138304 z_k1_s1152
                                 - 0.0849999363576041 z_k1_s1153
                                 - 0.11599995807859 z_k1_s1154
                                 - 0.166999892613166 z_k1_s1155
                                 - 0.108999913573871 z_k1_s1156
                                 - 0.285999285194071 z_k1_s1157
                                 - 0.0730000242788799 z_k1_s1160
                                 - 0.0390000363956059 z_k1_s1161
                                 - 0.111000392972602 z_k1_s1162
                                 - 0.0610001130496329 z_k1_s1163
                                 - 0.105000334757804 z_k1_s1164
                                 - 0.17900036873915 z_k1_s1165
                                 - 0.0689999854690145 z_k1_s1166
                                 - 0.0470001140153368 z_k1_s1167
                                 - 0.230000257114081 z_k1_s1170
                                 - 0.222999969288557 z_k1_s1171
                                 - 0.0480000416585774 z_k1_s1172
                                 - 0.103000555142885 z_k1_s1173
                                 - 0.0780000791952012 z_k1_s1174
                                 - 0.0790001203131332 z_k1_s1175
                                 - 0.0560000946017251 z_k1_s1176
                                 - 0.103000555142885 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.891116 w_k1_ADM_s000
                                 + 1.24348 w_k1_AXP_s000
                                 + 1.41581 w_k1_DIS_s000
                                 + 1.18277 w_k1_FDX_s000
                                 + 1.14812 w_k1_IBM_s000
                                 + 1.26476 w_k1_INTC_s000
                                 + 1.26086 w_k1_KO_s000 + 1.24542 w_k1_MAT_s000
                                 + 1.2395 w_k1_MSFT_s000 + 1.32429 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.11217 w_k1_ADM_s000
                                 + 1.11022 w_k1_AXP_s000
                                 + 0.824727 w_k1_DIS_s000
                                 + 1.00091 w_k1_FDX_s000
                                 + 0.867713 w_k1_IBM_s000
                                 + 0.902372 w_k1_INTC_s000
                                 + 0.863533 w_k1_KO_s000
                                 + 1.04192 w_k1_MAT_s000
                                 + 1.02468 w_k1_MSFT_s000
                                 + 1.07724 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.871332 w_k1_ADM_s000
                                 + 1.20128 w_k1_AXP_s000
                                 + 0.990311 w_k1_DIS_s000
                                 + 0.881456 w_k1_FDX_s000
                                 + 1.36636 w_k1_IBM_s000
                                 + 0.922106 w_k1_INTC_s000
                                 + 1.23061 w_k1_KO_s000 + 1.41757 w_k1_MAT_s000
                                 + 0.985381 w_k1_MSFT_s000
                                 + 1.23819 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.849693 w_k1_ADM_s000
                                 + 1.06334 w_k1_AXP_s000
                                 + 1.02688 w_k1_DIS_s000
                                 + 1.02293 w_k1_FDX_s000
                                 + 0.705788 w_k1_IBM_s000
                                 + 1.08474 w_k1_INTC_s000
                                 + 1.00294 w_k1_KO_s000
                                 + 0.864808 w_k1_MAT_s000
                                 + 1.27227 w_k1_MSFT_s000
                                 + 0.912429 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.651129 w_k1_ADM_s000
                                 + 0.735693 w_k1_AXP_s000
                                 + 0.721238 w_k1_DIS_s000
                                 + 0.780853 w_k1_FDX_s000
                                 + 0.761295 w_k1_IBM_s000
                                 + 0.646405 w_k1_INTC_s000
                                 + 0.972971 w_k1_KO_s000
                                 + 0.800512 w_k1_MAT_s000
                                 + 0.809455 w_k1_MSFT_s000
                                 + 0.985594 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.751296 w_k1_ADM_s000
                                 + 0.979423 w_k1_AXP_s000
                                 + 0.99732 w_k1_DIS_s000
                                 + 0.944806 w_k1_FDX_s000
                                 + 0.97055 w_k1_IBM_s000
                                 + 0.97451 w_k1_INTC_s000
                                 + 1.14326 w_k1_KO_s000
                                 + 0.876219 w_k1_MAT_s000
                                 + 0.948909 w_k1_MSFT_s000
                                 + 1.06912 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.774076 w_k1_ADM_s000
                                 + 0.72214 w_k1_AXP_s000
                                 + 0.708566 w_k1_DIS_s000
                                 + 0.68377 w_k1_FDX_s000
                                 + 1.05794 w_k1_IBM_s000
                                 + 0.723267 w_k1_INTC_s000
                                 + 1.03411 w_k1_KO_s000
                                 + 0.749044 w_k1_MAT_s000
                                 + 0.854104 w_k1_MSFT_s000
                                 + 0.876098 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.990602 w_k1_ADM_s000
                                 + 1.73608 w_k1_AXP_s000
                                 + 1.62106 w_k1_DIS_s000
                                 + 1.32714 w_k1_FDX_s000
                                 + 1.34342 w_k1_IBM_s000
                                 + 1.17995 w_k1_INTC_s000
                                 + 1.52109 w_k1_KO_s000 + 1.8125 w_k1_MAT_s000
                                 + 1.31852 w_k1_MSFT_s000
                                 + 1.40371 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.990435 w_k1_ADM_s001
                                 + 1.0554 w_k1_AXP_s001
                                 + 0.826473 w_k1_DIS_s001
                                 + 0.873513 w_k1_FDX_s001
                                 + 1.37957 w_k1_IBM_s001
                                 + 1.00161 w_k1_INTC_s001
                                 + 0.938142 w_k1_KO_s001
                                 + 1.12531 w_k1_MAT_s001
                                 + 0.887775 w_k1_MSFT_s001
                                 + 0.925569 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.870916 w_k1_ADM_s001
                                 + 0.890045 w_k1_AXP_s001
                                 + 0.93253 w_k1_DIS_s001
                                 + 0.916769 w_k1_FDX_s001
                                 + 1.04236 w_k1_IBM_s001
                                 + 0.749165 w_k1_INTC_s001
                                 + 0.884518 w_k1_KO_s001
                                 + 1.11032 w_k1_MAT_s001
                                 + 0.992918 w_k1_MSFT_s001
                                 + 0.955378 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.15236 w_k1_ADM_s001
                                 + 1.20031 w_k1_AXP_s001
                                 + 1.03057 w_k1_DIS_s001
                                 + 0.971775 w_k1_FDX_s001
                                 + 1.56419 w_k1_IBM_s001
                                 + 0.92943 w_k1_INTC_s001
                                 + 1.13412 w_k1_KO_s001 + 1.66584 w_k1_MAT_s001
                                 + 0.879975 w_k1_MSFT_s001
                                 + 0.978625 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.17314 w_k1_ADM_s001
                                 + 1.1493 w_k1_AXP_s001 + 1.38138 w_k1_DIS_s001
                                 + 1.22145 w_k1_FDX_s001
                                 + 0.966384 w_k1_IBM_s001
                                 + 1.11711 w_k1_INTC_s001
                                 + 1.00132 w_k1_KO_s001
                                 + 0.963372 w_k1_MAT_s001
                                 + 1.24044 w_k1_MSFT_s001
                                 + 1.01443 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.930495 w_k1_ADM_s001
                                 + 0.685362 w_k1_AXP_s001
                                 + 0.65944 w_k1_DIS_s001
                                 + 0.767496 w_k1_FDX_s001
                                 + 0.784775 w_k1_IBM_s001
                                 + 0.861296 w_k1_INTC_s001
                                 + 0.764389 w_k1_KO_s001
                                 + 0.748688 w_k1_MAT_s001
                                 + 0.876037 w_k1_MSFT_s001
                                 + 0.871408 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.30664 w_k1_ADM_s001
                                 + 1.73391 w_k1_AXP_s001
                                 + 1.70461 w_k1_DIS_s001
                                 + 1.36778 w_k1_FDX_s001
                                 + 1.51757 w_k1_IBM_s001
                                 + 1.56368 w_k1_INTC_s001
                                 + 1.31733 w_k1_KO_s001 + 1.60303 w_k1_MAT_s001
                                 + 1.46287 w_k1_MSFT_s001 + 1.2664 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.15072 w_k1_ADM_s001
                                 + 1.31373 w_k1_AXP_s001
                                 + 1.14999 w_k1_DIS_s001
                                 + 1.22881 w_k1_FDX_s001
                                 + 1.12411 w_k1_IBM_s001
                                 + 1.01385 w_k1_INTC_s001
                                 + 1.01684 w_k1_KO_s001 + 1.30485 w_k1_MAT_s001
                                 + 1.18328 w_k1_MSFT_s001
                                 + 1.12253 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.24547 w_k1_ADM_s001
                                 + 1.00082 w_k1_AXP_s001
                                 + 0.941495 w_k1_DIS_s001
                                 + 1.15174 w_k1_FDX_s001
                                 + 0.671014 w_k1_IBM_s001
                                 + 1.20275 w_k1_INTC_s001
                                 + 0.891407 w_k1_KO_s001
                                 + 0.584751 w_k1_MAT_s001
                                 + 1.19235 w_k1_MSFT_s001 + 1.0729 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.882317 w_k1_ADM_s002
                                 + 0.778703 w_k1_AXP_s002
                                 + 1.02919 w_k1_DIS_s002
                                 + 0.879333 w_k1_FDX_s002
                                 + 1.00693 w_k1_IBM_s002
                                 + 1.05791 w_k1_INTC_s002
                                 + 1.02801 w_k1_KO_s002 + 1.09738 w_k1_MAT_s002
                                 + 1.12527 w_k1_MSFT_s002 + 0.969 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.863017 w_k1_ADM_s002
                                 + 0.581113 w_k1_AXP_s002
                                 + 0.694737 w_k1_DIS_s002
                                 + 0.745059 w_k1_FDX_s002
                                 + 1.11602 w_k1_IBM_s002
                                 + 0.896557 w_k1_INTC_s002
                                 + 0.87408 w_k1_KO_s002
                                 + 0.864267 w_k1_MAT_s002
                                 + 1.00636 w_k1_MSFT_s002
                                 + 0.77486 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.881249 w_k1_ADM_s002
                                 + 0.809884 w_k1_AXP_s002
                                 + 0.626334 w_k1_DIS_s002
                                 + 0.796443 w_k1_FDX_s002
                                 + 0.703912 w_k1_IBM_s002
                                 + 0.789916 w_k1_INTC_s002
                                 + 0.661524 w_k1_KO_s002
                                 + 0.881861 w_k1_MAT_s002
                                 + 0.92992 w_k1_MSFT_s002
                                 + 0.773014 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.905411 w_k1_ADM_s002
                                 + 0.823396 w_k1_AXP_s002
                                 + 0.758854 w_k1_DIS_s002
                                 + 0.767602 w_k1_FDX_s002
                                 + 1.1608 w_k1_IBM_s002
                                 + 0.835311 w_k1_INTC_s002
                                 + 0.758457 w_k1_KO_s002
                                 + 1.04163 w_k1_MAT_s002
                                 + 0.769126 w_k1_MSFT_s002
                                 + 0.79891 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.39189 w_k1_ADM_s002
                                 + 1.45775 w_k1_AXP_s002
                                 + 1.22936 w_k1_DIS_s002
                                 + 1.22846 w_k1_FDX_s002
                                 + 1.25071 w_k1_IBM_s002
                                 + 1.21006 w_k1_INTC_s002
                                 + 1.05263 w_k1_KO_s002 + 1.59266 w_k1_MAT_s002
                                 + 1.0785 w_k1_MSFT_s002 + 1.02987 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.07433 w_k1_ADM_s002
                                 + 1.1707 w_k1_AXP_s002 + 1.42031 w_k1_DIS_s002
                                 + 1.1102 w_k1_FDX_s002 + 1.55013 w_k1_IBM_s002
                                 + 1.06036 w_k1_INTC_s002
                                 + 1.30737 w_k1_KO_s002 + 1.46127 w_k1_MAT_s002
                                 + 1.18726 w_k1_MSFT_s002
                                 + 1.03897 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.06436 w_k1_ADM_s002
                                 + 1.15552 w_k1_AXP_s002
                                 + 0.873358 w_k1_DIS_s002
                                 + 1.00088 w_k1_FDX_s002
                                 + 0.864286 w_k1_IBM_s002
                                 + 1.21314 w_k1_INTC_s002
                                 + 0.770818 w_k1_KO_s002
                                 + 1.38344 w_k1_MAT_s002
                                 + 1.12157 w_k1_MSFT_s002
                                 + 0.935964 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.979433 w_k1_ADM_s002
                                 + 1.07832 w_k1_AXP_s002
                                 + 1.45607 w_k1_DIS_s002
                                 + 1.26956 w_k1_FDX_s002
                                 + 0.86821 w_k1_IBM_s002
                                 + 1.38605 w_k1_INTC_s002
                                 + 1.01765 w_k1_KO_s002 + 1.03424 w_k1_MAT_s002
                                 + 1.37179 w_k1_MSFT_s002
                                 + 0.961014 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.54153 w_k1_ADM_s003
                                 + 1.2728 w_k1_AXP_s003 + 1.52448 w_k1_DIS_s003
                                 + 1.21308 w_k1_FDX_s003
                                 + 1.59636 w_k1_IBM_s003
                                 + 0.859341 w_k1_INTC_s003
                                 + 1.28633 w_k1_KO_s003 + 1.48275 w_k1_MAT_s003
                                 + 1.10477 w_k1_MSFT_s003
                                 + 1.14567 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.960867 w_k1_ADM_s003
                                 + 1.23367 w_k1_AXP_s003
                                 + 1.03491 w_k1_DIS_s003
                                 + 1.03421 w_k1_FDX_s003
                                 + 1.09296 w_k1_IBM_s003
                                 + 0.993797 w_k1_INTC_s003
                                 + 0.927466 w_k1_KO_s003
                                 + 0.956652 w_k1_MAT_s003
                                 + 0.89005 w_k1_MSFT_s003
                                 + 1.09565 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.35759 w_k1_ADM_s003
                                 + 1.47411 w_k1_AXP_s003 + 1.4076 w_k1_DIS_s003
                                 + 1.38862 w_k1_FDX_s003
                                 + 1.09267 w_k1_IBM_s003
                                 + 1.4392 w_k1_INTC_s003 + 1.14176 w_k1_KO_s003
                                 + 0.800049 w_k1_MAT_s003
                                 + 1.18369 w_k1_MSFT_s003
                                 + 1.25578 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.854858 w_k1_ADM_s003
                                 + 0.644165 w_k1_AXP_s003
                                 + 0.654287 w_k1_DIS_s003
                                 + 0.930971 w_k1_FDX_s003
                                 + 0.595696 w_k1_IBM_s003
                                 + 1.11844 w_k1_INTC_s003
                                 + 0.728985 w_k1_KO_s003
                                 + 0.321354 w_k1_MAT_s003
                                 + 0.923309 w_k1_MSFT_s003
                                 + 0.934057 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.0519 w_k1_ADM_s003
                                 + 0.899491 w_k1_AXP_s003
                                 + 0.712507 w_k1_DIS_s003
                                 + 0.884282 w_k1_FDX_s003
                                 + 0.836143 w_k1_IBM_s003
                                 + 0.92342 w_k1_INTC_s003
                                 + 0.658689 w_k1_KO_s003
                                 + 0.711606 w_k1_MAT_s003
                                 + 0.749198 w_k1_MSFT_s003
                                 + 0.86304 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.08491 w_k1_ADM_s003
                                 + 0.901227 w_k1_AXP_s003
                                 + 0.940516 w_k1_DIS_s003
                                 + 0.91544 w_k1_FDX_s003
                                 + 1.31438 w_k1_IBM_s003
                                 + 0.849751 w_k1_INTC_s003
                                 + 0.915569 w_k1_KO_s003 + 1.1099 w_k1_MAT_s003
                                 + 0.954076 w_k1_MSFT_s003
                                 + 1.02948 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.824977 w_k1_ADM_s003
                                 + 0.964475 w_k1_AXP_s003
                                 + 1.02613 w_k1_DIS_s003
                                 + 0.943412 w_k1_FDX_s003
                                 + 0.963258 w_k1_IBM_s003
                                 + 1.06135 w_k1_INTC_s003
                                 + 0.909164 w_k1_KO_s003
                                 + 0.747974 w_k1_MAT_s003
                                 + 0.904714 w_k1_MSFT_s003
                                 + 1.02638 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.25209 w_k1_ADM_s003
                                 + 1.04362 w_k1_AXP_s003
                                 + 1.01944 w_k1_DIS_s003
                                 + 1.15761 w_k1_FDX_s003
                                 + 0.859873 w_k1_IBM_s003
                                 + 1.15433 w_k1_INTC_s003
                                 + 0.958445 w_k1_KO_s003
                                 + 0.798467 w_k1_MAT_s003
                                 + 0.954218 w_k1_MSFT_s003
                                 + 1.0258 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.07192 w_k1_ADM_s004
                                 + 1.06662 w_k1_AXP_s004
                                 + 1.32355 w_k1_DIS_s004
                                 + 1.13599 w_k1_FDX_s004
                                 + 0.836145 w_k1_IBM_s004
                                 + 1.11754 w_k1_INTC_s004
                                 + 0.847203 w_k1_KO_s004
                                 + 0.923561 w_k1_MAT_s004
                                 + 1.16709 w_k1_MSFT_s004
                                 + 0.995096 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.94316 w_k1_ADM_s004
                                 + 0.731018 w_k1_AXP_s004
                                 + 0.866109 w_k1_DIS_s004
                                 + 0.88941 w_k1_FDX_s004
                                 + 0.679239 w_k1_IBM_s004
                                 + 0.836452 w_k1_INTC_s004
                                 + 0.845006 w_k1_KO_s004
                                 + 0.690538 w_k1_MAT_s004
                                 + 1.03328 w_k1_MSFT_s004
                                 + 0.890549 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.34605 w_k1_ADM_s004
                                 + 1.43232 w_k1_AXP_s004
                                 + 1.65691 w_k1_DIS_s004
                                 + 1.24747 w_k1_FDX_s004
                                 + 1.03807 w_k1_IBM_s004
                                 + 1.22734 w_k1_INTC_s004
                                 + 1.18873 w_k1_KO_s004 + 1.27149 w_k1_MAT_s004
                                 + 1.22163 w_k1_MSFT_s004
                                 + 1.23811 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.958382 w_k1_ADM_s004
                                 + 1.36411 w_k1_AXP_s004
                                 + 1.29616 w_k1_DIS_s004 + 1.0155 w_k1_FDX_s004
                                 + 1.4385 w_k1_IBM_s004
                                 + 1.09315 w_k1_INTC_s004
                                 + 1.33695 w_k1_KO_s004 + 1.75089 w_k1_MAT_s004
                                 + 1.1642 w_k1_MSFT_s004 + 1.09979 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.57417 w_k1_ADM_s004
                                 + 1.32245 w_k1_AXP_s004 + 1.9883 w_k1_DIS_s004
                                 + 1.48576 w_k1_FDX_s004
                                 + 1.37713 w_k1_IBM_s004
                                 + 1.34962 w_k1_INTC_s004
                                 + 1.31018 w_k1_KO_s004 + 1.87122 w_k1_MAT_s004
                                 + 1.6891 w_k1_MSFT_s004 + 1.3167 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.807715 w_k1_ADM_s004
                                 + 0.752868 w_k1_AXP_s004
                                 + 0.616919 w_k1_DIS_s004
                                 + 0.776522 w_k1_FDX_s004
                                 + 0.826293 w_k1_IBM_s004
                                 + 0.817091 w_k1_INTC_s004
                                 + 0.688876 w_k1_KO_s004
                                 + 0.758712 w_k1_MAT_s004
                                 + 0.763717 w_k1_MSFT_s004
                                 + 0.83539 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.19135 w_k1_ADM_s004
                                 + 0.960264 w_k1_AXP_s004
                                 + 1.01165 w_k1_DIS_s004
                                 + 0.950412 w_k1_FDX_s004
                                 + 1.02859 w_k1_IBM_s004
                                 + 1.03061 w_k1_INTC_s004
                                 + 0.948332 w_k1_KO_s004 + 1.1551 w_k1_MAT_s004
                                 + 0.992024 w_k1_MSFT_s004
                                 + 1.03682 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.940047 w_k1_ADM_s004
                                 + 0.937793 w_k1_AXP_s004
                                 + 1.17961 w_k1_DIS_s004
                                 + 0.96625 w_k1_FDX_s004
                                 + 1.23786 w_k1_IBM_s004
                                 + 0.865178 w_k1_INTC_s004
                                 + 1.04985 w_k1_KO_s004 + 1.21225 w_k1_MAT_s004
                                 + 0.938941 w_k1_MSFT_s004
                                 + 1.12879 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.70452 w_k1_ADM_s005
                                 + 0.7464 w_k1_AXP_s005
                                 + 0.825084 w_k1_DIS_s005
                                 + 0.800405 w_k1_FDX_s005
                                 + 0.924296 w_k1_IBM_s005
                                 + 0.773624 w_k1_INTC_s005
                                 + 0.883898 w_k1_KO_s005
                                 + 1.08134 w_k1_MAT_s005
                                 + 0.837459 w_k1_MSFT_s005
                                 + 0.87488 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.66375 w_k1_ADM_s005
                                 + 1.31459 w_k1_AXP_s005
                                 + 1.66687 w_k1_DIS_s005
                                 + 1.40145 w_k1_FDX_s005 + 1.544 w_k1_IBM_s005
                                 + 1.19423 w_k1_INTC_s005
                                 + 1.54012 w_k1_KO_s005 + 1.75725 w_k1_MAT_s005
                                 + 1.30212 w_k1_MSFT_s005
                                 + 1.38225 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.18999 w_k1_ADM_s005
                                 + 1.19301 w_k1_AXP_s005 + 1.1632 w_k1_DIS_s005
                                 + 1.19242 w_k1_FDX_s005
                                 + 0.796169 w_k1_IBM_s005
                                 + 1.11538 w_k1_INTC_s005
                                 + 1.12129 w_k1_KO_s005 + 1.14815 w_k1_MAT_s005
                                 + 1.03437 w_k1_MSFT_s005 + 1.0976 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.03494 w_k1_ADM_s005
                                 + 1.02955 w_k1_AXP_s005
                                 + 1.15163 w_k1_DIS_s005
                                 + 1.07343 w_k1_FDX_s005
                                 + 1.19698 w_k1_IBM_s005
                                 + 1.09321 w_k1_INTC_s005
                                 + 1.21276 w_k1_KO_s005 + 1.46039 w_k1_MAT_s005
                                 + 1.06385 w_k1_MSFT_s005
                                 + 1.09138 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.14056 w_k1_ADM_s005
                                 + 1.03828 w_k1_AXP_s005
                                 + 0.851818 w_k1_DIS_s005
                                 + 1.0994 w_k1_FDX_s005
                                 + 0.646877 w_k1_IBM_s005
                                 + 1.04813 w_k1_INTC_s005
                                 + 0.935252 w_k1_KO_s005
                                 + 1.04059 w_k1_MAT_s005
                                 + 1.09016 w_k1_MSFT_s005
                                 + 0.925474 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.607382 w_k1_ADM_s005
                                 + 0.646283 w_k1_AXP_s005
                                 + 0.830223 w_k1_DIS_s005
                                 + 0.852772 w_k1_FDX_s005
                                 + 0.589068 w_k1_IBM_s005
                                 + 0.834792 w_k1_INTC_s005
                                 + 0.772384 w_k1_KO_s005
                                 + 0.593542 w_k1_MAT_s005
                                 + 0.897096 w_k1_MSFT_s005
                                 + 0.798867 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.25055 w_k1_ADM_s005
                                 + 1.30895 w_k1_AXP_s005
                                 + 1.64519 w_k1_DIS_s005
                                 + 1.44916 w_k1_FDX_s005
                                 + 1.03816 w_k1_IBM_s005
                                 + 1.49665 w_k1_INTC_s005
                                 + 1.29362 w_k1_KO_s005 + 1.50992 w_k1_MAT_s005
                                 + 1.47933 w_k1_MSFT_s005
                                 + 1.09801 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.871094 w_k1_ADM_s005
                                 + 1.09467 w_k1_AXP_s005
                                 + 0.886486 w_k1_DIS_s005
                                 + 0.830809 w_k1_FDX_s005
                                 + 1.58435 w_k1_IBM_s005
                                 + 0.930959 w_k1_INTC_s005
                                 + 1.17378 w_k1_KO_s005 + 1.91385 w_k1_MAT_s005
                                 + 0.781801 w_k1_MSFT_s005
                                 + 0.990405 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.842843 w_k1_ADM_s006
                                 + 0.707489 w_k1_AXP_s006
                                 + 0.597708 w_k1_DIS_s006
                                 + 0.711294 w_k1_FDX_s006
                                 + 0.655294 w_k1_IBM_s006
                                 + 0.651904 w_k1_INTC_s006
                                 + 0.57772 w_k1_KO_s006
                                 + 0.641278 w_k1_MAT_s006
                                 + 0.652573 w_k1_MSFT_s006
                                 + 0.734471 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.13951 w_k1_ADM_s006
                                 + 1.03222 w_k1_AXP_s006
                                 + 0.835242 w_k1_DIS_s006
                                 + 1.05796 w_k1_FDX_s006
                                 + 0.776187 w_k1_IBM_s006
                                 + 0.953161 w_k1_INTC_s006
                                 + 0.703676 w_k1_KO_s006
                                 + 0.632459 w_k1_MAT_s006
                                 + 0.872844 w_k1_MSFT_s006
                                 + 0.846268 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.802538 w_k1_ADM_s006
                                 + 1.27812 w_k1_AXP_s006
                                 + 1.10594 w_k1_DIS_s006
                                 + 0.991496 w_k1_FDX_s006
                                 + 1.41929 w_k1_IBM_s006
                                 + 1.06819 w_k1_INTC_s006
                                 + 1.10708 w_k1_KO_s006 + 1.27246 w_k1_MAT_s006
                                 + 1.00553 w_k1_MSFT_s006
                                 + 1.05948 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.937304 w_k1_ADM_s006
                                 + 0.994793 w_k1_AXP_s006
                                 + 0.645498 w_k1_DIS_s006
                                 + 0.812572 w_k1_FDX_s006
                                 + 0.756681 w_k1_IBM_s006
                                 + 0.709359 w_k1_INTC_s006
                                 + 0.605368 w_k1_KO_s006
                                 + 0.739001 w_k1_MAT_s006
                                 + 0.759073 w_k1_MSFT_s006
                                 + 0.812987 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.89747 w_k1_ADM_s006
                                 + 0.823814 w_k1_AXP_s006
                                 + 0.815731 w_k1_DIS_s006
                                 + 0.850692 w_k1_FDX_s006
                                 + 0.980836 w_k1_IBM_s006
                                 + 0.913714 w_k1_INTC_s006
                                 + 0.823257 w_k1_KO_s006
                                 + 0.740322 w_k1_MAT_s006
                                 + 0.835315 w_k1_MSFT_s006
                                 + 0.826734 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.58023 w_k1_ADM_s006
                                 + 1.33808 w_k1_AXP_s006
                                 + 1.60557 w_k1_DIS_s006
                                 + 1.41968 w_k1_FDX_s006
                                 + 1.13005 w_k1_IBM_s006
                                 + 1.04334 w_k1_INTC_s006 + 1.2056 w_k1_KO_s006
                                 + 1.3603 w_k1_MAT_s006
                                 + 1.09578 w_k1_MSFT_s006
                                 + 1.25164 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.77992 w_k1_ADM_s006
                                 + 1.7186 w_k1_AXP_s006 + 2.19665 w_k1_DIS_s006
                                 + 1.6261 w_k1_FDX_s006 + 1.61105 w_k1_IBM_s006
                                 + 1.54409 w_k1_INTC_s006
                                 + 1.37837 w_k1_KO_s006 + 1.32504 w_k1_MAT_s006
                                 + 1.39347 w_k1_MSFT_s006
                                 + 1.39489 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.880104 w_k1_ADM_s006
                                 + 0.870603 w_k1_AXP_s006
                                 + 1.22583 w_k1_DIS_s006
                                 + 1.09135 w_k1_FDX_s006
                                 + 0.952003 w_k1_IBM_s006
                                 + 1.32771 w_k1_INTC_s006
                                 + 0.896288 w_k1_KO_s006
                                 + 0.568962 w_k1_MAT_s006
                                 + 1.13362 w_k1_MSFT_s006
                                 + 0.935073 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.901165 w_k1_ADM_s007
                                 + 1.30465 w_k1_AXP_s007
                                 + 0.927158 w_k1_DIS_s007
                                 + 0.946765 w_k1_FDX_s007
                                 + 1.02021 w_k1_IBM_s007
                                 + 1.07534 w_k1_INTC_s007
                                 + 1.02056 w_k1_KO_s007 + 1.33494 w_k1_MAT_s007
                                 + 0.996414 w_k1_MSFT_s007
                                 + 0.835006 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.8181 w_k1_ADM_s007
                                 + 1.28226 w_k1_AXP_s007
                                 + 1.30857 w_k1_DIS_s007
                                 + 0.927726 w_k1_FDX_s007
                                 + 1.59146 w_k1_IBM_s007
                                 + 1.07026 w_k1_INTC_s007
                                 + 1.34635 w_k1_KO_s007 + 1.86374 w_k1_MAT_s007
                                 + 1.09551 w_k1_MSFT_s007
                                 + 1.00629 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.35414 w_k1_ADM_s007
                                 + 1.23965 w_k1_AXP_s007
                                 + 1.34183 w_k1_DIS_s007
                                 + 1.12985 w_k1_FDX_s007
                                 + 0.750731 w_k1_IBM_s007
                                 + 1.32502 w_k1_INTC_s007
                                 + 1.04008 w_k1_KO_s007
                                 + 0.851128 w_k1_MAT_s007
                                 + 1.17224 w_k1_MSFT_s007
                                 + 1.04648 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.939715 w_k1_ADM_s007
                                 + 0.810793 w_k1_AXP_s007
                                 + 0.75761 w_k1_DIS_s007
                                 + 0.856224 w_k1_FDX_s007
                                 + 0.816009 w_k1_IBM_s007
                                 + 0.987431 w_k1_INTC_s007
                                 + 0.886086 w_k1_KO_s007
                                 + 0.936652 w_k1_MAT_s007
                                 + 0.91215 w_k1_MSFT_s007
                                 + 0.882348 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.736513 w_k1_ADM_s007
                                 + 0.817442 w_k1_AXP_s007
                                 + 1.00409 w_k1_DIS_s007
                                 + 0.936052 w_k1_FDX_s007
                                 + 1.05002 w_k1_IBM_s007
                                 + 1.08356 w_k1_INTC_s007
                                 + 1.15211 w_k1_KO_s007 + 1.16979 w_k1_MAT_s007
                                 + 0.969761 w_k1_MSFT_s007
                                 + 0.896808 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.93299 w_k1_ADM_s007
                                 + 0.629196 w_k1_AXP_s007
                                 + 0.940208 w_k1_DIS_s007
                                 + 0.967826 w_k1_FDX_s007
                                 + 0.674831 w_k1_IBM_s007
                                 + 0.899886 w_k1_INTC_s007
                                 + 0.989886 w_k1_KO_s007
                                 + 0.631622 w_k1_MAT_s007
                                 + 0.967166 w_k1_MSFT_s007
                                 + 0.802887 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.26548 w_k1_ADM_s007
                                 + 1.04495 w_k1_AXP_s007
                                 + 1.27219 w_k1_DIS_s007
                                 + 1.09504 w_k1_FDX_s007
                                 + 1.06808 w_k1_IBM_s007 + 1.031 w_k1_INTC_s007
                                 + 1.23597 w_k1_KO_s007
                                 + 0.844795 w_k1_MAT_s007
                                 + 1.22849 w_k1_MSFT_s007
                                 + 1.01059 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.32819 w_k1_ADM_s007
                                 + 1.37 w_k1_AXP_s007 + 1.73867 w_k1_DIS_s007
                                 + 1.24181 w_k1_FDX_s007
                                 + 1.24157 w_k1_IBM_s007
                                 + 1.00887 w_k1_INTC_s007
                                 + 1.52159 w_k1_KO_s007 + 1.40921 w_k1_MAT_s007
                                 + 0.989652 w_k1_MSFT_s007
                                 + 1.21112 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.863106 w_k1_ADM_s010
                                 + 0.974148 w_k1_AXP_s010
                                 + 1.05496 w_k1_DIS_s010
                                 + 0.877448 w_k1_FDX_s010
                                 + 1.20951 w_k1_IBM_s010
                                 + 0.787581 w_k1_INTC_s010
                                 + 1.31789 w_k1_KO_s010 + 1.25872 w_k1_MAT_s010
                                 + 0.875002 w_k1_MSFT_s010
                                 + 1.00585 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.13846 w_k1_ADM_s010
                                 + 1.15604 w_k1_AXP_s010
                                 + 1.24046 w_k1_DIS_s010
                                 + 1.13093 w_k1_FDX_s010
                                 + 0.99498 w_k1_IBM_s010
                                 + 1.08411 w_k1_INTC_s010
                                 + 1.11929 w_k1_KO_s010 + 1.04353 w_k1_MAT_s010
                                 + 1.13346 w_k1_MSFT_s010
                                 + 1.08957 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.822572 w_k1_ADM_s010
                                 + 0.788523 w_k1_AXP_s010
                                 + 0.964052 w_k1_DIS_s010
                                 + 0.860179 w_k1_FDX_s010
                                 + 0.806249 w_k1_IBM_s010
                                 + 1.0022 w_k1_INTC_s010
                                 + 0.996712 w_k1_KO_s010
                                 + 0.81903 w_k1_MAT_s010
                                 + 0.911202 w_k1_MSFT_s010
                                 + 0.880927 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.09742 w_k1_ADM_s010
                                 + 1.34135 w_k1_AXP_s010
                                 + 1.64938 w_k1_DIS_s010
                                 + 1.21078 w_k1_FDX_s010
                                 + 1.42448 w_k1_IBM_s010
                                 + 0.992838 w_k1_INTC_s010
                                 + 1.41785 w_k1_KO_s010 + 1.34202 w_k1_MAT_s010
                                 + 1.0195 w_k1_MSFT_s010 + 1.16576 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.24009 w_k1_ADM_s010
                                 + 1.55561 w_k1_AXP_s010
                                 + 1.46135 w_k1_DIS_s010
                                 + 1.31644 w_k1_FDX_s010
                                 + 1.06144 w_k1_IBM_s010
                                 + 1.34121 w_k1_INTC_s010
                                 + 1.30525 w_k1_KO_s010 + 1.60205 w_k1_MAT_s010
                                 + 1.21458 w_k1_MSFT_s010
                                 + 1.11646 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.829735 w_k1_ADM_s010
                                 + 0.916748 w_k1_AXP_s010
                                 + 0.746978 w_k1_DIS_s010
                                 + 0.895036 w_k1_FDX_s010
                                 + 0.879043 w_k1_IBM_s010
                                 + 0.828448 w_k1_INTC_s010
                                 + 0.943912 w_k1_KO_s010
                                 + 1.24951 w_k1_MAT_s010
                                 + 0.865915 w_k1_MSFT_s010
                                 + 0.918293 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.821071 w_k1_ADM_s010
                                 + 0.602341 w_k1_AXP_s010
                                 + 0.613758 w_k1_DIS_s010
                                 + 0.694353 w_k1_FDX_s010
                                 + 0.895473 w_k1_IBM_s010
                                 + 0.650373 w_k1_INTC_s010
                                 + 0.901758 w_k1_KO_s010
                                 + 0.768787 w_k1_MAT_s010
                                 + 0.748375 w_k1_MSFT_s010
                                 + 0.732576 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.02241 w_k1_ADM_s010
                                 + 0.859374 w_k1_AXP_s010
                                 + 1.14088 w_k1_DIS_s010
                                 + 1.12253 w_k1_FDX_s010
                                 + 0.673512 w_k1_IBM_s010
                                 + 1.03241 w_k1_INTC_s010
                                 + 1.10857 w_k1_KO_s010
                                 + 0.869398 w_k1_MAT_s010
                                 + 1.15221 w_k1_MSFT_s010
                                 + 0.952538 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.989949 w_k1_ADM_s011
                                 + 0.902944 w_k1_AXP_s011
                                 + 0.731618 w_k1_DIS_s011
                                 + 0.927957 w_k1_FDX_s011
                                 + 0.59781 w_k1_IBM_s011
                                 + 0.974734 w_k1_INTC_s011
                                 + 0.724028 w_k1_KO_s011
                                 + 0.609895 w_k1_MAT_s011
                                 + 0.863485 w_k1_MSFT_s011
                                 + 0.892598 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.00343 w_k1_ADM_s011
                                 + 0.817945 w_k1_AXP_s011
                                 + 0.765923 w_k1_DIS_s011
                                 + 0.818858 w_k1_FDX_s011
                                 + 1.10709 w_k1_IBM_s011
                                 + 0.791045 w_k1_INTC_s011
                                 + 0.872503 w_k1_KO_s011
                                 + 1.05764 w_k1_MAT_s011
                                 + 0.820933 w_k1_MSFT_s011
                                 + 0.818821 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.34149 w_k1_ADM_s011
                                 + 1.64008 w_k1_AXP_s011
                                 + 1.40895 w_k1_DIS_s011
                                 + 1.43145 w_k1_FDX_s011
                                 + 1.02386 w_k1_IBM_s011
                                 + 1.28401 w_k1_INTC_s011
                                 + 1.19208 w_k1_KO_s011 + 1.51268 w_k1_MAT_s011
                                 + 1.27486 w_k1_MSFT_s011
                                 + 1.06661 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.887533 w_k1_ADM_s011
                                 + 1.16906 w_k1_AXP_s011
                                 + 0.980056 w_k1_DIS_s011
                                 + 0.924124 w_k1_FDX_s011
                                 + 1.68444 w_k1_IBM_s011
                                 + 1.03713 w_k1_INTC_s011
                                 + 1.23368 w_k1_KO_s011 + 1.4721 w_k1_MAT_s011
                                 + 0.872472 w_k1_MSFT_s011
                                 + 0.976178 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.38832 w_k1_ADM_s011
                                 + 1.16582 w_k1_AXP_s011
                                 + 1.34482 w_k1_DIS_s011
                                 + 1.15329 w_k1_FDX_s011
                                 + 1.28047 w_k1_IBM_s011
                                 + 1.15041 w_k1_INTC_s011
                                 + 1.29573 w_k1_KO_s011 + 1.08188 w_k1_MAT_s011
                                 + 1.12144 w_k1_MSFT_s011
                                 + 1.07565 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.859724 w_k1_ADM_s011
                                 + 0.780249 w_k1_AXP_s011
                                 + 0.64995 w_k1_DIS_s011
                                 + 0.77789 w_k1_FDX_s011
                                 + 0.769409 w_k1_IBM_s011
                                 + 0.874623 w_k1_INTC_s011
                                 + 0.757203 w_k1_KO_s011
                                 + 0.787362 w_k1_MAT_s011
                                 + 0.887961 w_k1_MSFT_s011
                                 + 0.710267 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.957403 w_k1_ADM_s011
                                 + 1.13603 w_k1_AXP_s011 + 1.0014 w_k1_DIS_s011
                                 + 1.10629 w_k1_FDX_s011
                                 + 0.80374 w_k1_IBM_s011
                                 + 1.07319 w_k1_INTC_s011
                                 + 0.94132 w_k1_KO_s011
                                 + 0.993735 w_k1_MAT_s011
                                 + 1.16774 w_k1_MSFT_s011
                                 + 0.985166 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.1915 w_k1_ADM_s011
                                 + 1.07261 w_k1_AXP_s011
                                 + 1.56787 w_k1_DIS_s011 + 1.1959 w_k1_FDX_s011
                                 + 1.01121 w_k1_IBM_s011
                                 + 1.34082 w_k1_INTC_s011
                                 + 1.33558 w_k1_KO_s011 + 1.19381 w_k1_MAT_s011
                                 + 1.28204 w_k1_MSFT_s011
                                 + 1.03786 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.08993 w_k1_ADM_s012
                                 + 0.929233 w_k1_AXP_s012
                                 + 1.18527 w_k1_DIS_s012
                                 + 1.06616 w_k1_FDX_s012
                                 + 0.833456 w_k1_IBM_s012
                                 + 1.23968 w_k1_INTC_s012
                                 + 1.05817 w_k1_KO_s012
                                 + 0.681989 w_k1_MAT_s012
                                 + 1.233 w_k1_MSFT_s012 + 1.04739 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.984605 w_k1_ADM_s012
                                 + 0.985981 w_k1_AXP_s012
                                 + 0.641025 w_k1_DIS_s012
                                 + 0.804241 w_k1_FDX_s012
                                 + 0.846553 w_k1_IBM_s012
                                 + 0.855964 w_k1_INTC_s012
                                 + 0.821863 w_k1_KO_s012
                                 + 0.729913 w_k1_MAT_s012
                                 + 0.792678 w_k1_MSFT_s012
                                 + 0.823679 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.30556 w_k1_ADM_s012
                                 + 1.38018 w_k1_AXP_s012
                                 + 1.16539 w_k1_DIS_s012
                                 + 1.26842 w_k1_FDX_s012
                                 + 1.12872 w_k1_IBM_s012
                                 + 1.48027 w_k1_INTC_s012
                                 + 1.19719 w_k1_KO_s012 + 1.31854 w_k1_MAT_s012
                                 + 1.26082 w_k1_MSFT_s012
                                 + 1.13682 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.14162 w_k1_ADM_s012
                                 + 0.856178 w_k1_AXP_s012
                                 + 0.862225 w_k1_DIS_s012
                                 + 1.02835 w_k1_FDX_s012
                                 + 0.704412 w_k1_IBM_s012
                                 + 0.959394 w_k1_INTC_s012
                                 + 0.858146 w_k1_KO_s012
                                 + 0.461646 w_k1_MAT_s012
                                 + 1.02611 w_k1_MSFT_s012
                                 + 0.967327 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.868775 w_k1_ADM_s012
                                 + 0.870631 w_k1_AXP_s012
                                 + 0.865455 w_k1_DIS_s012
                                 + 0.828364 w_k1_FDX_s012
                                 + 1.27086 w_k1_IBM_s012
                                 + 1.10674 w_k1_INTC_s012
                                 + 1.09445 w_k1_KO_s012
                                 + 0.822575 w_k1_MAT_s012
                                 + 0.879564 w_k1_MSFT_s012
                                 + 1.04781 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.03366 w_k1_ADM_s012
                                 + 1.12155 w_k1_AXP_s012
                                 + 1.09061 w_k1_DIS_s012
                                 + 1.03902 w_k1_FDX_s012
                                 + 1.26817 w_k1_IBM_s012
                                 + 1.19039 w_k1_INTC_s012
                                 + 1.06207 w_k1_KO_s012 + 1.08309 w_k1_MAT_s012
                                 + 1.18831 w_k1_MSFT_s012
                                 + 1.07644 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.33104 w_k1_ADM_s012
                                 + 1.24143 w_k1_AXP_s012
                                 + 1.68499 w_k1_DIS_s012 + 1.4824 w_k1_FDX_s012
                                 + 1.06438 w_k1_IBM_s012
                                 + 1.09511 w_k1_INTC_s012
                                 + 1.14946 w_k1_KO_s012
                                 + 0.823176 w_k1_MAT_s012
                                 + 1.38149 w_k1_MSFT_s012
                                 + 1.21929 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.790123 w_k1_ADM_s012
                                 + 0.764319 w_k1_AXP_s012
                                 + 0.754363 w_k1_DIS_s012
                                 + 0.819043 w_k1_FDX_s012
                                 + 0.797309 w_k1_IBM_s012
                                 + 0.814282 w_k1_INTC_s012
                                 + 0.753869 w_k1_KO_s012
                                 + 0.784332 w_k1_MAT_s012
                                 + 0.877165 w_k1_MSFT_s012
                                 + 0.842283 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.92692 w_k1_ADM_s013
                                 + 1.22403 w_k1_AXP_s013
                                 + 1.19296 w_k1_DIS_s013
                                 + 1.14946 w_k1_FDX_s013 + 1.0431 w_k1_IBM_s013
                                 + 1.35945 w_k1_INTC_s013
                                 + 1.04227 w_k1_KO_s013 + 1.23479 w_k1_MAT_s013
                                 + 1.22294 w_k1_MSFT_s013
                                 + 1.08468 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.61891 w_k1_ADM_s013
                                 + 1.36448 w_k1_AXP_s013
                                 + 1.73809 w_k1_DIS_s013
                                 + 1.37747 w_k1_FDX_s013
                                 + 1.26175 w_k1_IBM_s013
                                 + 1.54657 w_k1_INTC_s013 + 1.5066 w_k1_KO_s013
                                 + 1.42707 w_k1_MAT_s013
                                 + 1.40551 w_k1_MSFT_s013 + 1.2666 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.00028 w_k1_ADM_s013
                                 + 1.07741 w_k1_AXP_s013
                                 + 0.946608 w_k1_DIS_s013
                                 + 0.930038 w_k1_FDX_s013
                                 + 1.12154 w_k1_IBM_s013
                                 + 1.21254 w_k1_INTC_s013
                                 + 1.04967 w_k1_KO_s013 + 1.2073 w_k1_MAT_s013
                                 + 1.11622 w_k1_MSFT_s013
                                 + 1.05536 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.49001 w_k1_ADM_s013
                                 + 1.14229 w_k1_AXP_s013
                                 + 1.34199 w_k1_DIS_s013 + 1.4404 w_k1_FDX_s013
                                 + 0.777362 w_k1_IBM_s013
                                 + 1.58277 w_k1_INTC_s013
                                 + 0.960099 w_k1_KO_s013
                                 + 0.655318 w_k1_MAT_s013
                                 + 1.34584 w_k1_MSFT_s013
                                 + 1.14266 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.00682 w_k1_ADM_s013
                                 + 0.960506 w_k1_AXP_s013
                                 + 0.885677 w_k1_DIS_s013
                                 + 1.00375 w_k1_FDX_s013
                                 + 0.840341 w_k1_IBM_s013
                                 + 1.0236 w_k1_INTC_s013
                                 + 0.842525 w_k1_KO_s013
                                 + 0.858037 w_k1_MAT_s013
                                 + 0.985212 w_k1_MSFT_s013
                                 + 0.883874 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.863389 w_k1_ADM_s013
                                 + 0.717124 w_k1_AXP_s013
                                 + 0.551186 w_k1_DIS_s013
                                 + 0.752236 w_k1_FDX_s013
                                 + 1.01044 w_k1_IBM_s013
                                 + 1.03705 w_k1_INTC_s013
                                 + 0.848808 w_k1_KO_s013
                                 + 0.876146 w_k1_MAT_s013
                                 + 0.913919 w_k1_MSFT_s013
                                 + 0.812692 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.991286 w_k1_ADM_s013
                                 + 0.966714 w_k1_AXP_s013
                                 + 0.981692 w_k1_DIS_s013
                                 + 0.923612 w_k1_FDX_s013
                                 + 1.52195 w_k1_IBM_s013
                                 + 1.01568 w_k1_INTC_s013
                                 + 1.06106 w_k1_KO_s013 + 1.4514 w_k1_MAT_s013
                                 + 0.934321 w_k1_MSFT_s013
                                 + 1.1069 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.894513 w_k1_ADM_s013
                                 + 0.700738 w_k1_AXP_s013
                                 + 0.702916 w_k1_DIS_s013
                                 + 0.874467 w_k1_FDX_s013
                                 + 0.700936 w_k1_IBM_s013
                                 + 1.13778 w_k1_INTC_s013
                                 + 0.794024 w_k1_KO_s013
                                 + 0.643694 w_k1_MAT_s013
                                 + 0.963665 w_k1_MSFT_s013
                                 + 0.91382 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.2011 w_k1_ADM_s014
                                 + 1.37921 w_k1_AXP_s014
                                 + 1.10724 w_k1_DIS_s014 + 1.1628 w_k1_FDX_s014
                                 + 1.03623 w_k1_IBM_s014 + 1.219 w_k1_INTC_s014
                                 + 1.15502 w_k1_KO_s014 + 1.34424 w_k1_MAT_s014
                                 + 1.11967 w_k1_MSFT_s014
                                 + 1.12372 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.95238 w_k1_ADM_s014
                                 + 1.18036 w_k1_AXP_s014
                                 + 1.01897 w_k1_DIS_s014
                                 + 0.941186 w_k1_FDX_s014
                                 + 1.22083 w_k1_IBM_s014
                                 + 1.1368 w_k1_INTC_s014 + 1.24872 w_k1_KO_s014
                                 + 1.26775 w_k1_MAT_s014
                                 + 0.89363 w_k1_MSFT_s014
                                 + 1.02984 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.22754 w_k1_ADM_s014
                                 + 1.22098 w_k1_AXP_s014
                                 + 1.49203 w_k1_DIS_s014
                                 + 1.27845 w_k1_FDX_s014
                                 + 1.03156 w_k1_IBM_s014
                                 + 1.48968 w_k1_INTC_s014
                                 + 1.22431 w_k1_KO_s014
                                 + 0.895115 w_k1_MAT_s014
                                 + 1.06593 w_k1_MSFT_s014
                                 + 1.07888 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.52023 w_k1_ADM_s014
                                 + 1.39588 w_k1_AXP_s014
                                 + 1.68332 w_k1_DIS_s014
                                 + 1.33023 w_k1_FDX_s014
                                 + 1.53347 w_k1_IBM_s014
                                 + 1.13728 w_k1_INTC_s014
                                 + 1.59963 w_k1_KO_s014 + 1.82078 w_k1_MAT_s014
                                 + 1.33688 w_k1_MSFT_s014
                                 + 1.39193 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.8324 w_k1_ADM_s014
                                 + 0.867433 w_k1_AXP_s014
                                 + 0.686056 w_k1_DIS_s014
                                 + 0.825741 w_k1_FDX_s014
                                 + 0.792367 w_k1_IBM_s014
                                 + 0.907989 w_k1_INTC_s014
                                 + 0.767218 w_k1_KO_s014
                                 + 0.798592 w_k1_MAT_s014
                                 + 0.752027 w_k1_MSFT_s014
                                 + 0.904161 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.2963 w_k1_ADM_s014
                                 + 1.0846 w_k1_AXP_s014 + 1.15759 w_k1_DIS_s014
                                 + 1.12112 w_k1_FDX_s014
                                 + 0.786433 w_k1_IBM_s014
                                 + 1.13684 w_k1_INTC_s014
                                 + 1.04716 w_k1_KO_s014 + 0.79422 w_k1_MAT_s014
                                 + 1.03683 w_k1_MSFT_s014
                                 + 1.06006 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.941405 w_k1_ADM_s014
                                 + 0.787608 w_k1_AXP_s014
                                 + 0.759803 w_k1_DIS_s014
                                 + 0.934656 w_k1_FDX_s014
                                 + 0.763803 w_k1_IBM_s014
                                 + 0.933605 w_k1_INTC_s014
                                 + 0.89209 w_k1_KO_s014
                                 + 0.847349 w_k1_MAT_s014
                                 + 0.823705 w_k1_MSFT_s014
                                 + 0.930926 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.929246 w_k1_ADM_s014
                                 + 0.761638 w_k1_AXP_s014
                                 + 0.675043 w_k1_DIS_s014
                                 + 0.775702 w_k1_FDX_s014
                                 + 1.00689 w_k1_IBM_s014
                                 + 0.959452 w_k1_INTC_s014
                                 + 0.878809 w_k1_KO_s014
                                 + 0.609903 w_k1_MAT_s014
                                 + 0.835 w_k1_MSFT_s014 + 0.813639 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.917733 w_k1_ADM_s015
                                 + 0.975544 w_k1_AXP_s015
                                 + 0.797625 w_k1_DIS_s015
                                 + 0.908743 w_k1_FDX_s015
                                 + 0.780975 w_k1_IBM_s015
                                 + 0.930946 w_k1_INTC_s015
                                 + 0.773894 w_k1_KO_s015
                                 + 0.974939 w_k1_MAT_s015
                                 + 0.810383 w_k1_MSFT_s015
                                 + 0.952485 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.02741 w_k1_ADM_s015
                                 + 0.992376 w_k1_AXP_s015
                                 + 1.12295 w_k1_DIS_s015
                                 + 1.11815 w_k1_FDX_s015
                                 + 0.922706 w_k1_IBM_s015
                                 + 1.04512 w_k1_INTC_s015
                                 + 0.906359 w_k1_KO_s015
                                 + 1.17676 w_k1_MAT_s015
                                 + 1.06515 w_k1_MSFT_s015 + 1.0061 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.07661 w_k1_ADM_s015
                                 + 1.50027 w_k1_AXP_s015
                                 + 1.80451 w_k1_DIS_s015
                                 + 1.44299 w_k1_FDX_s015
                                 + 1.57288 w_k1_IBM_s015
                                 + 1.15813 w_k1_INTC_s015
                                 + 1.39521 w_k1_KO_s015 + 2.15079 w_k1_MAT_s015
                                 + 1.22095 w_k1_MSFT_s015
                                 + 1.33112 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.03974 w_k1_ADM_s015
                                 + 1.04411 w_k1_AXP_s015
                                 + 0.990369 w_k1_DIS_s015
                                 + 0.929734 w_k1_FDX_s015
                                 + 1.39823 w_k1_IBM_s015
                                 + 0.953214 w_k1_INTC_s015
                                 + 1.00695 w_k1_KO_s015 + 1.95363 w_k1_MAT_s015
                                 + 0.957928 w_k1_MSFT_s015
                                 + 1.09717 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.704547 w_k1_ADM_s015
                                 + 0.720252 w_k1_AXP_s015
                                 + 0.767292 w_k1_DIS_s015
                                 + 0.798007 w_k1_FDX_s015
                                 + 0.968139 w_k1_IBM_s015
                                 + 0.993759 w_k1_INTC_s015
                                 + 0.829537 w_k1_KO_s015
                                 + 0.880141 w_k1_MAT_s015
                                 + 0.88645 w_k1_MSFT_s015
                                 + 1.02197 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.790463 w_k1_ADM_s015
                                 + 0.633263 w_k1_AXP_s015
                                 + 0.545125 w_k1_DIS_s015
                                 + 0.711878 w_k1_FDX_s015
                                 + 0.70679 w_k1_IBM_s015
                                 + 0.707034 w_k1_INTC_s015
                                 + 0.639667 w_k1_KO_s015
                                 + 0.655301 w_k1_MAT_s015
                                 + 0.767813 w_k1_MSFT_s015
                                 + 0.814195 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.837489 w_k1_ADM_s015
                                 + 0.710515 w_k1_AXP_s015
                                 + 0.698979 w_k1_DIS_s015
                                 + 0.757114 w_k1_FDX_s015
                                 + 1.12226 w_k1_IBM_s015
                                 + 0.776387 w_k1_INTC_s015
                                 + 0.835274 w_k1_KO_s015
                                 + 1.29857 w_k1_MAT_s015
                                 + 0.791299 w_k1_MSFT_s015
                                 + 0.963215 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.24593 w_k1_ADM_s015
                                 + 1.42472 w_k1_AXP_s015
                                 + 1.60943 w_k1_DIS_s015
                                 + 1.35251 w_k1_FDX_s015
                                 + 1.03886 w_k1_IBM_s015
                                 + 1.21262 w_k1_INTC_s015 + 1.2738 w_k1_KO_s015
                                 + 1.30302 w_k1_MAT_s015
                                 + 1.28202 w_k1_MSFT_s015
                                 + 1.25051 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.28277 w_k1_ADM_s016
                                 + 1.36249 w_k1_AXP_s016
                                 + 1.79823 w_k1_DIS_s016
                                 + 1.27633 w_k1_FDX_s016
                                 + 1.16287 w_k1_IBM_s016
                                 + 0.956564 w_k1_INTC_s016
                                 + 1.26617 w_k1_KO_s016
                                 + 0.989432 w_k1_MAT_s016
                                 + 1.18523 w_k1_MSFT_s016
                                 + 1.21267 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.725405 w_k1_ADM_s016
                                 + 0.730632 w_k1_AXP_s016
                                 + 0.62074 w_k1_DIS_s016
                                 + 0.678678 w_k1_FDX_s016
                                 + 0.830108 w_k1_IBM_s016
                                 + 0.766842 w_k1_INTC_s016
                                 + 0.755487 w_k1_KO_s016
                                 + 0.840274 w_k1_MAT_s016
                                 + 0.648985 w_k1_MSFT_s016
                                 + 0.861047 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.875386 w_k1_ADM_s016
                                 + 0.770742 w_k1_AXP_s016
                                 + 0.899121 w_k1_DIS_s016
                                 + 0.840226 w_k1_FDX_s016
                                 + 0.745983 w_k1_IBM_s016
                                 + 0.989709 w_k1_INTC_s016
                                 + 0.858885 w_k1_KO_s016
                                 + 0.390272 w_k1_MAT_s016
                                 + 0.791649 w_k1_MSFT_s016
                                 + 0.809778 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.40959 w_k1_ADM_s016
                                 + 1.37576 w_k1_AXP_s016
                                 + 1.45256 w_k1_DIS_s016
                                 + 1.24238 w_k1_FDX_s016
                                 + 1.51947 w_k1_IBM_s016
                                 + 1.67853 w_k1_INTC_s016
                                 + 1.34619 w_k1_KO_s016 + 1.04086 w_k1_MAT_s016
                                 + 1.33033 w_k1_MSFT_s016
                                 + 1.23531 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.02406 w_k1_ADM_s016
                                 + 1.22616 w_k1_AXP_s016
                                 + 1.01267 w_k1_DIS_s016
                                 + 0.926669 w_k1_FDX_s016
                                 + 1.63396 w_k1_IBM_s016
                                 + 1.17515 w_k1_INTC_s016
                                 + 1.19993 w_k1_KO_s016 + 1.47923 w_k1_MAT_s016
                                 + 0.905082 w_k1_MSFT_s016
                                 + 1.22559 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.01742 w_k1_ADM_s016
                                 + 1.10313 w_k1_AXP_s016
                                 + 0.930317 w_k1_DIS_s016
                                 + 0.899791 w_k1_FDX_s016
                                 + 1.17369 w_k1_IBM_s016
                                 + 1.22068 w_k1_INTC_s016
                                 + 1.12259 w_k1_KO_s016 + 1.05484 w_k1_MAT_s016
                                 + 0.966518 w_k1_MSFT_s016
                                 + 1.0165 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.908929 w_k1_ADM_s016
                                 + 0.908449 w_k1_AXP_s016
                                 + 0.694189 w_k1_DIS_s016
                                 + 0.80491 w_k1_FDX_s016
                                 + 0.86754 w_k1_IBM_s016
                                 + 0.970943 w_k1_INTC_s016
                                 + 0.753142 w_k1_KO_s016
                                 + 0.767606 w_k1_MAT_s016
                                 + 0.86765 w_k1_MSFT_s016
                                 + 0.901881 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.0486 w_k1_ADM_s016
                                 + 1.15051 w_k1_AXP_s016
                                 + 1.10207 w_k1_DIS_s016
                                 + 1.14048 w_k1_FDX_s016
                                 + 0.889789 w_k1_IBM_s016
                                 + 1.32086 w_k1_INTC_s016
                                 + 0.961411 w_k1_KO_s016
                                 + 0.812274 w_k1_MAT_s016
                                 + 1.16922 w_k1_MSFT_s016
                                 + 1.02693 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.05186 w_k1_ADM_s017
                                 + 1.11564 w_k1_AXP_s017
                                 + 0.99269 w_k1_DIS_s017
                                 + 1.10725 w_k1_FDX_s017
                                 + 0.643905 w_k1_IBM_s017
                                 + 0.94087 w_k1_INTC_s017
                                 + 0.782023 w_k1_KO_s017
                                 + 0.702092 w_k1_MAT_s017
                                 + 1.21293 w_k1_MSFT_s017
                                 + 0.948652 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.982335 w_k1_ADM_s017
                                 + 0.926382 w_k1_AXP_s017
                                 + 1.03369 w_k1_DIS_s017
                                 + 0.974811 w_k1_FDX_s017
                                 + 0.893384 w_k1_IBM_s017
                                 + 0.857484 w_k1_INTC_s017
                                 + 0.901915 w_k1_KO_s017
                                 + 0.842392 w_k1_MAT_s017
                                 + 1.00277 w_k1_MSFT_s017
                                 + 0.991384 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.95058 w_k1_ADM_s017
                                 + 1.10295 w_k1_AXP_s017
                                 + 0.809129 w_k1_DIS_s017
                                 + 0.833755 w_k1_FDX_s017
                                 + 1.53131 w_k1_IBM_s017
                                 + 0.806069 w_k1_INTC_s017
                                 + 0.974088 w_k1_KO_s017
                                 + 1.32548 w_k1_MAT_s017
                                 + 1.0148 w_k1_MSFT_s017
                                 + 0.959338 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.32969 w_k1_ADM_s017
                                 + 1.29076 w_k1_AXP_s017
                                 + 1.56911 w_k1_DIS_s017
                                 + 1.29418 w_k1_FDX_s017
                                 + 0.924769 w_k1_IBM_s017
                                 + 1.16278 w_k1_INTC_s017
                                 + 1.19275 w_k1_KO_s017
                                 + 0.736677 w_k1_MAT_s017
                                 + 1.33786 w_k1_MSFT_s017
                                 + 1.17837 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.576 w_k1_ADM_s017
                                 + 1.50713 w_k1_AXP_s017 + 1.8227 w_k1_DIS_s017
                                 + 1.38985 w_k1_FDX_s017
                                 + 1.39427 w_k1_IBM_s017
                                 + 1.11634 w_k1_INTC_s017
                                 + 1.54866 w_k1_KO_s017 + 1.45858 w_k1_MAT_s017
                                 + 1.44795 w_k1_MSFT_s017
                                 + 1.24853 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.724128 w_k1_ADM_s017
                                 + 0.651422 w_k1_AXP_s017
                                 + 0.588991 w_k1_DIS_s017
                                 + 0.686755 w_k1_FDX_s017
                                 + 0.769085 w_k1_IBM_s017
                                 + 0.679133 w_k1_INTC_s017
                                 + 0.665932 w_k1_KO_s017
                                 + 0.587363 w_k1_MAT_s017
                                 + 0.766492 w_k1_MSFT_s017
                                 + 0.835412 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.08817 w_k1_ADM_s017
                                 + 1.1095 w_k1_AXP_s017 + 1.00812 w_k1_DIS_s017
                                 + 0.996724 w_k1_FDX_s017
                                 + 1.16554 w_k1_IBM_s017
                                 + 1.01125 w_k1_INTC_s017
                                 + 1.05175 w_k1_KO_s017 + 1.21515 w_k1_MAT_s017
                                 + 1.05417 w_k1_MSFT_s017
                                 + 1.03071 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.0809 w_k1_ADM_s017
                                 + 1.01489 w_k1_AXP_s017
                                 + 1.05228 w_k1_DIS_s017
                                 + 1.08001 w_k1_FDX_s017
                                 + 1.29496 w_k1_IBM_s017
                                 + 1.16199 w_k1_INTC_s017 + 1.064 w_k1_KO_s017
                                 + 1.06471 w_k1_MAT_s017
                                 + 1.1782 w_k1_MSFT_s017 + 1.04331 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.14446 w_k1_ADM_s100
                                 + 1.22152 w_k1_AXP_s100
                                 + 0.824945 w_k1_DIS_s100
                                 + 0.945659 w_k1_FDX_s100
                                 + 1.13522 w_k1_IBM_s100
                                 + 0.923513 w_k1_INTC_s100
                                 + 0.825209 w_k1_KO_s100
                                 + 1.16997 w_k1_MAT_s100
                                 + 0.908066 w_k1_MSFT_s100
                                 + 0.945388 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.21405 w_k1_ADM_s100
                                 + 1.06338 w_k1_AXP_s100
                                 + 1.29291 w_k1_DIS_s100
                                 + 1.08724 w_k1_FDX_s100
                                 + 0.922122 w_k1_IBM_s100
                                 + 1.1057 w_k1_INTC_s100 + 1.02019 w_k1_KO_s100
                                 + 0.844574 w_k1_MAT_s100
                                 + 1.10025 w_k1_MSFT_s100
                                 + 1.04685 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.08098 w_k1_ADM_s100
                                 + 0.877321 w_k1_AXP_s100
                                 + 0.915833 w_k1_DIS_s100
                                 + 0.876458 w_k1_FDX_s100
                                 + 1.02517 w_k1_IBM_s100
                                 + 0.887504 w_k1_INTC_s100
                                 + 0.910193 w_k1_KO_s100
                                 + 1.00779 w_k1_MAT_s100
                                 + 0.899527 w_k1_MSFT_s100
                                 + 0.952029 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.18052 w_k1_ADM_s100
                                 + 1.39176 w_k1_AXP_s100
                                 + 1.07549 w_k1_DIS_s100
                                 + 1.25171 w_k1_FDX_s100
                                 + 0.999013 w_k1_IBM_s100
                                 + 1.11303 w_k1_INTC_s100
                                 + 0.996377 w_k1_KO_s100
                                 + 1.27094 w_k1_MAT_s100
                                 + 1.09889 w_k1_MSFT_s100
                                 + 1.00618 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.04349 w_k1_ADM_s100
                                 + 0.80148 w_k1_AXP_s100
                                 + 0.841008 w_k1_DIS_s100
                                 + 0.902205 w_k1_FDX_s100
                                 + 0.729281 w_k1_IBM_s100
                                 + 0.746223 w_k1_INTC_s100
                                 + 0.766834 w_k1_KO_s100
                                 + 0.752909 w_k1_MAT_s100
                                 + 0.823918 w_k1_MSFT_s100
                                 + 0.915573 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.06646 w_k1_ADM_s100
                                 + 1.47466 w_k1_AXP_s100
                                 + 1.20155 w_k1_DIS_s100
                                 + 1.06851 w_k1_FDX_s100
                                 + 1.47698 w_k1_IBM_s100
                                 + 1.11323 w_k1_INTC_s100
                                 + 1.11573 w_k1_KO_s100 + 2.04733 w_k1_MAT_s100
                                 + 0.937103 w_k1_MSFT_s100
                                 + 1.06534 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.45504 w_k1_ADM_s100
                                 + 1.27697 w_k1_AXP_s100
                                 + 1.49439 w_k1_DIS_s100
                                 + 1.31185 w_k1_FDX_s100
                                 + 0.981477 w_k1_IBM_s100
                                 + 1.02477 w_k1_INTC_s100
                                 + 1.04568 w_k1_KO_s100 + 1.1386 w_k1_MAT_s100
                                 + 1.21028 w_k1_MSFT_s100
                                 + 1.24302 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.57781 w_k1_ADM_s100
                                 + 1.58757 w_k1_AXP_s100
                                 + 1.76604 w_k1_DIS_s100 + 1.4236 w_k1_FDX_s100
                                 + 1.36185 w_k1_IBM_s100
                                 + 1.33258 w_k1_INTC_s100
                                 + 1.45525 w_k1_KO_s100 + 1.88733 w_k1_MAT_s100
                                 + 1.46079 w_k1_MSFT_s100
                                 + 1.40727 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.04633 w_k1_ADM_s101
                                 + 1.29598 w_k1_AXP_s101
                                 + 1.06245 w_k1_DIS_s101
                                 + 1.00388 w_k1_FDX_s101
                                 + 1.50331 w_k1_IBM_s101
                                 + 1.04458 w_k1_INTC_s101
                                 + 1.18159 w_k1_KO_s101 + 1.74475 w_k1_MAT_s101
                                 + 0.904541 w_k1_MSFT_s101
                                 + 1.04942 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.838807 w_k1_ADM_s101
                                 + 1.01056 w_k1_AXP_s101
                                 + 0.750572 w_k1_DIS_s101
                                 + 0.909137 w_k1_FDX_s101
                                 + 1.18842 w_k1_IBM_s101
                                 + 0.896467 w_k1_INTC_s101
                                 + 0.966492 w_k1_KO_s101
                                 + 1.21575 w_k1_MAT_s101
                                 + 0.925322 w_k1_MSFT_s101
                                 + 0.981999 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.929604 w_k1_ADM_s101
                                 + 0.92632 w_k1_AXP_s101
                                 + 0.887859 w_k1_DIS_s101
                                 + 0.946943 w_k1_FDX_s101
                                 + 0.792756 w_k1_IBM_s101
                                 + 0.989967 w_k1_INTC_s101
                                 + 0.961738 w_k1_KO_s101
                                 + 0.886396 w_k1_MAT_s101
                                 + 0.978637 w_k1_MSFT_s101
                                 + 0.907766 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.07651 w_k1_ADM_s101
                                 + 0.932185 w_k1_AXP_s101
                                 + 0.760658 w_k1_DIS_s101
                                 + 1.12464 w_k1_FDX_s101
                                 + 0.569685 w_k1_IBM_s101
                                 + 0.879877 w_k1_INTC_s101
                                 + 0.766254 w_k1_KO_s101
                                 + 0.646995 w_k1_MAT_s101
                                 + 1.041 w_k1_MSFT_s101 + 0.999582 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.26002 w_k1_ADM_s101
                                 + 1.15518 w_k1_AXP_s101
                                 + 1.02493 w_k1_DIS_s101
                                 + 1.22283 w_k1_FDX_s101
                                 + 0.931571 w_k1_IBM_s101
                                 + 1.05492 w_k1_INTC_s101
                                 + 0.990585 w_k1_KO_s101
                                 + 0.851097 w_k1_MAT_s101
                                 + 1.10945 w_k1_MSFT_s101
                                 + 0.947165 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.25685 w_k1_ADM_s101
                                 + 1.22649 w_k1_AXP_s101
                                 + 1.41685 w_k1_DIS_s101
                                 + 1.34075 w_k1_FDX_s101 + 1.1419 w_k1_IBM_s101
                                 + 1.19246 w_k1_INTC_s101
                                 + 1.45665 w_k1_KO_s101 + 1.07814 w_k1_MAT_s101
                                 + 1.21557 w_k1_MSFT_s101
                                 + 1.17458 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.912539 w_k1_ADM_s101
                                 + 0.654289 w_k1_AXP_s101
                                 + 0.595069 w_k1_DIS_s101
                                 + 0.696959 w_k1_FDX_s101
                                 + 0.811583 w_k1_IBM_s101
                                 + 0.710175 w_k1_INTC_s101
                                 + 0.759718 w_k1_KO_s101
                                 + 0.836024 w_k1_MAT_s101
                                 + 0.716754 w_k1_MSFT_s101
                                 + 0.8123 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.46362 w_k1_ADM_s101
                                 + 1.66204 w_k1_AXP_s101
                                 + 1.53641 w_k1_DIS_s101
                                 + 1.45391 w_k1_FDX_s101
                                 + 1.66781 w_k1_IBM_s101
                                 + 1.2226 w_k1_INTC_s101 + 1.45542 w_k1_KO_s101
                                 + 1.87058 w_k1_MAT_s101
                                 + 1.31957 w_k1_MSFT_s101
                                 + 1.28903 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.05416 w_k1_ADM_s102
                                 + 1.20271 w_k1_AXP_s102
                                 + 1.41708 w_k1_DIS_s102
                                 + 1.17862 w_k1_FDX_s102
                                 + 1.41608 w_k1_IBM_s102
                                 + 1.11597 w_k1_INTC_s102
                                 + 1.39508 w_k1_KO_s102 + 1.82458 w_k1_MAT_s102
                                 + 1.11624 w_k1_MSFT_s102
                                 + 1.13724 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.22867 w_k1_ADM_s102
                                 + 1.14474 w_k1_AXP_s102
                                 + 1.32171 w_k1_DIS_s102
                                 + 1.24448 w_k1_FDX_s102
                                 + 1.08379 w_k1_IBM_s102
                                 + 1.20438 w_k1_INTC_s102
                                 + 1.30208 w_k1_KO_s102 + 1.05653 w_k1_MAT_s102
                                 + 1.14817 w_k1_MSFT_s102
                                 + 0.99795 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.802277 w_k1_ADM_s102
                                 + 0.770941 w_k1_AXP_s102
                                 + 0.720234 w_k1_DIS_s102
                                 + 0.782224 w_k1_FDX_s102
                                 + 0.812081 w_k1_IBM_s102
                                 + 0.879832 w_k1_INTC_s102
                                 + 0.869131 w_k1_KO_s102
                                 + 0.923046 w_k1_MAT_s102
                                 + 0.833974 w_k1_MSFT_s102
                                 + 0.812458 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.906474 w_k1_ADM_s102
                                 + 0.790849 w_k1_AXP_s102
                                 + 0.807298 w_k1_DIS_s102
                                 + 0.957257 w_k1_FDX_s102
                                 + 0.574482 w_k1_IBM_s102
                                 + 1.01727 w_k1_INTC_s102
                                 + 0.814511 w_k1_KO_s102
                                 + 0.471164 w_k1_MAT_s102
                                 + 0.985863 w_k1_MSFT_s102
                                 + 0.823403 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.001 w_k1_ADM_s102
                                 + 1.06315 w_k1_AXP_s102
                                 + 0.969886 w_k1_DIS_s102
                                 + 1.01136 w_k1_FDX_s102
                                 + 0.79407 w_k1_IBM_s102
                                 + 1.04467 w_k1_INTC_s102
                                 + 0.911121 w_k1_KO_s102
                                 + 0.942462 w_k1_MAT_s102
                                 + 0.874621 w_k1_MSFT_s102
                                 + 0.908911 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.840132 w_k1_ADM_s102
                                 + 0.922979 w_k1_AXP_s102
                                 + 0.903055 w_k1_DIS_s102
                                 + 0.889288 w_k1_FDX_s102
                                 + 1.23827 w_k1_IBM_s102
                                 + 0.937022 w_k1_INTC_s102
                                 + 1.19392 w_k1_KO_s102 + 1.17682 w_k1_MAT_s102
                                 + 0.876757 w_k1_MSFT_s102
                                 + 0.942315 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.08437 w_k1_ADM_s102
                                 + 1.59572 w_k1_AXP_s102
                                 + 1.88393 w_k1_DIS_s102
                                 + 1.45379 w_k1_FDX_s102
                                 + 1.34612 w_k1_IBM_s102
                                 + 1.29404 w_k1_INTC_s102
                                 + 1.48432 w_k1_KO_s102 + 1.88863 w_k1_MAT_s102
                                 + 1.13488 w_k1_MSFT_s102
                                 + 1.28056 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.22142 w_k1_ADM_s102
                                 + 1.12043 w_k1_AXP_s102
                                 + 1.22157 w_k1_DIS_s102
                                 + 1.07944 w_k1_FDX_s102
                                 + 0.910413 w_k1_IBM_s102
                                 + 1.21764 w_k1_INTC_s102
                                 + 1.05044 w_k1_KO_s102 + 1.01274 w_k1_MAT_s102
                                 + 1.12309 w_k1_MSFT_s102
                                 + 0.975689 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.698627 w_k1_ADM_s103
                                 + 0.691568 w_k1_AXP_s103
                                 + 0.628066 w_k1_DIS_s103
                                 + 0.705569 w_k1_FDX_s103
                                 + 0.749911 w_k1_IBM_s103
                                 + 0.840302 w_k1_INTC_s103
                                 + 0.717892 w_k1_KO_s103
                                 + 0.718562 w_k1_MAT_s103
                                 + 0.796213 w_k1_MSFT_s103
                                 + 0.781872 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.929079 w_k1_ADM_s103
                                 + 0.889799 w_k1_AXP_s103
                                 + 0.862205 w_k1_DIS_s103
                                 + 0.929757 w_k1_FDX_s103
                                 + 0.772243 w_k1_IBM_s103
                                 + 0.856489 w_k1_INTC_s103
                                 + 0.759257 w_k1_KO_s103
                                 + 0.478716 w_k1_MAT_s103
                                 + 0.842629 w_k1_MSFT_s103
                                 + 0.8504 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.2909 w_k1_ADM_s103
                                 + 1.7145 w_k1_AXP_s103 + 1.82089 w_k1_DIS_s103
                                 + 1.54117 w_k1_FDX_s103
                                 + 1.17658 w_k1_IBM_s103
                                 + 1.21803 w_k1_INTC_s103
                                 + 1.11729 w_k1_KO_s103 + 1.19272 w_k1_MAT_s103
                                 + 1.18168 w_k1_MSFT_s103
                                 + 1.18984 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.1167 w_k1_ADM_s103
                                 + 1.03934 w_k1_AXP_s103
                                 + 1.30933 w_k1_DIS_s103
                                 + 1.07327 w_k1_FDX_s103
                                 + 1.27908 w_k1_IBM_s103
                                 + 1.0987 w_k1_INTC_s103 + 1.08355 w_k1_KO_s103
                                 + 0.885879 w_k1_MAT_s103
                                 + 1.09983 w_k1_MSFT_s103
                                 + 1.14127 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.930881 w_k1_ADM_s103
                                 + 0.833628 w_k1_AXP_s103
                                 + 0.88922 w_k1_DIS_s103
                                 + 0.912251 w_k1_FDX_s103
                                 + 1.32276 w_k1_IBM_s103
                                 + 0.952041 w_k1_INTC_s103
                                 + 0.96353 w_k1_KO_s103
                                 + 0.690118 w_k1_MAT_s103
                                 + 0.843872 w_k1_MSFT_s103
                                 + 0.931664 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.11298 w_k1_ADM_s103
                                 + 1.533 w_k1_AXP_s103 + 1.17211 w_k1_DIS_s103
                                 + 1.23006 w_k1_FDX_s103
                                 + 1.13753 w_k1_IBM_s103
                                 + 1.01016 w_k1_INTC_s103 + 1.1518 w_k1_KO_s103
                                 + 1.3359 w_k1_MAT_s103
                                 + 1.21481 w_k1_MSFT_s103
                                 + 1.15302 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.03216 w_k1_ADM_s103
                                 + 1.09693 w_k1_AXP_s103
                                 + 1.18037 w_k1_DIS_s103
                                 + 1.13668 w_k1_FDX_s103
                                 + 0.905433 w_k1_IBM_s103
                                 + 1.16141 w_k1_INTC_s103
                                 + 1.05074 w_k1_KO_s103
                                 + 0.910739 w_k1_MAT_s103
                                 + 1.16997 w_k1_MSFT_s103
                                 + 1.02694 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.854185 w_k1_ADM_s103
                                 + 0.781357 w_k1_AXP_s103
                                 + 0.973415 w_k1_DIS_s103
                                 + 0.815151 w_k1_FDX_s103
                                 + 0.935427 w_k1_IBM_s103
                                 + 0.925036 w_k1_INTC_s103
                                 + 0.93127 w_k1_KO_s103
                                 + 0.761878 w_k1_MAT_s103
                                 + 0.880074 w_k1_MSFT_s103
                                 + 0.939945 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.992396 w_k1_ADM_s104
                                 + 0.880792 w_k1_AXP_s104
                                 + 1.22809 w_k1_DIS_s104
                                 + 1.10031 w_k1_FDX_s104
                                 + 1.03888 w_k1_IBM_s104
                                 + 0.885645 w_k1_INTC_s104
                                 + 1.1382 w_k1_KO_s104 + 1.11952 w_k1_MAT_s104
                                 + 1.12267 w_k1_MSFT_s104
                                 + 1.00147 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.78766 w_k1_ADM_s104
                                 + 0.649689 w_k1_AXP_s104
                                 + 0.706385 w_k1_DIS_s104
                                 + 0.776318 w_k1_FDX_s104
                                 + 0.871511 w_k1_IBM_s104
                                 + 0.751522 w_k1_INTC_s104
                                 + 0.778028 w_k1_KO_s104
                                 + 1.11753 w_k1_MAT_s104
                                 + 0.856396 w_k1_MSFT_s104
                                 + 0.854452 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.07908 w_k1_ADM_s104
                                 + 1.14122 w_k1_AXP_s104
                                 + 1.01453 w_k1_DIS_s104 + 1.1035 w_k1_FDX_s104
                                 + 1.07237 w_k1_IBM_s104
                                 + 1.11398 w_k1_INTC_s104
                                 + 1.01619 w_k1_KO_s104 + 1.05769 w_k1_MAT_s104
                                 + 1.04666 w_k1_MSFT_s104
                                 + 1.02567 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.886913 w_k1_ADM_s104
                                 + 0.791162 w_k1_AXP_s104
                                 + 0.85426 w_k1_DIS_s104
                                 + 0.977112 w_k1_FDX_s104
                                 + 0.779574 w_k1_IBM_s104
                                 + 0.827989 w_k1_INTC_s104
                                 + 0.836522 w_k1_KO_s104
                                 + 1.15477 w_k1_MAT_s104
                                 + 0.963713 w_k1_MSFT_s104
                                 + 0.940923 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.22277 w_k1_ADM_s104
                                 + 1.44707 w_k1_AXP_s104 + 1.2497 w_k1_DIS_s104
                                 + 1.21915 w_k1_FDX_s104
                                 + 1.24883 w_k1_IBM_s104
                                 + 1.09102 w_k1_INTC_s104
                                 + 1.17885 w_k1_KO_s104 + 1.8703 w_k1_MAT_s104
                                 + 1.18118 w_k1_MSFT_s104 + 1.1738 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.10021 w_k1_ADM_s104
                                 + 1.51142 w_k1_AXP_s104
                                 + 1.56844 w_k1_DIS_s104
                                 + 1.38432 w_k1_FDX_s104 + 1.3449 w_k1_IBM_s104
                                 + 1.22853 w_k1_INTC_s104
                                 + 1.32417 w_k1_KO_s104 + 2.19977 w_k1_MAT_s104
                                 + 1.37323 w_k1_MSFT_s104
                                 + 1.22278 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.853134 w_k1_ADM_s104
                                 + 0.93604 w_k1_AXP_s104 + 1.444 w_k1_DIS_s104
                                 + 1.22201 w_k1_FDX_s104
                                 + 1.26951 w_k1_IBM_s104
                                 + 1.03373 w_k1_INTC_s104
                                 + 1.37657 w_k1_KO_s104 + 1.67111 w_k1_MAT_s104
                                 + 1.2426 w_k1_MSFT_s104 + 1.1739 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.836818 w_k1_ADM_s104
                                 + 0.96988 w_k1_AXP_s104
                                 + 0.88266 w_k1_DIS_s104
                                 + 0.936007 w_k1_FDX_s104
                                 + 1.2806 w_k1_IBM_s104
                                 + 0.872018 w_k1_INTC_s104
                                 + 1.02577 w_k1_KO_s104 + 1.73392 w_k1_MAT_s104
                                 + 0.915812 w_k1_MSFT_s104
                                 + 0.948354 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.60914 w_k1_ADM_s105
                                 + 1.37929 w_k1_AXP_s105
                                 + 1.58576 w_k1_DIS_s105
                                 + 1.57512 w_k1_FDX_s105
                                 + 0.985426 w_k1_IBM_s105
                                 + 1.50677 w_k1_INTC_s105
                                 + 1.25182 w_k1_KO_s105
                                 + 0.872161 w_k1_MAT_s105
                                 + 1.56081 w_k1_MSFT_s105
                                 + 1.21062 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.21886 w_k1_ADM_s105
                                 + 1.50389 w_k1_AXP_s105
                                 + 1.06788 w_k1_DIS_s105
                                 + 1.13488 w_k1_FDX_s105
                                 + 1.09755 w_k1_IBM_s105
                                 + 0.920516 w_k1_INTC_s105
                                 + 1.10027 w_k1_KO_s105 + 1.74308 w_k1_MAT_s105
                                 + 1.01865 w_k1_MSFT_s105
                                 + 1.05363 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.20594 w_k1_ADM_s105
                                 + 1.0437 w_k1_AXP_s105 + 1.05903 w_k1_DIS_s105
                                 + 1.15593 w_k1_FDX_s105
                                 + 0.772842 w_k1_IBM_s105
                                 + 1.09278 w_k1_INTC_s105
                                 + 0.90445 w_k1_KO_s105 + 1.01716 w_k1_MAT_s105
                                 + 1.08623 w_k1_MSFT_s105
                                 + 0.862831 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.09647 w_k1_ADM_s105
                                 + 1.09903 w_k1_AXP_s105
                                 + 1.03965 w_k1_DIS_s105
                                 + 0.983863 w_k1_FDX_s105
                                 + 1.35686 w_k1_IBM_s105
                                 + 1.06454 w_k1_INTC_s105
                                 + 1.27056 w_k1_KO_s105 + 1.29605 w_k1_MAT_s105
                                 + 0.930768 w_k1_MSFT_s105
                                 + 1.06382 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.971878 w_k1_ADM_s105
                                 + 0.766864 w_k1_AXP_s105
                                 + 0.827974 w_k1_DIS_s105
                                 + 0.905573 w_k1_FDX_s105
                                 + 0.908295 w_k1_IBM_s105
                                 + 0.89893 w_k1_INTC_s105
                                 + 0.897095 w_k1_KO_s105
                                 + 0.887484 w_k1_MAT_s105
                                 + 0.922149 w_k1_MSFT_s105
                                 + 0.898862 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.819985 w_k1_ADM_s105
                                 + 0.80065 w_k1_AXP_s105
                                 + 0.591164 w_k1_DIS_s105
                                 + 0.713861 w_k1_FDX_s105
                                 + 0.876162 w_k1_IBM_s105
                                 + 0.756391 w_k1_INTC_s105
                                 + 0.812984 w_k1_KO_s105
                                 + 0.914716 w_k1_MAT_s105
                                 + 0.768639 w_k1_MSFT_s105
                                 + 0.765251 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.31666 w_k1_ADM_s105
                                 + 1.53101 w_k1_AXP_s105
                                 + 1.65575 w_k1_DIS_s105
                                 + 1.36216 w_k1_FDX_s105
                                 + 1.37765 w_k1_IBM_s105
                                 + 1.10541 w_k1_INTC_s105
                                 + 1.48617 w_k1_KO_s105 + 1.72244 w_k1_MAT_s105
                                 + 1.21177 w_k1_MSFT_s105 + 1.2217 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.0533 w_k1_ADM_s105
                                 + 1.02391 w_k1_AXP_s105
                                 + 0.796299 w_k1_DIS_s105
                                 + 0.998962 w_k1_FDX_s105
                                 + 0.732735 w_k1_IBM_s105
                                 + 0.955289 w_k1_INTC_s105
                                 + 0.840637 w_k1_KO_s105
                                 + 0.945714 w_k1_MAT_s105
                                 + 0.917039 w_k1_MSFT_s105
                                 + 0.831325 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.00107 w_k1_ADM_s106
                                 + 0.827024 w_k1_AXP_s106
                                 + 0.720976 w_k1_DIS_s106
                                 + 0.83317 w_k1_FDX_s106
                                 + 0.933928 w_k1_IBM_s106
                                 + 0.757922 w_k1_INTC_s106
                                 + 0.823991 w_k1_KO_s106
                                 + 0.732935 w_k1_MAT_s106
                                 + 0.756335 w_k1_MSFT_s106
                                 + 0.877496 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.47512 w_k1_ADM_s106
                                 + 1.29746 w_k1_AXP_s106
                                 + 1.30089 w_k1_DIS_s106
                                 + 1.30964 w_k1_FDX_s106
                                 + 0.93634 w_k1_IBM_s106
                                 + 1.19384 w_k1_INTC_s106
                                 + 1.11796 w_k1_KO_s106
                                 + 0.970671 w_k1_MAT_s106
                                 + 1.20428 w_k1_MSFT_s106
                                 + 1.17006 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.10976 w_k1_ADM_s106
                                 + 1.06083 w_k1_AXP_s106
                                 + 1.09231 w_k1_DIS_s106
                                 + 0.915904 w_k1_FDX_s106
                                 + 1.51193 w_k1_IBM_s106
                                 + 0.747591 w_k1_INTC_s106
                                 + 1.14104 w_k1_KO_s106 + 1.29358 w_k1_MAT_s106
                                 + 0.820664 w_k1_MSFT_s106
                                 + 0.991024 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.92472 w_k1_ADM_s106
                                 + 0.685538 w_k1_AXP_s106
                                 + 0.700516 w_k1_DIS_s106
                                 + 0.910821 w_k1_FDX_s106
                                 + 0.576897 w_k1_IBM_s106
                                 + 0.795422 w_k1_INTC_s106
                                 + 0.687722 w_k1_KO_s106
                                 + 0.506771 w_k1_MAT_s106
                                 + 0.881034 w_k1_MSFT_s106
                                 + 0.828097 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.01294 w_k1_ADM_s106
                                 + 0.987233 w_k1_AXP_s106
                                 + 0.945407 w_k1_DIS_s106
                                 + 0.980121 w_k1_FDX_s106
                                 + 1.09555 w_k1_IBM_s106
                                 + 1.0268 w_k1_INTC_s106
                                 + 0.996314 w_k1_KO_s106
                                 + 0.875746 w_k1_MAT_s106
                                 + 0.924946 w_k1_MSFT_s106
                                 + 1.02364 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.735057 w_k1_ADM_s106
                                 + 0.759754 w_k1_AXP_s106
                                 + 0.614703 w_k1_DIS_s106
                                 + 0.736023 w_k1_FDX_s106
                                 + 0.798882 w_k1_IBM_s106
                                 + 0.895657 w_k1_INTC_s106
                                 + 0.841599 w_k1_KO_s106
                                 + 0.785993 w_k1_MAT_s106
                                 + 0.767186 w_k1_MSFT_s106
                                 + 0.812267 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.14669 w_k1_ADM_s106
                                 + 1.51373 w_k1_AXP_s106
                                 + 1.32163 w_k1_DIS_s106 + 1.2515 w_k1_FDX_s106
                                 + 1.27627 w_k1_IBM_s106
                                 + 1.07741 w_k1_INTC_s106
                                 + 1.29856 w_k1_KO_s106 + 1.41084 w_k1_MAT_s106
                                 + 1.12375 w_k1_MSFT_s106 + 1.2972 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.14871 w_k1_ADM_s106
                                 + 1.17742 w_k1_AXP_s106
                                 + 0.914202 w_k1_DIS_s106
                                 + 1.13717 w_k1_FDX_s106
                                 + 0.791663 w_k1_IBM_s106
                                 + 1.01506 w_k1_INTC_s106
                                 + 0.780684 w_k1_KO_s106
                                 + 0.826631 w_k1_MAT_s106
                                 + 0.971429 w_k1_MSFT_s106
                                 + 0.960071 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.09175 w_k1_ADM_s107
                                 + 1.31863 w_k1_AXP_s107
                                 + 1.01675 w_k1_DIS_s107
                                 + 1.03955 w_k1_FDX_s107
                                 + 1.24387 w_k1_IBM_s107
                                 + 1.02461 w_k1_INTC_s107 + 1.1287 w_k1_KO_s107
                                 + 1.6157 w_k1_MAT_s107
                                 + 1.17986 w_k1_MSFT_s107
                                 + 0.992022 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.1246 w_k1_ADM_s107
                                 + 1.12399 w_k1_AXP_s107
                                 + 1.03953 w_k1_DIS_s107
                                 + 1.06779 w_k1_FDX_s107
                                 + 0.927336 w_k1_IBM_s107
                                 + 1.04366 w_k1_INTC_s107 + 1.0997 w_k1_KO_s107
                                 + 1.09632 w_k1_MAT_s107
                                 + 0.995757 w_k1_MSFT_s107
                                 + 1.11657 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.859597 w_k1_ADM_s107
                                 + 0.840919 w_k1_AXP_s107
                                 + 0.65823 w_k1_DIS_s107
                                 + 0.80173 w_k1_FDX_s107
                                 + 0.640676 w_k1_IBM_s107
                                 + 0.682817 w_k1_INTC_s107
                                 + 0.666977 w_k1_KO_s107
                                 + 0.811442 w_k1_MAT_s107
                                 + 0.71368 w_k1_MSFT_s107
                                 + 0.771713 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.84052 w_k1_ADM_s107
                                 + 0.983866 w_k1_AXP_s107
                                 + 0.762016 w_k1_DIS_s107
                                 + 0.723815 w_k1_FDX_s107
                                 + 0.98012 w_k1_IBM_s107
                                 + 0.8116 w_k1_INTC_s107
                                 + 0.965087 w_k1_KO_s107
                                 + 1.01576 w_k1_MAT_s107
                                 + 0.739386 w_k1_MSFT_s107
                                 + 0.89054 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.00909 w_k1_ADM_s107
                                 + 1.01128 w_k1_AXP_s107
                                 + 1.31229 w_k1_DIS_s107
                                 + 1.05921 w_k1_FDX_s107
                                 + 1.69363 w_k1_IBM_s107
                                 + 1.03098 w_k1_INTC_s107
                                 + 1.45014 w_k1_KO_s107 + 1.36715 w_k1_MAT_s107
                                 + 1.00023 w_k1_MSFT_s107
                                 + 1.11749 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.962082 w_k1_ADM_s107
                                 + 0.890575 w_k1_AXP_s107
                                 + 0.87248 w_k1_DIS_s107
                                 + 0.867669 w_k1_FDX_s107
                                 + 0.810884 w_k1_IBM_s107
                                 + 1.00986 w_k1_INTC_s107
                                 + 0.899215 w_k1_KO_s107
                                 + 0.640698 w_k1_MAT_s107
                                 + 0.944761 w_k1_MSFT_s107
                                 + 0.925422 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.936318 w_k1_ADM_s107
                                 + 1.04436 w_k1_AXP_s107
                                 + 0.937317 w_k1_DIS_s107
                                 + 1.06929 w_k1_FDX_s107
                                 + 0.694061 w_k1_IBM_s107
                                 + 0.972504 w_k1_INTC_s107
                                 + 0.828823 w_k1_KO_s107
                                 + 0.865397 w_k1_MAT_s107
                                 + 0.951089 w_k1_MSFT_s107
                                 + 0.994827 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.54053 w_k1_ADM_s107
                                 + 1.54377 w_k1_AXP_s107
                                 + 1.51677 w_k1_DIS_s107
                                 + 1.31249 w_k1_FDX_s107 + 1.0764 w_k1_IBM_s107
                                 + 1.48094 w_k1_INTC_s107
                                 + 1.40038 w_k1_KO_s107 + 1.21156 w_k1_MAT_s107
                                 + 1.34521 w_k1_MSFT_s107
                                 + 1.20593 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.11929 w_k1_ADM_s110
                                 + 1.27483 w_k1_AXP_s110
                                 + 1.36526 w_k1_DIS_s110 + 1.3327 w_k1_FDX_s110
                                 + 0.885781 w_k1_IBM_s110
                                 + 0.911266 w_k1_INTC_s110
                                 + 1.00023 w_k1_KO_s110
                                 + 0.895147 w_k1_MAT_s110
                                 + 1.21577 w_k1_MSFT_s110
                                 + 1.07648 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.17203 w_k1_ADM_s110
                                 + 1.74756 w_k1_AXP_s110
                                 + 1.47574 w_k1_DIS_s110
                                 + 1.24859 w_k1_FDX_s110
                                 + 1.64972 w_k1_IBM_s110
                                 + 1.0264 w_k1_INTC_s110 + 1.12771 w_k1_KO_s110
                                 + 1.63201 w_k1_MAT_s110
                                 + 1.1494 w_k1_MSFT_s110 + 1.1817 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.859036 w_k1_ADM_s110
                                 + 0.841328 w_k1_AXP_s110
                                 + 1.01486 w_k1_DIS_s110
                                 + 1.06256 w_k1_FDX_s110
                                 + 0.727277 w_k1_IBM_s110
                                 + 0.896034 w_k1_INTC_s110
                                 + 0.90777 w_k1_KO_s110
                                 + 0.718014 w_k1_MAT_s110
                                 + 1.04884 w_k1_MSFT_s110 + 1.0776 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.915778 w_k1_ADM_s110
                                 + 1.01299 w_k1_AXP_s110
                                 + 1.04607 w_k1_DIS_s110
                                 + 0.917914 w_k1_FDX_s110
                                 + 1.35639 w_k1_IBM_s110
                                 + 0.906525 w_k1_INTC_s110
                                 + 1.07937 w_k1_KO_s110 + 1.10385 w_k1_MAT_s110
                                 + 0.927099 w_k1_MSFT_s110
                                 + 0.981392 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.989859 w_k1_ADM_s110
                                 + 1.43024 w_k1_AXP_s110
                                 + 1.03566 w_k1_DIS_s110
                                 + 1.02313 w_k1_FDX_s110 + 1.3849 w_k1_IBM_s110
                                 + 0.979034 w_k1_INTC_s110
                                 + 1.15097 w_k1_KO_s110 + 1.57972 w_k1_MAT_s110
                                 + 0.946612 w_k1_MSFT_s110
                                 + 1.19544 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.727296 w_k1_ADM_s110
                                 + 0.862901 w_k1_AXP_s110
                                 + 0.653398 w_k1_DIS_s110
                                 + 0.826535 w_k1_FDX_s110
                                 + 0.918162 w_k1_IBM_s110
                                 + 0.744925 w_k1_INTC_s110
                                 + 0.782206 w_k1_KO_s110
                                 + 0.976282 w_k1_MAT_s110
                                 + 0.782844 w_k1_MSFT_s110
                                 + 0.906939 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.795908 w_k1_ADM_s110
                                 + 1.38873 w_k1_AXP_s110
                                 + 0.974248 w_k1_DIS_s110
                                 + 1.19598 w_k1_FDX_s110
                                 + 0.862526 w_k1_IBM_s110
                                 + 1.09471 w_k1_INTC_s110
                                 + 0.898537 w_k1_KO_s110
                                 + 0.977135 w_k1_MAT_s110
                                 + 0.941804 w_k1_MSFT_s110
                                 + 1.0089 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.990602 w_k1_ADM_s110
                                 + 1.25967 w_k1_AXP_s110
                                 + 1.29656 w_k1_DIS_s110
                                 + 1.20128 w_k1_FDX_s110
                                 + 1.16321 w_k1_IBM_s110
                                 + 1.1183 w_k1_INTC_s110 + 1.2541 w_k1_KO_s110
                                 + 1.29346 w_k1_MAT_s110
                                 + 1.10189 w_k1_MSFT_s110
                                 + 1.17228 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.39069 w_k1_ADM_s111
                                 + 1.12888 w_k1_AXP_s111
                                 + 1.16323 w_k1_DIS_s111
                                 + 1.07317 w_k1_FDX_s111
                                 + 0.914484 w_k1_IBM_s111
                                 + 1.0736 w_k1_INTC_s111 + 1.0118 w_k1_KO_s111
                                 + 0.669354 w_k1_MAT_s111
                                 + 0.870831 w_k1_MSFT_s111
                                 + 0.980268 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.24998 w_k1_ADM_s111
                                 + 1.51674 w_k1_AXP_s111
                                 + 1.71941 w_k1_DIS_s111
                                 + 1.46771 w_k1_FDX_s111
                                 + 1.24319 w_k1_IBM_s111
                                 + 1.15488 w_k1_INTC_s111
                                 + 1.38431 w_k1_KO_s111 + 1.96933 w_k1_MAT_s111
                                 + 1.31615 w_k1_MSFT_s111
                                 + 1.31075 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.862418 w_k1_ADM_s111
                                 + 0.8569 w_k1_AXP_s111 + 1.06718 w_k1_DIS_s111
                                 + 0.98192 w_k1_FDX_s111
                                 + 1.15033 w_k1_IBM_s111
                                 + 1.16888 w_k1_INTC_s111
                                 + 1.36169 w_k1_KO_s111
                                 + 0.954119 w_k1_MAT_s111
                                 + 1.02836 w_k1_MSFT_s111
                                 + 0.959597 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.23599 w_k1_ADM_s111
                                 + 1.57 w_k1_AXP_s111 + 1.08983 w_k1_DIS_s111
                                 + 1.06083 w_k1_FDX_s111
                                 + 1.75151 w_k1_IBM_s111
                                 + 1.21312 w_k1_INTC_s111
                                 + 1.37179 w_k1_KO_s111 + 1.69312 w_k1_MAT_s111
                                 + 1.07249 w_k1_MSFT_s111
                                 + 1.09167 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.963933 w_k1_ADM_s111
                                 + 0.999677 w_k1_AXP_s111
                                 + 0.972947 w_k1_DIS_s111
                                 + 1.10975 w_k1_FDX_s111
                                 + 0.719549 w_k1_IBM_s111
                                 + 0.975188 w_k1_INTC_s111
                                 + 0.895266 w_k1_KO_s111
                                 + 0.622314 w_k1_MAT_s111
                                 + 0.88406 w_k1_MSFT_s111
                                 + 0.878975 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.958957 w_k1_ADM_s111
                                 + 1.21297 w_k1_AXP_s111
                                 + 1.22981 w_k1_DIS_s111
                                 + 1.22552 w_k1_FDX_s111
                                 + 1.07761 w_k1_IBM_s111
                                 + 1.06099 w_k1_INTC_s111
                                 + 1.17371 w_k1_KO_s111 + 1.246 w_k1_MAT_s111
                                 + 1.2706 w_k1_MSFT_s111 + 1.04852 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.850707 w_k1_ADM_s111
                                 + 0.757421 w_k1_AXP_s111
                                 + 0.657469 w_k1_DIS_s111
                                 + 0.767432 w_k1_FDX_s111
                                 + 0.915406 w_k1_IBM_s111
                                 + 0.840376 w_k1_INTC_s111
                                 + 0.916123 w_k1_KO_s111
                                 + 0.772814 w_k1_MAT_s111
                                 + 0.777352 w_k1_MSFT_s111
                                 + 0.88927 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.1159 w_k1_ADM_s111
                                 + 1.00296 w_k1_AXP_s111
                                 + 0.85807 w_k1_DIS_s111
                                 + 0.952106 w_k1_FDX_s111
                                 + 1.01006 w_k1_IBM_s111
                                 + 0.772661 w_k1_INTC_s111
                                 + 0.942401 w_k1_KO_s111
                                 + 1.14007 w_k1_MAT_s111
                                 + 0.844948 w_k1_MSFT_s111
                                 + 0.856324 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.46005 w_k1_ADM_s112
                                 + 1.49154 w_k1_AXP_s112
                                 + 1.50798 w_k1_DIS_s112
                                 + 1.34388 w_k1_FDX_s112
                                 + 1.13614 w_k1_IBM_s112
                                 + 1.14225 w_k1_INTC_s112
                                 + 1.21606 w_k1_KO_s112 + 1.04701 w_k1_MAT_s112
                                 + 1.13637 w_k1_MSFT_s112 + 1.145 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.08545 w_k1_ADM_s112
                                 + 1.15248 w_k1_AXP_s112
                                 + 1.43251 w_k1_DIS_s112
                                 + 1.14915 w_k1_FDX_s112
                                 + 1.56223 w_k1_IBM_s112
                                 + 1.0268 w_k1_INTC_s112 + 1.31855 w_k1_KO_s112
                                 + 1.38831 w_k1_MAT_s112
                                 + 1.04628 w_k1_MSFT_s112
                                 + 1.19246 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.07508 w_k1_ADM_s112
                                 + 0.824797 w_k1_AXP_s112
                                 + 1.08346 w_k1_DIS_s112
                                 + 1.02672 w_k1_FDX_s112
                                 + 0.978754 w_k1_IBM_s112
                                 + 0.86759 w_k1_INTC_s112
                                 + 0.976284 w_k1_KO_s112
                                 + 0.799192 w_k1_MAT_s112
                                 + 1.07565 w_k1_MSFT_s112
                                 + 0.970497 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.97148 w_k1_ADM_s112
                                 + 1.18165 w_k1_AXP_s112
                                 + 1.09333 w_k1_DIS_s112
                                 + 1.10959 w_k1_FDX_s112
                                 + 1.13396 w_k1_IBM_s112
                                 + 1.19545 w_k1_INTC_s112
                                 + 0.930889 w_k1_KO_s112
                                 + 1.16454 w_k1_MAT_s112
                                 + 1.01507 w_k1_MSFT_s112
                                 + 0.963343 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.897514 w_k1_ADM_s112
                                 + 0.800984 w_k1_AXP_s112
                                 + 0.996721 w_k1_DIS_s112
                                 + 0.954027 w_k1_FDX_s112
                                 + 1.39305 w_k1_IBM_s112
                                 + 1.02651 w_k1_INTC_s112
                                 + 1.23068 w_k1_KO_s112 + 1.12879 w_k1_MAT_s112
                                 + 1.04823 w_k1_MSFT_s112
                                 + 0.970836 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.878202 w_k1_ADM_s112
                                 + 1.03094 w_k1_AXP_s112
                                 + 0.758621 w_k1_DIS_s112
                                 + 0.886008 w_k1_FDX_s112
                                 + 0.948301 w_k1_IBM_s112
                                 + 0.920279 w_k1_INTC_s112
                                 + 0.803328 w_k1_KO_s112
                                 + 1.12398 w_k1_MAT_s112
                                 + 0.789613 w_k1_MSFT_s112
                                 + 0.935705 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.688412 w_k1_ADM_s112
                                 + 0.69158 w_k1_AXP_s112
                                 + 0.725821 w_k1_DIS_s112
                                 + 0.715849 w_k1_FDX_s112
                                 + 0.892047 w_k1_IBM_s112
                                 + 0.819066 w_k1_INTC_s112
                                 + 0.844196 w_k1_KO_s112
                                 + 0.960749 w_k1_MAT_s112
                                 + 0.788972 w_k1_MSFT_s112
                                 + 0.823226 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.719277 w_k1_ADM_s112
                                 + 0.814022 w_k1_AXP_s112
                                 + 0.851888 w_k1_DIS_s112
                                 + 1.00351 w_k1_FDX_s112
                                 + 0.648317 w_k1_IBM_s112
                                 + 0.924636 w_k1_INTC_s112
                                 + 0.665498 w_k1_KO_s112
                                 + 0.513376 w_k1_MAT_s112
                                 + 0.893851 w_k1_MSFT_s112
                                 + 0.853036 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 2.05875 w_k1_ADM_s113
                                 + 1.50495 w_k1_AXP_s113
                                 + 1.06648 w_k1_DIS_s113
                                 + 1.28545 w_k1_FDX_s113
                                 + 0.735844 w_k1_IBM_s113
                                 + 1.4053 w_k1_INTC_s113 + 1.03833 w_k1_KO_s113
                                 + 1.0106 w_k1_MAT_s113
                                 + 1.06524 w_k1_MSFT_s113
                                 + 1.22273 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.17177 w_k1_ADM_s113
                                 + 0.840176 w_k1_AXP_s113
                                 + 0.62453 w_k1_DIS_s113
                                 + 0.733053 w_k1_FDX_s113
                                 + 0.988729 w_k1_IBM_s113
                                 + 0.790451 w_k1_INTC_s113
                                 + 0.839952 w_k1_KO_s113 + 1.1068 w_k1_MAT_s113
                                 + 0.856903 w_k1_MSFT_s113
                                 + 0.892036 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.13283 w_k1_ADM_s113
                                 + 0.652367 w_k1_AXP_s113
                                 + 0.66782 w_k1_DIS_s113
                                 + 0.773396 w_k1_FDX_s113
                                 + 0.851289 w_k1_IBM_s113
                                 + 0.694956 w_k1_INTC_s113
                                 + 0.816369 w_k1_KO_s113
                                 + 0.683383 w_k1_MAT_s113
                                 + 0.803477 w_k1_MSFT_s113
                                 + 0.91735 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.46386 w_k1_ADM_s113
                                 + 0.804178 w_k1_AXP_s113
                                 + 0.894811 w_k1_DIS_s113
                                 + 0.853366 w_k1_FDX_s113
                                 + 0.953072 w_k1_IBM_s113
                                 + 0.87921 w_k1_INTC_s113 + 1.0155 w_k1_KO_s113
                                 + 0.800026 w_k1_MAT_s113
                                 + 0.853076 w_k1_MSFT_s113
                                 + 1.1096 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.77701 w_k1_ADM_s113
                                 + 1.4639 w_k1_AXP_s113 + 1.30189 w_k1_DIS_s113
                                 + 1.17426 w_k1_FDX_s113 + 1.3651 w_k1_IBM_s113
                                 + 1.06008 w_k1_INTC_s113
                                 + 1.18613 w_k1_KO_s113 + 1.76749 w_k1_MAT_s113
                                 + 1.01179 w_k1_MSFT_s113
                                 + 1.30111 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.936909 w_k1_ADM_s113
                                 + 0.723924 w_k1_AXP_s113
                                 + 0.576721 w_k1_DIS_s113
                                 + 0.713595 w_k1_FDX_s113
                                 + 0.586051 w_k1_IBM_s113
                                 + 0.822397 w_k1_INTC_s113
                                 + 0.673867 w_k1_KO_s113
                                 + 0.546549 w_k1_MAT_s113
                                 + 0.715771 w_k1_MSFT_s113
                                 + 0.864257 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.16176 w_k1_ADM_s113
                                 + 0.937976 w_k1_AXP_s113
                                 + 0.65074 w_k1_DIS_s113
                                 + 0.809494 w_k1_FDX_s113
                                 + 0.807792 w_k1_IBM_s113
                                 + 0.875654 w_k1_INTC_s113
                                 + 0.928126 w_k1_KO_s113
                                 + 1.28842 w_k1_MAT_s113
                                 + 0.846724 w_k1_MSFT_s113
                                 + 1.15961 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.35008 w_k1_ADM_s113
                                 + 1.12524 w_k1_AXP_s113
                                 + 1.09751 w_k1_DIS_s113
                                 + 1.03299 w_k1_FDX_s113
                                 + 0.890846 w_k1_IBM_s113
                                 + 1.14146 w_k1_INTC_s113
                                 + 0.972055 w_k1_KO_s113
                                 + 1.12528 w_k1_MAT_s113
                                 + 1.16801 w_k1_MSFT_s113
                                 + 1.18701 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.12069 w_k1_ADM_s114
                                 + 0.816524 w_k1_AXP_s114
                                 + 0.835208 w_k1_DIS_s114
                                 + 0.951148 w_k1_FDX_s114
                                 + 0.72523 w_k1_IBM_s114
                                 + 1.01579 w_k1_INTC_s114
                                 + 0.781208 w_k1_KO_s114
                                 + 0.926829 w_k1_MAT_s114
                                 + 1.00102 w_k1_MSFT_s114
                                 + 0.927415 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.09495 w_k1_ADM_s114
                                 + 1.30729 w_k1_AXP_s114
                                 + 0.876073 w_k1_DIS_s114
                                 + 0.960665 w_k1_FDX_s114
                                 + 1.00638 w_k1_IBM_s114
                                 + 0.856651 w_k1_INTC_s114
                                 + 0.960295 w_k1_KO_s114
                                 + 1.86664 w_k1_MAT_s114
                                 + 0.852589 w_k1_MSFT_s114
                                 + 1.09149 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.84181 w_k1_ADM_s114
                                 + 0.791376 w_k1_AXP_s114
                                 + 0.93514 w_k1_DIS_s114
                                 + 0.807946 w_k1_FDX_s114
                                 + 1.00905 w_k1_IBM_s114
                                 + 0.948898 w_k1_INTC_s114
                                 + 1.00983 w_k1_KO_s114 + 1.09743 w_k1_MAT_s114
                                 + 0.9689 w_k1_MSFT_s114
                                 + 0.864469 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.41167 w_k1_ADM_s114
                                 + 0.990496 w_k1_AXP_s114
                                 + 1.31764 w_k1_DIS_s114 + 1.2334 w_k1_FDX_s114
                                 + 0.740967 w_k1_IBM_s114
                                 + 0.926616 w_k1_INTC_s114
                                 + 0.998253 w_k1_KO_s114
                                 + 0.777036 w_k1_MAT_s114
                                 + 1.30263 w_k1_MSFT_s114
                                 + 1.07846 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.72777 w_k1_ADM_s114
                                 + 0.785156 w_k1_AXP_s114
                                 + 0.500899 w_k1_DIS_s114
                                 + 0.689672 w_k1_FDX_s114
                                 + 0.731777 w_k1_IBM_s114
                                 + 0.678405 w_k1_INTC_s114
                                 + 0.591395 w_k1_KO_s114
                                 + 0.991997 w_k1_MAT_s114
                                 + 0.686348 w_k1_MSFT_s114
                                 + 0.695612 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.18083 w_k1_ADM_s114
                                 + 1.11394 w_k1_AXP_s114
                                 + 1.09819 w_k1_DIS_s114
                                 + 1.07534 w_k1_FDX_s114
                                 + 0.969824 w_k1_IBM_s114
                                 + 0.707184 w_k1_INTC_s114
                                 + 0.986908 w_k1_KO_s114
                                 + 1.60104 w_k1_MAT_s114
                                 + 1.00421 w_k1_MSFT_s114
                                 + 1.13776 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.69044 w_k1_ADM_s114
                                 + 0.956302 w_k1_AXP_s114
                                 + 1.16279 w_k1_DIS_s114
                                 + 0.935957 w_k1_FDX_s114
                                 + 1.55917 w_k1_IBM_s114
                                 + 0.858677 w_k1_INTC_s114
                                 + 1.12402 w_k1_KO_s114 + 2.56935 w_k1_MAT_s114
                                 + 0.986607 w_k1_MSFT_s114
                                 + 1.06643 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.52268 w_k1_ADM_s114
                                 + 1.42534 w_k1_AXP_s114
                                 + 1.86173 w_k1_DIS_s114
                                 + 1.38716 w_k1_FDX_s114
                                 + 1.29712 w_k1_IBM_s114
                                 + 1.07603 w_k1_INTC_s114
                                 + 1.50028 w_k1_KO_s114 + 1.66163 w_k1_MAT_s114
                                 + 1.35775 w_k1_MSFT_s114
                                 + 1.15907 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.856791 w_k1_ADM_s115
                                 + 0.870979 w_k1_AXP_s115
                                 + 0.914966 w_k1_DIS_s115
                                 + 0.910597 w_k1_FDX_s115
                                 + 0.974919 w_k1_IBM_s115
                                 + 0.966945 w_k1_INTC_s115
                                 + 0.909534 w_k1_KO_s115
                                 + 0.759757 w_k1_MAT_s115
                                 + 0.939278 w_k1_MSFT_s115
                                 + 0.861481 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.06601 w_k1_ADM_s115
                                 + 1.41712 w_k1_AXP_s115
                                 + 1.58725 w_k1_DIS_s115
                                 + 1.34307 w_k1_FDX_s115
                                 + 0.864483 w_k1_IBM_s115
                                 + 1.18067 w_k1_INTC_s115
                                 + 1.20166 w_k1_KO_s115
                                 + 0.849993 w_k1_MAT_s115
                                 + 1.31044 w_k1_MSFT_s115
                                 + 1.11444 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.00687 w_k1_ADM_s115
                                 + 0.722629 w_k1_AXP_s115
                                 + 0.785218 w_k1_DIS_s115
                                 + 0.913549 w_k1_FDX_s115
                                 + 0.681778 w_k1_IBM_s115
                                 + 0.994092 w_k1_INTC_s115
                                 + 0.781414 w_k1_KO_s115
                                 + 0.454064 w_k1_MAT_s115
                                 + 0.936192 w_k1_MSFT_s115
                                 + 0.848159 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.32967 w_k1_ADM_s115
                                 + 1.87186 w_k1_AXP_s115
                                 + 1.16761 w_k1_DIS_s115
                                 + 1.15758 w_k1_FDX_s115
                                 + 1.56877 w_k1_IBM_s115
                                 + 1.16546 w_k1_INTC_s115
                                 + 1.34132 w_k1_KO_s115 + 1.76495 w_k1_MAT_s115
                                 + 1.08744 w_k1_MSFT_s115
                                 + 1.23217 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.792943 w_k1_ADM_s115
                                 + 1.06262 w_k1_AXP_s115
                                 + 1.10248 w_k1_DIS_s115
                                 + 0.946598 w_k1_FDX_s115
                                 + 1.3982 w_k1_IBM_s115
                                 + 1.10738 w_k1_INTC_s115
                                 + 1.17306 w_k1_KO_s115 + 1.21994 w_k1_MAT_s115
                                 + 1.07646 w_k1_MSFT_s115
                                 + 0.915495 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.13293 w_k1_ADM_s115
                                 + 1.37028 w_k1_AXP_s115
                                 + 0.942927 w_k1_DIS_s115
                                 + 1.10614 w_k1_FDX_s115
                                 + 1.05085 w_k1_IBM_s115
                                 + 0.94697 w_k1_INTC_s115
                                 + 0.985861 w_k1_KO_s115
                                 + 1.36864 w_k1_MAT_s115
                                 + 1.07551 w_k1_MSFT_s115
                                 + 0.913353 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.768307 w_k1_ADM_s115
                                 + 0.802541 w_k1_AXP_s115
                                 + 0.579699 w_k1_DIS_s115
                                 + 0.739132 w_k1_FDX_s115
                                 + 0.866783 w_k1_IBM_s115
                                 + 0.791425 w_k1_INTC_s115
                                 + 0.899059 w_k1_KO_s115
                                 + 0.827346 w_k1_MAT_s115
                                 + 0.699973 w_k1_MSFT_s115
                                 + 0.783114 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.27545 w_k1_ADM_s115
                                 + 1.05162 w_k1_AXP_s115
                                 + 1.34555 w_k1_DIS_s115
                                 + 1.15292 w_k1_FDX_s115
                                 + 1.11292 w_k1_IBM_s115
                                 + 1.15087 w_k1_INTC_s115
                                 + 1.39997 w_k1_KO_s115
                                 + 0.961194 w_k1_MAT_s115
                                 + 1.27461 w_k1_MSFT_s115
                                 + 1.06459 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.979005 w_k1_ADM_s116
                                 + 1.38443 w_k1_AXP_s116 + 1.2741 w_k1_DIS_s116
                                 + 1.1702 w_k1_FDX_s116
                                 + 0.996623 w_k1_IBM_s116
                                 + 0.935448 w_k1_INTC_s116
                                 + 1.05779 w_k1_KO_s116 + 1.28986 w_k1_MAT_s116
                                 + 1.19419 w_k1_MSFT_s116
                                 + 1.07342 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.07345 w_k1_ADM_s116
                                 + 1.41234 w_k1_AXP_s116 + 1.5121 w_k1_DIS_s116
                                 + 1.27138 w_k1_FDX_s116
                                 + 1.40041 w_k1_IBM_s116
                                 + 0.880242 w_k1_INTC_s116
                                 + 1.25827 w_k1_KO_s116 + 1.91905 w_k1_MAT_s116
                                 + 1.13995 w_k1_MSFT_s116
                                 + 1.22721 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.88591 w_k1_ADM_s116
                                 + 0.971999 w_k1_AXP_s116
                                 + 0.787436 w_k1_DIS_s116
                                 + 0.862291 w_k1_FDX_s116
                                 + 1.07679 w_k1_IBM_s116
                                 + 0.739739 w_k1_INTC_s116
                                 + 1.00681 w_k1_KO_s116 + 1.36289 w_k1_MAT_s116
                                 + 0.89378 w_k1_MSFT_s116
                                 + 1.03283 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.1313 w_k1_ADM_s116
                                 + 0.95034 w_k1_AXP_s116
                                 + 1.40195 w_k1_DIS_s116
                                 + 1.38706 w_k1_FDX_s116
                                 + 0.715705 w_k1_IBM_s116
                                 + 1.23596 w_k1_INTC_s116
                                 + 0.872873 w_k1_KO_s116
                                 + 0.512754 w_k1_MAT_s116
                                 + 1.33407 w_k1_MSFT_s116
                                 + 1.01398 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.758387 w_k1_ADM_s116
                                 + 0.743384 w_k1_AXP_s116
                                 + 0.944767 w_k1_DIS_s116
                                 + 0.874365 w_k1_FDX_s116
                                 + 1.00713 w_k1_IBM_s116
                                 + 1.0134 w_k1_INTC_s116 + 1.0516 w_k1_KO_s116
                                 + 0.807075 w_k1_MAT_s116
                                 + 1.22079 w_k1_MSFT_s116
                                 + 1.00645 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.687789 w_k1_ADM_s116
                                 + 0.713771 w_k1_AXP_s116
                                 + 0.853756 w_k1_DIS_s116
                                 + 0.932569 w_k1_FDX_s116
                                 + 0.697136 w_k1_IBM_s116
                                 + 0.989384 w_k1_INTC_s116
                                 + 0.774593 w_k1_KO_s116
                                 + 0.868069 w_k1_MAT_s116
                                 + 0.974511 w_k1_MSFT_s116
                                 + 0.832423 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.598988 w_k1_ADM_s116
                                 + 0.879756 w_k1_AXP_s116
                                 + 0.988484 w_k1_DIS_s116
                                 + 0.823358 w_k1_FDX_s116
                                 + 1.33981 w_k1_IBM_s116
                                 + 0.913909 w_k1_INTC_s116
                                 + 1.10156 w_k1_KO_s116 + 1.15655 w_k1_MAT_s116
                                 + 0.924784 w_k1_MSFT_s116
                                 + 1.01252 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.52076 w_k1_ADM_s116
                                 + 1.28464 w_k1_AXP_s116
                                 + 1.81783 w_k1_DIS_s116
                                 + 1.34006 w_k1_FDX_s116
                                 + 1.44962 w_k1_IBM_s116
                                 + 1.33577 w_k1_INTC_s116
                                 + 1.46958 w_k1_KO_s116 + 1.56645 w_k1_MAT_s116
                                 + 1.77978 w_k1_MSFT_s116
                                 + 1.32074 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.963342 w_k1_ADM_s117
                                 + 0.875415 w_k1_AXP_s117
                                 + 0.951821 w_k1_DIS_s117
                                 + 0.923805 w_k1_FDX_s117
                                 + 0.834028 w_k1_IBM_s117
                                 + 1.12805 w_k1_INTC_s117
                                 + 0.76075 w_k1_KO_s117
                                 + 0.599597 w_k1_MAT_s117
                                 + 1.01087 w_k1_MSFT_s117
                                 + 0.901036 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.947594 w_k1_ADM_s117
                                 + 0.952783 w_k1_AXP_s117
                                 + 0.874511 w_k1_DIS_s117
                                 + 0.860264 w_k1_FDX_s117
                                 + 1.01495 w_k1_IBM_s117
                                 + 0.921833 w_k1_INTC_s117
                                 + 0.821728 w_k1_KO_s117
                                 + 0.978718 w_k1_MAT_s117
                                 + 0.978932 w_k1_MSFT_s117
                                 + 0.994497 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.891851 w_k1_ADM_s117
                                 + 0.898771 w_k1_AXP_s117
                                 + 1.14428 w_k1_DIS_s117
                                 + 1.14257 w_k1_FDX_s117 + 0.7525 w_k1_IBM_s117
                                 + 1.04826 w_k1_INTC_s117
                                 + 0.891393 w_k1_KO_s117
                                 + 0.946755 w_k1_MAT_s117
                                 + 1.21614 w_k1_MSFT_s117 + 1.0226 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.17656 w_k1_ADM_s117
                                 + 1.2162 w_k1_AXP_s117 + 1.32914 w_k1_DIS_s117
                                 + 1.13398 w_k1_FDX_s117
                                 + 0.947124 w_k1_IBM_s117
                                 + 1.21986 w_k1_INTC_s117
                                 + 1.05191 w_k1_KO_s117
                                 + 0.852073 w_k1_MAT_s117
                                 + 1.1676 w_k1_MSFT_s117 + 1.11063 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.37969 w_k1_ADM_s117
                                 + 1.67241 w_k1_AXP_s117
                                 + 1.91285 w_k1_DIS_s117
                                 + 1.38937 w_k1_FDX_s117
                                 + 1.25609 w_k1_IBM_s117
                                 + 1.53113 w_k1_INTC_s117
                                 + 1.44627 w_k1_KO_s117 + 1.06593 w_k1_MAT_s117
                                 + 1.62377 w_k1_MSFT_s117
                                 + 1.13054 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.764948 w_k1_ADM_s117
                                 + 1.08763 w_k1_AXP_s117
                                 + 1.32224 w_k1_DIS_s117
                                 + 1.06446 w_k1_FDX_s117
                                 + 1.37536 w_k1_IBM_s117
                                 + 1.08718 w_k1_INTC_s117
                                 + 1.11559 w_k1_KO_s117 + 1.61816 w_k1_MAT_s117
                                 + 1.05883 w_k1_MSFT_s117
                                 + 0.97319 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.973712 w_k1_ADM_s117
                                 + 1.05238 w_k1_AXP_s117 + 1.029 w_k1_DIS_s117
                                 + 0.907618 w_k1_FDX_s117
                                 + 1.5859 w_k1_IBM_s117
                                 + 0.945802 w_k1_INTC_s117
                                 + 1.1264 w_k1_KO_s117 + 1.60521 w_k1_MAT_s117
                                 + 0.965302 w_k1_MSFT_s117
                                 + 1.04151 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.579518 w_k1_ADM_s117
                                 + 0.626321 w_k1_AXP_s117
                                 + 0.65401 w_k1_DIS_s117
                                 + 0.65291 w_k1_FDX_s117
                                 + 0.996192 w_k1_IBM_s117
                                 + 0.701695 w_k1_INTC_s117
                                 + 0.688158 w_k1_KO_s117
                                 + 0.860454 w_k1_MAT_s117
                                 + 0.69078 w_k1_MSFT_s117
                                 + 0.765512 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0106425487883532 w_k1_ADM_s000
                                 + 0.0191862203021492 w_k1_ADM_s001
                                 + 0.0163098686046809 w_k1_ADM_s002
                                 + 0.0123266017514239 w_k1_ADM_s003
                                 + 0.0319469962672064 w_k1_ADM_s004
                                 + 0.0309651131973782 w_k1_ADM_s005
                                 + 0.0166076557998265 w_k1_ADM_s006
                                 + 0.0158759552199948 w_k1_ADM_s007
                                 + 0.0543307897041228 w_k1_ADM_s010
                                 + 0.0206375468589172 w_k1_ADM_s011
                                 + 0.035702770197251 w_k1_ADM_s012
                                 + 0.0632261908712761 w_k1_ADM_s013
                                 + 0.0289371909813777 w_k1_ADM_s014
                                 + 0.0495942668496661 w_k1_ADM_s015
                                 + 0.0266818984197961 w_k1_ADM_s016
                                 + 0.0966537454607272 w_k1_ADM_s017
                                 + 0.0715906656291395 w_k1_ADM_s100
                                 + 0.111991657928879 w_k1_ADM_s101
                                 + 0.0282746621518613 w_k1_ADM_s102
                                 + 0.0296777321567808 w_k1_ADM_s103
                                 + 0.0333713955896472 w_k1_ADM_s104
                                 + 0.0475073744270556 w_k1_ADM_s105
                                 + 0.0512892025092234 w_k1_ADM_s106
                                 + 0.0306705257261542 w_k1_ADM_s107
                                 + 0.0247499381842626 w_k1_ADM_s110
                                 + 0.00672669413618669 w_k1_ADM_s111
                                 + 0.0202571582581 w_k1_ADM_s112
                                 + 0.00908984835288957 w_k1_ADM_s113
                                 + 0.0146038993532915 w_k1_ADM_s114
                                 + 0.014066843672649 w_k1_ADM_s115
                                 + 0.00596229194046191 w_k1_ADM_s116
                                 + 0.00879143112027218 w_k1_ADM_s117
                                 + 0.0134077771348024 w_k1_AXP_s000
                                 + 0.0187464157390177 w_k1_AXP_s001
                                 + 0.0162408782721294 w_k1_AXP_s002
                                 + 0.0115569055024342 w_k1_AXP_s003
                                 + 0.0300199354169517 w_k1_AXP_s004
                                 + 0.0314347330608131 w_k1_AXP_s005
                                 + 0.0171643634462905 w_k1_AXP_s006
                                 + 0.0160822853174593 w_k1_AXP_s007
                                 + 0.0584488728581712 w_k1_AXP_s010
                                 + 0.0202129911008426 w_k1_AXP_s011
                                 + 0.0345253238056633 w_k1_AXP_s012
                                 + 0.0589783152884378 w_k1_AXP_s013
                                 + 0.0284153378105502 w_k1_AXP_s014
                                 + 0.0503986855563407 w_k1_AXP_s015
                                 + 0.0276879434287679 w_k1_AXP_s016
                                 + 0.094737962437713 w_k1_AXP_s017
                                 + 0.06753537775994 w_k1_AXP_s100
                                 + 0.109445425936074 w_k1_AXP_s101
                                 + 0.0291577904984907 w_k1_AXP_s102
                                 + 0.0310294795165946 w_k1_AXP_s103
                                 + 0.0350040827769025 w_k1_AXP_s104
                                 + 0.0463051439008999 w_k1_AXP_s105
                                 + 0.0518948911662247 w_k1_AXP_s106
                                 + 0.0318255592461493 w_k1_AXP_s107
                                 + 0.0309258322864224 w_k1_AXP_s110
                                 + 0.00678967148546291 w_k1_AXP_s111
                                 + 0.0205671562330137 w_k1_AXP_s112
                                 + 0.00681091348682235 w_k1_AXP_s113
                                 + 0.0134472958982522 w_k1_AXP_s114
                                 + 0.014407033698576 w_k1_AXP_s115
                                 + 0.00648910072153792 w_k1_AXP_s116
                                 + 0.00922042378984111 w_k1_AXP_s117
                                 + 0.012842329305723 w_k1_DIS_s000
                                 + 0.018261102755685 w_k1_DIS_s001
                                 + 0.0165236987198242 w_k1_DIS_s002
                                 + 0.0113259558855165 w_k1_DIS_s003
                                 + 0.0335155538827981 w_k1_DIS_s004
                                 + 0.0334791600871755 w_k1_DIS_s005
                                 + 0.0171131799569445 w_k1_DIS_s006
                                 + 0.0177344843506992 w_k1_DIS_s007
                                 + 0.062463763907209 w_k1_DIS_s010
                                 + 0.0193596485884311 w_k1_DIS_s011
                                 + 0.036455019100624 w_k1_DIS_s012
                                 + 0.0600089505341325 w_k1_DIS_s013
                                 + 0.0276149392548709 w_k1_DIS_s014
                                 + 0.052034874111296 w_k1_DIS_s015
                                 + 0.0270649745204004 w_k1_DIS_s016
                                 + 0.0963521118247467 w_k1_DIS_s017
                                 + 0.0646326056014758 w_k1_DIS_s100
                                 + 0.101095800423458 w_k1_DIS_s101
                                 + 0.030525203891725 w_k1_DIS_s102
                                 + 0.0321113835170469 w_k1_DIS_s103
                                 + 0.0359407994133666 w_k1_DIS_s104
                                 + 0.0438364823028766 w_k1_DIS_s105
                                 + 0.0471555872662043 w_k1_DIS_s106
                                 + 0.029653844286161 w_k1_DIS_s107
                                 + 0.0279722520921384 w_k1_DIS_s110
                                 + 0.00650657592277991 w_k1_DIS_s111
                                 + 0.0218841309986405 w_k1_DIS_s112
                                 + 0.00589498820228457 w_k1_DIS_s113
                                 + 0.0145466423036805 w_k1_DIS_s114
                                 + 0.0141520359529886 w_k1_DIS_s115
                                 + 0.00730573188876464 w_k1_DIS_s116
                                 + 0.0098440970738633 w_k1_DIS_s117
                                 + 0.0121919004454086 w_k1_FDX_s000
                                 + 0.0184607597227594 w_k1_FDX_s001
                                 + 0.0160128985226012 w_k1_FDX_s002
                                 + 0.0115634895470217 w_k1_FDX_s003
                                 + 0.029995311324144 w_k1_FDX_s004
                                 + 0.0324592005683562 w_k1_FDX_s005
                                 + 0.0167053003950827 w_k1_FDX_s006
                                 + 0.0153521589904394 w_k1_FDX_s007
                                 + 0.056330906646144 w_k1_FDX_s010
                                 + 0.0194909779654143 w_k1_FDX_s011
                                 + 0.0355922960584696 w_k1_FDX_s012
                                 + 0.0610127290223469 w_k1_FDX_s013
                                 + 0.0276038151440713 w_k1_FDX_s014
                                 + 0.0514286164361014 w_k1_FDX_s015
                                 + 0.0254038951704136 w_k1_FDX_s016
                                 + 0.092103348030925 w_k1_FDX_s017
                                 + 0.0632993408663838 w_k1_FDX_s100
                                 + 0.109228645627344 w_k1_FDX_s101
                                 + 0.0291057441342936 w_k1_FDX_s102
                                 + 0.0306884718437753 w_k1_FDX_s103
                                 + 0.0368739612644494 w_k1_FDX_s104
                                 + 0.0447874730555065 w_k1_FDX_s105
                                 + 0.0487491703883516 w_k1_FDX_s106
                                 + 0.0289790471321139 w_k1_FDX_s107
                                 + 0.028756988896009 w_k1_FDX_s110
                                 + 0.00668337912873152 w_k1_FDX_s111
                                 + 0.0213849570937668 w_k1_FDX_s112
                                 + 0.00605166548214427 w_k1_FDX_s113
                                 + 0.0135201193314654 w_k1_FDX_s114
                                 + 0.0138915308288032 w_k1_FDX_s115
                                 + 0.00693867426745549 w_k1_FDX_s116
                                 + 0.00886498434642619 w_k1_FDX_s117
                                 + 0.0129576776352072 w_k1_IBM_s000
                                 + 0.0189223643627096 w_k1_IBM_s001
                                 + 0.0160919680899263 w_k1_IBM_s002
                                 + 0.0114149959031354 w_k1_IBM_s003
                                 + 0.030636506998283 w_k1_IBM_s004
                                 + 0.0322566635741763 w_k1_IBM_s005
                                 + 0.0170752117920347 w_k1_IBM_s006
                                 + 0.0162692199723053 w_k1_IBM_s007
                                 + 0.0580343381648219 w_k1_IBM_s010
                                 + 0.019449069476779 w_k1_IBM_s011
                                 + 0.0355087246120568 w_k1_IBM_s012
                                 + 0.0602729117253536 w_k1_IBM_s013
                                 + 0.0269602781396918 w_k1_IBM_s014
                                 + 0.0517831420207458 w_k1_IBM_s015
                                 + 0.0274847346010397 w_k1_IBM_s016
                                 + 0.0947012148172339 w_k1_IBM_s017
                                 + 0.0617249512442198 w_k1_IBM_s100
                                 + 0.107499708802184 w_k1_IBM_s101
                                 + 0.0275518425723302 w_k1_IBM_s102
                                 + 0.029766983820538 w_k1_IBM_s103
                                 + 0.0377664284126556 w_k1_IBM_s104
                                 + 0.0442582731161277 w_k1_IBM_s105
                                 + 0.0503831873637473 w_k1_IBM_s106
                                 + 0.0276510569111945 w_k1_IBM_s107
                                 + 0.0290577403457158 w_k1_IBM_s110
                                 + 0.00681344278663204 w_k1_IBM_s111
                                 + 0.0208147300252645 w_k1_IBM_s112
                                 + 0.00588880229741912 w_k1_IBM_s113
                                 + 0.012905570187913 w_k1_IBM_s114
                                 + 0.0140020925704598 w_k1_IBM_s115
                                 + 0.00681667507240932 w_k1_IBM_s116
                                 + 0.00949882406414941 w_k1_IBM_s117
                                 + 0.0121408965446385 w_k1_INTC_s000
                                 + 0.0179450483208099 w_k1_INTC_s001
                                 + 0.0172400537546323 w_k1_INTC_s002
                                 + 0.0114558278754895 w_k1_INTC_s003
                                 + 0.0300225716002798 w_k1_INTC_s004
                                 + 0.0319009346821561 w_k1_INTC_s005
                                 + 0.0167056469904936 w_k1_INTC_s006
                                 + 0.0162179777555603 w_k1_INTC_s007
                                 + 0.0535336455041766 w_k1_INTC_s010
                                 + 0.0195842311587462 w_k1_INTC_s011
                                 + 0.0376348201707167 w_k1_INTC_s012
                                 + 0.0720885928666849 w_k1_INTC_s013
                                 + 0.0300701262420692 w_k1_INTC_s014
                                 + 0.0499140938461166 w_k1_INTC_s015
                                 + 0.030101995769999 w_k1_INTC_s016
                                 + 0.0853646468173316 w_k1_INTC_s017
                                 + 0.058584814254252 w_k1_INTC_s100
                                 + 0.102674879426449 w_k1_INTC_s101
                                 + 0.0299638028488046 w_k1_INTC_s102
                                 + 0.0300419982169596 w_k1_INTC_s103
                                 + 0.0332964484063191 w_k1_INTC_s104
                                 + 0.0426995739227828 w_k1_INTC_s105
                                 + 0.0449967789833782 w_k1_INTC_s106
                                 + 0.0297831464324014 w_k1_INTC_s107
                                 + 0.0250794237798122 w_k1_INTC_s110
                                 + 0.0063534296759252 w_k1_INTC_s111
                                 + 0.0201420644954469 w_k1_INTC_s112
                                 + 0.00622857860980507 w_k1_INTC_s113
                                 + 0.0113814837760285 w_k1_INTC_s114
                                 + 0.0140655696801396 w_k1_INTC_s115
                                 + 0.00669470706256996 w_k1_INTC_s116
                                 + 0.00971538889668006 w_k1_INTC_s117
                                 + 0.0142611643652953 w_k1_KO_s000
                                 + 0.0170669092333529 w_k1_KO_s001
                                 + 0.0148130445988462 w_k1_KO_s002
                                 + 0.0102326700600131 w_k1_KO_s003
                                 + 0.0293332379875087 w_k1_KO_s004
                                 + 0.034009705640874 w_k1_KO_s005
                                 + 0.0147678137629193 w_k1_KO_s006
                                 + 0.0176994718455601 w_k1_KO_s007
                                 + 0.0653232130495518 w_k1_KO_s010
                                 + 0.0193358625348393 w_k1_KO_s011
                                 + 0.0347796392970115 w_k1_KO_s012
                                 + 0.059406180122393 w_k1_KO_s013
                                 + 0.0292094596365234 w_k1_KO_s014
                                 + 0.0471981949840178 w_k1_KO_s015
                                 + 0.0264580255187357 w_k1_KO_s016
                                 + 0.0899368756617433 w_k1_KO_s017
                                 + 0.0574522641076987 w_k1_KO_s100
                                 + 0.109290566863916 w_k1_KO_s101
                                 + 0.0304406100999946 w_k1_KO_s102
                                 + 0.0286586430345153 w_k1_KO_s103
                                 + 0.035927601550878 w_k1_KO_s104
                                 + 0.0453434967393986 w_k1_KO_s105
                                 + 0.0479636357569103 w_k1_KO_s106
                                 + 0.0301049746091734 w_k1_KO_s107
                                 + 0.0266280959447052 w_k1_KO_s110
                                 + 0.00704729418394166 w_k1_KO_s111
                                 + 0.0196638923888492 w_k1_KO_s112
                                 + 0.0060912481406449 w_k1_KO_s113
                                 + 0.0131982356701494 w_k1_KO_s114
                                 + 0.0145931408517803 w_k1_KO_s115
                                 + 0.0068544176500615 w_k1_KO_s116
                                 + 0.00844252310800774 w_k1_KO_s117
                                 + 0.0134809126639694 w_k1_MAT_s000
                                 + 0.0191189169740296 w_k1_MAT_s001
                                 + 0.0188211494663188 w_k1_MAT_s002
                                 + 0.00959532110483482 w_k1_MAT_s003
                                 + 0.0336482052962159 w_k1_MAT_s004
                                 + 0.0404428630729227 w_k1_MAT_s005
                                 + 0.0144610621441812 w_k1_MAT_s006
                                 + 0.0176068016508575 w_k1_MAT_s007
                                 + 0.0652628805621913 w_k1_MAT_s010
                                 + 0.0202861094862591 w_k1_MAT_s011
                                 + 0.0293737070507741 w_k1_MAT_s012
                                 + 0.0612703830733737 w_k1_MAT_s013
                                 + 0.0271551351257942 w_k1_MAT_s014
                                 + 0.0640621455748048 w_k1_MAT_s015
                                 + 0.0229651525332947 w_k1_MAT_s016
                                 + 0.0865346228920619 w_k1_MAT_s017
                                 + 0.068326765132182 w_k1_MAT_s100
                                 + 0.111808626508024 w_k1_MAT_s101
                                 + 0.0309029709677234 w_k1_MAT_s102
                                 + 0.0247689868079293 w_k1_MAT_s103
                                 + 0.0507330328785935 w_k1_MAT_s104
                                 + 0.0501336101162931 w_k1_MAT_s105
                                 + 0.0475577997255344 w_k1_MAT_s106
                                 + 0.0298620909763483 w_k1_MAT_s107
                                 + 0.0294261705158788 w_k1_MAT_s110
                                 + 0.0069198607267334 w_k1_MAT_s111
                                 + 0.0191435934839935 w_k1_MAT_s112
                                 + 0.00717618826385151 w_k1_MAT_s113
                                 + 0.0181374172535822 w_k1_MAT_s114
                                 + 0.0128737127031699 w_k1_MAT_s115
                                 + 0.00745115286765617 w_k1_MAT_s116
                                 + 0.00878866430202711 w_k1_MAT_s117
                                 + 0.0131333128613415 w_k1_MSFT_s000
                                 + 0.0192031891355584 w_k1_MSFT_s001
                                 + 0.0174172953040954 w_k1_MSFT_s002
                                 + 0.0103875936029123 w_k1_MSFT_s003
                                 + 0.0316209022092758 w_k1_MSFT_s004
                                 + 0.0318026056458276 w_k1_MSFT_s005
                                 + 0.0155416722468422 w_k1_MSFT_s006
                                 + 0.0161327911859475 w_k1_MSFT_s007
                                 + 0.0548173359262426 w_k1_MSFT_s010
                                 + 0.0195986581923759 w_k1_MSFT_s011
                                 + 0.0375562101114966 w_k1_MSFT_s012
                                 + 0.064671350205265 w_k1_MSFT_s013
                                 + 0.0256216456485495 w_k1_MSFT_s014
                                 + 0.0497779562959552 w_k1_MSFT_s015
                                 + 0.0257845386644616 w_k1_MSFT_s016
                                 + 0.099000362462087 w_k1_MSFT_s017
                                 + 0.060223578827558 w_k1_MSFT_s100
                                 + 0.104179692433394 w_k1_MSFT_s101
                                 + 0.0276876732710473 w_k1_MSFT_s102
                                 + 0.0301554282443524 w_k1_MSFT_s103
                                 + 0.0365514771836626 w_k1_MSFT_s104
                                 + 0.0428603173782652 w_k1_MSFT_s105
                                 + 0.0447281744059599 w_k1_MSFT_s106
                                 + 0.0288124761279737 w_k1_MSFT_s107
                                 + 0.0267732772264639 w_k1_MSFT_s110
                                 + 0.00631264515289236 w_k1_MSFT_s111
                                 + 0.020111779354956 w_k1_MSFT_s112
                                 + 0.00588352499926553 w_k1_MSFT_s113
                                 + 0.0137191332734733 w_k1_MSFT_s114
                                 + 0.0143428506202181 w_k1_MSFT_s115
                                 + 0.00762303475023473 w_k1_MSFT_s116
                                 + 0.00964949997478775 w_k1_MSFT_s117
                                 + 0.0141630274915206 w_k1_PG_s000
                                 + 0.0181065973551472 w_k1_PG_s001
                                 + 0.0147094461110228 w_k1_PG_s002
                                 + 0.0113815695250661 w_k1_PG_s003
                                 + 0.031047945455733 w_k1_PG_s004
                                 + 0.0317750227088346 w_k1_PG_s005
                                 + 0.015534295679152 w_k1_PG_s006
                                 + 0.0149139380260281 w_k1_PG_s007
                                 + 0.0558831481777521 w_k1_PG_s010
                                 + 0.0176569022033076 w_k1_PG_s011
                                 + 0.0354934239420186 w_k1_PG_s012
                                 + 0.0601747155181112 w_k1_PG_s013
                                 + 0.0275280012968318 w_k1_PG_s014
                                 + 0.0526990346606414 w_k1_PG_s015
                                 + 0.0264019861403399 w_k1_PG_s016
                                 + 0.0929692327324731 w_k1_PG_s017
                                 + 0.0624569554270087 w_k1_PG_s100
                                 + 0.103350585745598 w_k1_PG_s101
                                 + 0.0268390487675579 w_k1_PG_s102
                                 + 0.0297386091512642 w_k1_PG_s103
                                 + 0.0356340910533925 w_k1_PG_s104
                                 + 0.0416498794720075 w_k1_PG_s105
                                 + 0.0488576639278692 w_k1_PG_s106
                                 + 0.0288685026610804 w_k1_PG_s107
                                 + 0.0284948779842862 w_k1_PG_s110
                                 + 0.00630932263404905 w_k1_PG_s111
                                 + 0.0198334230643415 w_k1_PG_s112
                                 + 0.0070344844443917 w_k1_PG_s113
                                 + 0.0134016165647704 w_k1_PG_s114
                                 + 0.0129058576277367 w_k1_PG_s115
                                 + 0.0068933875796609 w_k1_PG_s116
                                 + 0.00894728817403073 w_k1_PG_s117 >= 1.06
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
