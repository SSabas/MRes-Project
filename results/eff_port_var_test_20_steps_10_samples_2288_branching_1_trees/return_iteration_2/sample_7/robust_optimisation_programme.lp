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
 w_k1_ADM_s0_return_constraint:  1.06052 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.97545 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.20194 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.810847 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.15552 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.970609 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.907206 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.776262 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.820502 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.667836 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.943297 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.05806 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.05644 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.11031 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.43416 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.24057 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.781743 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.20814 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.887259 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.929829 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.19078 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.19644 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.87246 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.03376 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.14677 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.19448 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.30032 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.91509 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.05396 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.43248 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.649895 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.26186 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.74708 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.972582 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.795707 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.897936 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.01566 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.04272 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.986357 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.05993 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.34142 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.823749 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.31318 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.774352 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.809541 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.682099 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.783399 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.60895 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.941451 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.3037 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.31089 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.13937 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.53709 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.28501 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.688983 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.22238 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.984854 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.818659 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.40933 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.07792 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.699046 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.889117 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.989324 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.09814 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.171 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.940208 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.05187 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.47961 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.734884 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.34357 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.898317 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.11084 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.0384 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.800481 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.44145 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.23745 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.04011 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.05523 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.36393 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.714761 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.33406 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.761786 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.846744 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.645108 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.784065 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.604372 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.716642 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.27088 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.38582 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.28817 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.54247 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.2856 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.614447 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.33073 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.913859 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.837898 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.54209 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.21153 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.755169 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.809496 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.17375 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.29499 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.37114 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.91471 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.93537 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.69109 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.655807 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.46277 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.677234 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.16056 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.844134 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.872539 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.40916 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.22873 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.02921 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.992417 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.20997 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.826724 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.15432 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.798559 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.01872 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.843422 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.848337 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.693619 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.871152 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.08961 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.07467 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.26249 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.37694 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.10312 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.747476 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.09337 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.867391 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.894498 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.21944 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.20853 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.813738 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.01554 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.17035 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.10885 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.34564 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.87651 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.932399 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.31788 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.752389 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.35475 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.821443 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.0164 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.910681 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.999196 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.22202 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.27888 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.898622 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.29378 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.18596 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.918846 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.17057 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.895807 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.641388 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.520623 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.01305 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.782389 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.8314 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.31793 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.87298 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.890402 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.15476 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.19085 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.619612 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.51077 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.30805 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.843857 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 2.05713 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.790647 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.776714 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.680684 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.995715 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.20743 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.827993 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.12561 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.52587 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.65337 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.0072 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.05512 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.751697 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.33611 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.27239 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.768258 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.75551 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.702984 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.98411 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.990183 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.2684 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.92059 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.20428 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.908986 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.878904 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.801434 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.881219 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.710322 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.950835 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.17938 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.24475 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.984887 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.29166 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.927185 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.88391 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.998798 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.959031 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.97498 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.22588 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.986364 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.865848 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.987409 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.11536 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.15032 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.978905 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.870852 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.830276 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.36191 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.827608 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.08187 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.775629 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.0626 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.933021 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.8813 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.13805 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.990878 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.955975 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.08652 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.23856 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.889491 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.21496 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.802457 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.809232 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.65736 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.862779 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.699505 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.779988 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.26783 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.42784 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.05814 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.16933 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.08335 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.692708 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.3246 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.988527 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.831633 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.42723 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.02839 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.803094 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.794127 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.15025 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.23082 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.00224 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.03809 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.22847 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.69451 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.945682 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.37313 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.875412 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.225 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.15973 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.949199 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.6797 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.04436 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.91432 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.38327 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.12156 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.805931 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.08226 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.799261 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.831045 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.514917 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.788368 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.737049 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.06346 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.44168 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 2.08557 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.03228 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.3129 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.40179 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.496007 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.71139 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.28432 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.688144 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.93419 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.875771 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.711407 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.826293 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.981226 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.26626 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.766274 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.07326 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.40897 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.58437 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.952654 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.20748 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.794139 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.57612 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.10104 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.87223 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.85286 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.912248 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.03136 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.02733 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.15145 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.808963 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.21661 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.825282 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.995889 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.888793 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.918925 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.746889 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.908321 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.0954 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.13753 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.14233 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.26643 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.06541 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.818358 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.03653 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.92195 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.01585 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.31002 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.21297 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.849799 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.894781 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.09574 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.04616 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.06014 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.83192 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.815921 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.17741 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.791393 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.25693 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.771137 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.02593 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.909898 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.05319 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.23324 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.09304 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.983398 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.01395 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.2121 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.880023 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.12622 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.90589 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.938365 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.793364 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.814254 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.750663 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.855453 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.03938 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.05503 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.994414 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.25077 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.20107 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.779353 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.17938 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.915348 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.950291 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.30386 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.05505 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.859364 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.963437 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.10089 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.19477 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.07289 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.971103 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.13658 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.37143 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.894415 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.20698 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.853002 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.05248 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.00126 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.951815 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.30281 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.05287 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.481000634759342 z_k1_s0000
                                 - 0.56300025146137 z_k1_s0001
                                 - 0.182000509742045 z_k1_s0002
                                 - 0.861003363389745 z_k1_s0003
                                 - 0.45100158587038 z_k1_s0004
                                 - 0.38800025859526 z_k1_s0005
                                 - 0.453000044859289 z_k1_s0006
                                 - 0.315000725960842 z_k1_s0007
                                 - 0.25199973367921 z_k1_s0010
                                 - 0.174999707060666 z_k1_s0011
                                 - 0.25199973367921 z_k1_s0012
                                 - 0.667998377188627 z_k1_s0013
                                 - 0.385000133092718 z_k1_s0014
                                 - 0.166999733147023 z_k1_s0015
                                 - 0.483000368934307 z_k1_s0016
                                 - 0.395000100484771 z_k1_s0017
                                 - 0.26999926368846 z_k1_s0020
                                 - 0.539999342975015 z_k1_s0021
                                 - 0.715997255978652 z_k1_s0022
                                 - 1.01899602640553 z_k1_s0023
                                 - 0.387999180531007 z_k1_s0024
                                 - 0.562000101598088 z_k1_s0025
                                 - 0.272999441282342 z_k1_s0026
                                 - 0.309999456678637 z_k1_s0027
                                 - 0.313000569739166 z_k1_s0030
                                 - 0.11899985732902 z_k1_s0031
                                 - 0.235999868255841 z_k1_s0032
                                 - 0.485001119525498 z_k1_s0033
                                 - 0.475999429316078 z_k1_s0034
                                 - 0.275998913099389 z_k1_s0035
                                 - 0.230000064449298 z_k1_s0036
                                 - 0.511000710353762 z_k1_s0037
                                 - 0.404999988173916 z_k1_s0040
                                 - 0.459999588172879 z_k1_s0041
                                 - 0.262000106176696 z_k1_s0042
                                 - 0.307000617084862 z_k1_s0043
                                 - 0.177999920814715 z_k1_s0044
                                 - 0.41599944717375 z_k1_s0045
                                 - 0.141000064179061 z_k1_s0046
                                 - 0.136000058633704 z_k1_s0047
                                 - 1.59299324291403 z_k1_s0050
                                 - 0.40899873439244 z_k1_s0051
                                 - 0.4869982623932 z_k1_s0052
                                 - 1.13199590001765 z_k1_s0053
                                 - 0.762996443142517 z_k1_s0054
                                 - 1.04499881207075 z_k1_s0055
                                 - 0.639998131746004 z_k1_s0056
                                 - 0.391998605331475 z_k1_s0057
                                 - 0.313998330427192 z_k1_s0060
                                 - 0.304999937006165 z_k1_s0061
                                 - 0.373998766559224 z_k1_s0062
                                 - 0.798999575829454 z_k1_s0063
                                 - 0.187999846548146 z_k1_s0064
                                 - 0.378999562903005 z_k1_s0065
                                 - 0.334998635073292 z_k1_s0066
                                 - 0.346000387029609 z_k1_s0067
                                 - 2.24299556174633 z_k1_s0070
                                 - 1.70500109411489 z_k1_s0071
                                 - 0.493999572843837 z_k1_s0072
                                 - 0.522999472892218 z_k1_s0073
                                 - 0.730999111942396 z_k1_s0074
                                 - 0.717999779651258 z_k1_s0075
                                 - 1.3209995107317 z_k1_s0076
                                 - 0.867003507779294 z_k1_s0077
                                 - 0.311000235337954 z_k1_s0100
                                 - 0.201000300375775 z_k1_s0101
                                 - 0.684001443562679 z_k1_s0102
                                 - 0.835001060545978 z_k1_s0103
                                 - 0.425001103332241 z_k1_s0104
                                 - 0.210000046946116 z_k1_s0105
                                 - 0.207000131422669 z_k1_s0106
                                 - 0.264000094869432 z_k1_s0107
                                 - 0.184000065722313 z_k1_s0110
                                 - 0.228001068104376 z_k1_s0111
                                 - 0.186000128621509 z_k1_s0112
                                 - 0.310000214369181 z_k1_s0113
                                 - 0.245000077146551 z_k1_s0114
                                 - 0.268000419087057 z_k1_s0115
                                 - 0.304000788472089 z_k1_s0116
                                 - 0.182000193523241 z_k1_s0117
                                 - 1.48200160325896 z_k1_s0120
                                 - 0.618000602504783 z_k1_s0121
                                 - 0.308999757352685 z_k1_s0122
                                 - 0.438999945699936 z_k1_s0123
                                 - 0.662997425222889 z_k1_s0124
                                 - 0.392999628023352 z_k1_s0125
                                 - 1.05799914806584 z_k1_s0126
                                 - 0.476999498694019 z_k1_s0127
                                 - 0.759999001414971 z_k1_s0130
                                 - 0.440999777026543 z_k1_s0131
                                 - 0.347999430605732 z_k1_s0132
                                 - 0.287000067487249 z_k1_s0133
                                 - 0.306001336223436 z_k1_s0134
                                 - 0.403001501156846 z_k1_s0135
                                 - 0.214999985858351 z_k1_s0136
                                 - 0.284000203602799 z_k1_s0137
                                 - 0.439001073494888 z_k1_s0140
                                 - 0.949003030697516 z_k1_s0141
                                 - 0.673000707627532 z_k1_s0142
                                 - 0.313000811399414 z_k1_s0143
                                 - 0.729001844338794 z_k1_s0144
                                 - 0.259000436729369 z_k1_s0145
                                 - 0.47900300039101 z_k1_s0146
                                 - 0.750001122853085 z_k1_s0147
                                 - 1.12100193237254 z_k1_s0150
                                 - 0.483000287800078 z_k1_s0151
                                 - 1.62200309365592 z_k1_s0152
                                 - 0.826998788146162 z_k1_s0153
                                 - 0.425999979103151 z_k1_s0154
                                 - 1.15100209484461 z_k1_s0155
                                 - 0.816002867240079 z_k1_s0156
                                 - 1.314000908276 z_k1_s0157
                                 - 0.0920002196265224 z_k1_s0160
                                 - 0.372001118455434 z_k1_s0161
                                 - 0.140000283814193 z_k1_s0162
                                 - 0.200000405448847 z_k1_s0163
                                 - 0.0810000565666123 z_k1_s0164
                                 - 0.23800069776447 z_k1_s0165
                                 - 0.327000903029247 z_k1_s0166
                                 - 0.206000102971813 z_k1_s0167
                                 - 0.686001803728135 z_k1_s0170
                                 - 0.849000832597593 z_k1_s0171
                                 - 0.906999523763622 z_k1_s0172
                                 - 0.752002188542122 z_k1_s0173
                                 - 0.595001108450891 z_k1_s0174
                                 - 0.320000712862096 z_k1_s0175
                                 - 0.674998720085434 z_k1_s0176
                                 - 1.25500374878794 z_k1_s0177
                                 - 0.455999356882361 z_k1_s1000
                                 - 0.789999180629112 z_k1_s1001
                                 - 0.455999356882361 z_k1_s1002
                                 - 0.39599929999804 z_k1_s1003
                                 - 0.509000549783559 z_k1_s1004
                                 - 0.566998050605354 z_k1_s1005
                                 - 0.8399981077843 z_k1_s1006
                                 - 0.467000196294979 z_k1_s1007
                                 - 0.188000162841277 z_k1_s1010
                                 - 0.21000029091855 z_k1_s1011
                                 - 0.21999914577073 z_k1_s1012
                                 - 0.114000139627196 z_k1_s1013
                                 - 0.349999417363251 z_k1_s1014
                                 - 0.335001028334881 z_k1_s1015
                                 - 0.306001296210702 z_k1_s1016
                                 - 0.411999875353837 z_k1_s1017
                                 - 0.0689999820217459 z_k1_s1020
                                 - 0.0920000068289967 z_k1_s1021
                                 - 0.203000042185723 z_k1_s1022
                                 - 0.0660000310121082 z_k1_s1023
                                 - 0.253000457679772 z_k1_s1024
                                 - 0.0720000254313902 z_k1_s1025
                                 - 0.0909999615591132 z_k1_s1026
                                 - 0.0780000198506721 z_k1_s1027
                                 - 0.168000032944415 z_k1_s1030
                                 - 0.146999599451598 z_k1_s1031
                                 - 0.214000382738532 z_k1_s1032
                                 - 0.170000346848343 z_k1_s1033
                                 - 0.0709999225956063 z_k1_s1034
                                 - 0.115999886437885 z_k1_s1035
                                 - 0.138999488835257 z_k1_s1036
                                 - 0.119999369246369 z_k1_s1037
                                 - 0.422998486367063 z_k1_s1040
                                 - 0.231998251135633 z_k1_s1041
                                 - 0.181999351354142 z_k1_s1042
                                 - 0.362998481833413 z_k1_s1043
                                 - 0.388999808593856 z_k1_s1044
                                 - 0.208999353394285 z_k1_s1045
                                 - 0.270000020401426 z_k1_s1046
                                 - 0.139999568979897 z_k1_s1047
                                 - 0.343001547590838 z_k1_s1050
                                 - 0.106000275374124 z_k1_s1051
                                 - 0.350000652130788 z_k1_s1052
                                 - 0.103000217869994 z_k1_s1053
                                 - 0.41400103094975 z_k1_s1054
                                 - 0.158000424176565 z_k1_s1055
                                 - 0.112000390382385 z_k1_s1056
                                 - 0.231001338908986 z_k1_s1057
                                 - 0.217000517124783 z_k1_s1060
                                 - 0.0880003850909128 z_k1_s1061
                                 - 0.245000145397253 z_k1_s1062
                                 - 0.0990004332272769 z_k1_s1063
                                 - 0.218001231994473 z_k1_s1064
                                 - 0.0910003209914408 z_k1_s1065
                                 - 0.436002463988946 z_k1_s1066
                                 - 0.305000562414382 z_k1_s1067
                                 - 0.194999672176981 z_k1_s1070
                                 - 0.156000478143575 z_k1_s1071
                                 - 0.412000389494786 z_k1_s1072
                                 - 0.181000521545982 z_k1_s1073
                                 - 0.16300042366007 z_k1_s1074
                                 - 0.465002065968147 z_k1_s1075
                                 - 0.109000315102831 z_k1_s1076
                                 - 0.170000369176564 z_k1_s1077
                                 - 0.195000099349992 z_k1_s1100
                                 - 0.138000112801563 z_k1_s1101
                                 - 0.186999960715026 z_k1_s1102
                                 - 0.180999511488558 z_k1_s1103
                                 - 0.0970001273228715 z_k1_s1104
                                 - 0.116000076805649 z_k1_s1105
                                 - 0.259999672132999 z_k1_s1106
                                 - 0.206999616801955 z_k1_s1107
                                 - 0.19700085347967 z_k1_s1110
                                 - 0.11900040271011 z_k1_s1111
                                 - 0.41100097841478 z_k1_s1112
                                 - 0.251000843865 z_k1_s1113
                                 - 0.401001795383723 z_k1_s1114
                                 - 0.268001435925727 z_k1_s1115
                                 - 0.318001753098635 z_k1_s1116
                                 - 0.236000968814009 z_k1_s1117
                                 - 0.179000440941212 z_k1_s1120
                                 - 0.223000449292465 z_k1_s1121
                                 - 0.374999346912511 z_k1_s1122
                                 - 0.390999901768714 z_k1_s1123
                                 - 0.386001245879457 z_k1_s1124
                                 - 0.274000700393307 z_k1_s1125
                                 - 0.601001114243158 z_k1_s1126
                                 - 0.606002805139028 z_k1_s1127
                                 - 0.122999617419198 z_k1_s1130
                                 - 0.279999815718558 z_k1_s1131
                                 - 0.19200017526418 z_k1_s1132
                                 - 0.151999698875772 z_k1_s1133
                                 - 0.112000043587433 z_k1_s1134
                                 - 0.0990000353862131 z_k1_s1135
                                 - 0.0879999923544083 z_k1_s1136
                                 - 0.126999547758029 z_k1_s1137
                                 - 0.0700000504239304 z_k1_s1140
                                 - 0.070999965544262 z_k1_s1141
                                 - 0.208999915151421 z_k1_s1142
                                 - 0.0650000468222211 z_k1_s1143
                                 - 0.293000275260174 z_k1_s1144
                                 - 0.122000344681738 z_k1_s1145
                                 - 0.119000064320679 z_k1_s1146
                                 - 0.120999894561342 z_k1_s1147
                                 - 0.806000111537754 z_k1_s1150
                                 - 0.762001848328968 z_k1_s1151
                                 - 0.606999016107409 z_k1_s1152
                                 - 0.320000176258295 z_k1_s1153
                                 - 0.401000503805204 z_k1_s1154
                                 - 0.328000414114982 z_k1_s1155
                                 - 0.374000259289435 z_k1_s1156
                                 - 0.462000033710197 z_k1_s1157
                                 - 0.134999905201884 z_k1_s1160
                                 - 0.401999461152358 z_k1_s1161
                                 - 0.532998955816539 z_k1_s1162
                                 - 0.148999952384284 z_k1_s1163
                                 - 0.144999969446446 z_k1_s1164
                                 - 0.407000241574402 z_k1_s1165
                                 - 0.207999967967311 z_k1_s1166
                                 - 0.159000016628845 z_k1_s1167
                                 - 0.256000635957414 z_k1_s1170
                                 - 0.0910001260550908 z_k1_s1171
                                 - 0.207999857839848 z_k1_s1172
                                 - 0.300999953566452 z_k1_s1173
                                 - 0.272000393329516 z_k1_s1174
                                 - 0.0760000523184931 z_k1_s1175
                                 - 0.182000252110182 z_k1_s1176
                                 - 0.120000138092404 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.25367 w_k1_ADM_s000
                                 + 1.52766 w_k1_AXP_s000
                                 + 1.57558 w_k1_DIS_s000
                                 + 1.25231 w_k1_FDX_s000
                                 + 1.54036 w_k1_IBM_s000
                                 + 1.13257 w_k1_INTC_s000
                                 + 1.34425 w_k1_KO_s000 + 1.37969 w_k1_MAT_s000
                                 + 1.16851 w_k1_MSFT_s000
                                 + 1.10315 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.689837 w_k1_ADM_s000
                                 + 0.712741 w_k1_AXP_s000
                                 + 0.665477 w_k1_DIS_s000
                                 + 0.752891 w_k1_FDX_s000
                                 + 0.888299 w_k1_IBM_s000
                                 + 0.969117 w_k1_INTC_s000
                                 + 0.826983 w_k1_KO_s000
                                 + 0.692603 w_k1_MAT_s000
                                 + 0.831616 w_k1_MSFT_s000
                                 + 0.842712 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.13226 w_k1_ADM_s000
                                 + 1.12959 w_k1_AXP_s000
                                 + 0.975964 w_k1_DIS_s000
                                 + 1.15475 w_k1_FDX_s000
                                 + 0.756755 w_k1_IBM_s000
                                 + 1.21479 w_k1_INTC_s000
                                 + 0.843748 w_k1_KO_s000
                                 + 0.481791 w_k1_MAT_s000
                                 + 0.989026 w_k1_MSFT_s000
                                 + 0.880319 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.965757 w_k1_ADM_s000
                                 + 1.1242 w_k1_AXP_s000 + 1.11568 w_k1_DIS_s000
                                 + 1.06675 w_k1_FDX_s000
                                 + 1.09851 w_k1_IBM_s000
                                 + 1.03009 w_k1_INTC_s000
                                 + 1.03401 w_k1_KO_s000
                                 + 0.875342 w_k1_MAT_s000
                                 + 0.939278 w_k1_MSFT_s000
                                 + 0.983366 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.929927 w_k1_ADM_s000
                                 + 0.780194 w_k1_AXP_s000
                                 + 0.875473 w_k1_DIS_s000
                                 + 0.946708 w_k1_FDX_s000
                                 + 0.735668 w_k1_IBM_s000
                                 + 0.844748 w_k1_INTC_s000
                                 + 0.693284 w_k1_KO_s000
                                 + 0.764348 w_k1_MAT_s000
                                 + 0.946267 w_k1_MSFT_s000
                                 + 0.826432 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.971867 w_k1_ADM_s000
                                 + 1.00965 w_k1_AXP_s000
                                 + 0.874609 w_k1_DIS_s000
                                 + 0.935961 w_k1_FDX_s000
                                 + 0.762488 w_k1_IBM_s000
                                 + 0.954077 w_k1_INTC_s000
                                 + 0.709128 w_k1_KO_s000
                                 + 0.640045 w_k1_MAT_s000
                                 + 1.00311 w_k1_MSFT_s000
                                 + 0.865734 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.958095 w_k1_ADM_s000
                                 + 0.928905 w_k1_AXP_s000
                                 + 1.15473 w_k1_DIS_s000
                                 + 0.917642 w_k1_FDX_s000
                                 + 1.24569 w_k1_IBM_s000
                                 + 1.05283 w_k1_INTC_s000
                                 + 1.11988 w_k1_KO_s000
                                 + 0.951298 w_k1_MAT_s000
                                 + 1.04132 w_k1_MSFT_s000
                                 + 0.977156 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.24038 w_k1_ADM_s000
                                 + 1.31837 w_k1_AXP_s000 + 1.3105 w_k1_DIS_s000
                                 + 1.2994 w_k1_FDX_s000
                                 + 0.957581 w_k1_IBM_s000
                                 + 1.32548 w_k1_INTC_s000
                                 + 1.12329 w_k1_KO_s000 + 1.05392 w_k1_MAT_s000
                                 + 1.25715 w_k1_MSFT_s000 + 1.1166 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.10857 w_k1_ADM_s001
                                 + 1.31554 w_k1_AXP_s001
                                 + 1.17471 w_k1_DIS_s001
                                 + 1.28288 w_k1_FDX_s001
                                 + 1.08426 w_k1_IBM_s001
                                 + 1.26358 w_k1_INTC_s001
                                 + 1.08737 w_k1_KO_s001 + 1.38378 w_k1_MAT_s001
                                 + 1.13064 w_k1_MSFT_s001
                                 + 1.13857 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.787507 w_k1_ADM_s001
                                 + 0.787366 w_k1_AXP_s001
                                 + 0.829339 w_k1_DIS_s001
                                 + 0.798004 w_k1_FDX_s001
                                 + 1.3086 w_k1_IBM_s001
                                 + 0.921108 w_k1_INTC_s001
                                 + 1.00383 w_k1_KO_s001 + 1.44548 w_k1_MAT_s001
                                 + 0.793206 w_k1_MSFT_s001
                                 + 0.907046 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.28096 w_k1_ADM_s001
                                 + 1.37644 w_k1_AXP_s001
                                 + 1.59529 w_k1_DIS_s001
                                 + 1.27509 w_k1_FDX_s001
                                 + 1.60087 w_k1_IBM_s001
                                 + 1.15351 w_k1_INTC_s001
                                 + 1.36874 w_k1_KO_s001 + 1.97786 w_k1_MAT_s001
                                 + 1.22707 w_k1_MSFT_s001
                                 + 1.24124 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.716362 w_k1_ADM_s001
                                 + 0.74576 w_k1_AXP_s001
                                 + 0.765295 w_k1_DIS_s001
                                 + 0.82398 w_k1_FDX_s001
                                 + 0.925958 w_k1_IBM_s001
                                 + 0.794833 w_k1_INTC_s001
                                 + 0.815155 w_k1_KO_s001
                                 + 0.722289 w_k1_MAT_s001
                                 + 0.816593 w_k1_MSFT_s001
                                 + 0.897168 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.03474 w_k1_ADM_s001
                                 + 1.01796 w_k1_AXP_s001
                                 + 0.730946 w_k1_DIS_s001
                                 + 0.912615 w_k1_FDX_s001
                                 + 0.829724 w_k1_IBM_s001
                                 + 0.866486 w_k1_INTC_s001
                                 + 0.806554 w_k1_KO_s001
                                 + 0.834372 w_k1_MAT_s001
                                 + 0.745683 w_k1_MSFT_s001
                                 + 0.918304 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.37752 w_k1_ADM_s001
                                 + 1.4277 w_k1_AXP_s001 + 1.18219 w_k1_DIS_s001
                                 + 1.03231 w_k1_FDX_s001 + 1.285 w_k1_IBM_s001
                                 + 0.99197 w_k1_INTC_s001 + 1.0889 w_k1_KO_s001
                                 + 1.13016 w_k1_MAT_s001
                                 + 0.944568 w_k1_MSFT_s001
                                 + 1.11094 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.08666 w_k1_ADM_s001
                                 + 0.92684 w_k1_AXP_s001
                                 + 1.15534 w_k1_DIS_s001
                                 + 1.17303 w_k1_FDX_s001
                                 + 0.823818 w_k1_IBM_s001
                                 + 1.11843 w_k1_INTC_s001
                                 + 0.996694 w_k1_KO_s001
                                 + 1.15861 w_k1_MAT_s001
                                 + 1.1058 w_k1_MSFT_s001
                                 + 0.931307 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.89181 w_k1_ADM_s001
                                 + 0.944908 w_k1_AXP_s001
                                 + 0.998732 w_k1_DIS_s001
                                 + 0.95143 w_k1_FDX_s001
                                 + 1.40413 w_k1_IBM_s001
                                 + 0.995251 w_k1_INTC_s001
                                 + 1.10657 w_k1_KO_s001 + 1.6245 w_k1_MAT_s001
                                 + 0.923294 w_k1_MSFT_s001
                                 + 0.937598 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.983978 w_k1_ADM_s002
                                 + 1.23742 w_k1_AXP_s002
                                 + 1.45856 w_k1_DIS_s002
                                 + 1.13245 w_k1_FDX_s002
                                 + 1.22026 w_k1_IBM_s002
                                 + 1.08149 w_k1_INTC_s002 + 1.4669 w_k1_KO_s002
                                 + 1.95233 w_k1_MAT_s002
                                 + 1.08168 w_k1_MSFT_s002
                                 + 1.34027 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.947652 w_k1_ADM_s002
                                 + 0.904053 w_k1_AXP_s002
                                 + 0.678243 w_k1_DIS_s002
                                 + 0.874321 w_k1_FDX_s002
                                 + 0.846078 w_k1_IBM_s002
                                 + 0.84892 w_k1_INTC_s002
                                 + 0.904803 w_k1_KO_s002
                                 + 0.967627 w_k1_MAT_s002
                                 + 0.806729 w_k1_MSFT_s002
                                 + 1.01496 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.941384 w_k1_ADM_s002
                                 + 0.788135 w_k1_AXP_s002
                                 + 0.845556 w_k1_DIS_s002
                                 + 0.862004 w_k1_FDX_s002
                                 + 0.930885 w_k1_IBM_s002
                                 + 0.847459 w_k1_INTC_s002
                                 + 1.03575 w_k1_KO_s002 + 1.09667 w_k1_MAT_s002
                                 + 0.960878 w_k1_MSFT_s002
                                 + 0.987617 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.934958 w_k1_ADM_s002
                                 + 1.07934 w_k1_AXP_s002
                                 + 1.08056 w_k1_DIS_s002
                                 + 1.08283 w_k1_FDX_s002
                                 + 0.940396 w_k1_IBM_s002
                                 + 1.00569 w_k1_INTC_s002
                                 + 1.06341 w_k1_KO_s002 + 1.19025 w_k1_MAT_s002
                                 + 1.04442 w_k1_MSFT_s002
                                 + 1.14293 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.886163 w_k1_ADM_s002
                                 + 0.602306 w_k1_AXP_s002
                                 + 0.742768 w_k1_DIS_s002
                                 + 0.88122 w_k1_FDX_s002
                                 + 0.615817 w_k1_IBM_s002
                                 + 0.839595 w_k1_INTC_s002
                                 + 0.823399 w_k1_KO_s002
                                 + 0.636723 w_k1_MAT_s002
                                 + 0.911577 w_k1_MSFT_s002
                                 + 0.953858 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.00701 w_k1_ADM_s002
                                 + 1.09112 w_k1_AXP_s002
                                 + 0.957949 w_k1_DIS_s002
                                 + 0.844202 w_k1_FDX_s002
                                 + 1.36437 w_k1_IBM_s002
                                 + 0.872621 w_k1_INTC_s002
                                 + 1.26869 w_k1_KO_s002 + 1.59527 w_k1_MAT_s002
                                 + 0.969743 w_k1_MSFT_s002
                                 + 1.04024 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.41576 w_k1_ADM_s002
                                 + 1.39226 w_k1_AXP_s002
                                 + 1.64148 w_k1_DIS_s002
                                 + 1.35814 w_k1_FDX_s002
                                 + 1.06616 w_k1_IBM_s002
                                 + 1.40694 w_k1_INTC_s002 + 1.3216 w_k1_KO_s002
                                 + 1.64721 w_k1_MAT_s002
                                 + 1.32207 w_k1_MSFT_s002
                                 + 1.24955 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.3713 w_k1_ADM_s002
                                 + 1.35684 w_k1_AXP_s002
                                 + 1.31859 w_k1_DIS_s002
                                 + 1.03996 w_k1_FDX_s002
                                 + 1.57707 w_k1_IBM_s002
                                 + 0.971541 w_k1_INTC_s002
                                 + 1.65555 w_k1_KO_s002 + 2.44271 w_k1_MAT_s002
                                 + 1.18281 w_k1_MSFT_s002
                                 + 1.30969 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.779523 w_k1_ADM_s003
                                 + 0.686077 w_k1_AXP_s003
                                 + 0.65768 w_k1_DIS_s003
                                 + 0.751752 w_k1_FDX_s003
                                 + 0.829712 w_k1_IBM_s003
                                 + 0.722508 w_k1_INTC_s003
                                 + 0.714044 w_k1_KO_s003
                                 + 0.679055 w_k1_MAT_s003
                                 + 0.763192 w_k1_MSFT_s003
                                 + 0.817089 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.689866 w_k1_ADM_s003
                                 + 0.700964 w_k1_AXP_s003
                                 + 0.579766 w_k1_DIS_s003
                                 + 0.715913 w_k1_FDX_s003
                                 + 0.567062 w_k1_IBM_s003
                                 + 0.723977 w_k1_INTC_s003
                                 + 0.637822 w_k1_KO_s003
                                 + 0.753815 w_k1_MAT_s003
                                 + 0.744893 w_k1_MSFT_s003
                                 + 0.798124 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.06488 w_k1_ADM_s003
                                 + 1.0786 w_k1_AXP_s003
                                 + 0.805374 w_k1_DIS_s003
                                 + 1.02718 w_k1_FDX_s003
                                 + 0.888304 w_k1_IBM_s003
                                 + 0.944916 w_k1_INTC_s003
                                 + 0.88248 w_k1_KO_s003
                                 + 0.937819 w_k1_MAT_s003
                                 + 0.897767 w_k1_MSFT_s003
                                 + 0.845429 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.85529 w_k1_ADM_s003
                                 + 0.844963 w_k1_AXP_s003
                                 + 0.838708 w_k1_DIS_s003
                                 + 0.887692 w_k1_FDX_s003
                                 + 1.06354 w_k1_IBM_s003
                                 + 1.08291 w_k1_INTC_s003
                                 + 0.986582 w_k1_KO_s003
                                 + 0.926683 w_k1_MAT_s003
                                 + 0.994511 w_k1_MSFT_s003
                                 + 0.861785 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.25968 w_k1_ADM_s003
                                 + 1.20945 w_k1_AXP_s003
                                 + 1.45801 w_k1_DIS_s003 + 1.2004 w_k1_FDX_s003
                                 + 1.18639 w_k1_IBM_s003
                                 + 1.02969 w_k1_INTC_s003
                                 + 1.38922 w_k1_KO_s003 + 1.57922 w_k1_MAT_s003
                                 + 1.06583 w_k1_MSFT_s003
                                 + 1.27315 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.15001 w_k1_ADM_s003
                                 + 1.07195 w_k1_AXP_s003
                                 + 0.961547 w_k1_DIS_s003
                                 + 0.881004 w_k1_FDX_s003
                                 + 1.51087 w_k1_IBM_s003
                                 + 0.819759 w_k1_INTC_s003
                                 + 1.15949 w_k1_KO_s003 + 1.27228 w_k1_MAT_s003
                                 + 0.916457 w_k1_MSFT_s003
                                 + 0.964328 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.944198 w_k1_ADM_s003
                                 + 0.925075 w_k1_AXP_s003
                                 + 0.884587 w_k1_DIS_s003
                                 + 0.96995 w_k1_FDX_s003
                                 + 0.628726 w_k1_IBM_s003
                                 + 0.912073 w_k1_INTC_s003
                                 + 0.805314 w_k1_KO_s003
                                 + 0.556978 w_k1_MAT_s003
                                 + 0.984354 w_k1_MSFT_s003
                                 + 0.897819 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.14797 w_k1_ADM_s003
                                 + 1.1678 w_k1_AXP_s003
                                 + 0.990872 w_k1_DIS_s003
                                 + 1.13002 w_k1_FDX_s003
                                 + 1.03692 w_k1_IBM_s003
                                 + 1.05587 w_k1_INTC_s003
                                 + 1.10712 w_k1_KO_s003 + 1.06538 w_k1_MAT_s003
                                 + 1.0978 w_k1_MSFT_s003 + 1.07733 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.865522 w_k1_ADM_s004
                                 + 1.15652 w_k1_AXP_s004 + 1.1713 w_k1_DIS_s004
                                 + 1.02901 w_k1_FDX_s004 + 1.262 w_k1_IBM_s004
                                 + 1.0847 w_k1_INTC_s004 + 1.22611 w_k1_KO_s004
                                 + 1.43411 w_k1_MAT_s004
                                 + 1.09423 w_k1_MSFT_s004
                                 + 1.18224 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.978789 w_k1_ADM_s004
                                 + 0.946858 w_k1_AXP_s004
                                 + 0.990185 w_k1_DIS_s004
                                 + 1.08037 w_k1_FDX_s004
                                 + 0.740772 w_k1_IBM_s004
                                 + 0.904672 w_k1_INTC_s004
                                 + 0.897755 w_k1_KO_s004
                                 + 0.655034 w_k1_MAT_s004
                                 + 0.963156 w_k1_MSFT_s004
                                 + 0.93631 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.09397 w_k1_ADM_s004
                                 + 1.57606 w_k1_AXP_s004
                                 + 1.77702 w_k1_DIS_s004
                                 + 1.39691 w_k1_FDX_s004
                                 + 1.42541 w_k1_IBM_s004
                                 + 1.07717 w_k1_INTC_s004
                                 + 1.55654 w_k1_KO_s004 + 1.68957 w_k1_MAT_s004
                                 + 1.26293 w_k1_MSFT_s004
                                 + 1.29275 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.808415 w_k1_ADM_s004
                                 + 0.742373 w_k1_AXP_s004
                                 + 0.704972 w_k1_DIS_s004
                                 + 0.788936 w_k1_FDX_s004
                                 + 0.965768 w_k1_IBM_s004
                                 + 0.832514 w_k1_INTC_s004
                                 + 0.800941 w_k1_KO_s004
                                 + 0.57874 w_k1_MAT_s004
                                 + 0.832712 w_k1_MSFT_s004
                                 + 0.87261 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.866013 w_k1_ADM_s004
                                 + 0.700085 w_k1_AXP_s004
                                 + 0.886787 w_k1_DIS_s004
                                 + 0.866269 w_k1_FDX_s004
                                 + 0.753085 w_k1_IBM_s004
                                 + 0.736021 w_k1_INTC_s004
                                 + 0.781181 w_k1_KO_s004
                                 + 0.516509 w_k1_MAT_s004
                                 + 0.81797 w_k1_MSFT_s004
                                 + 0.870759 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.882354 w_k1_ADM_s004
                                 + 1.04695 w_k1_AXP_s004
                                 + 0.96188 w_k1_DIS_s004
                                 + 0.909002 w_k1_FDX_s004
                                 + 1.15546 w_k1_IBM_s004
                                 + 0.871097 w_k1_INTC_s004
                                 + 1.09336 w_k1_KO_s004
                                 + 0.997502 w_k1_MAT_s004
                                 + 0.871543 w_k1_MSFT_s004
                                 + 0.913091 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.746216 w_k1_ADM_s004
                                 + 0.960758 w_k1_AXP_s004
                                 + 0.75596 w_k1_DIS_s004
                                 + 0.850517 w_k1_FDX_s004
                                 + 0.862269 w_k1_IBM_s004
                                 + 0.68261 w_k1_INTC_s004
                                 + 0.894225 w_k1_KO_s004
                                 + 0.945778 w_k1_MAT_s004
                                 + 0.757692 w_k1_MSFT_s004
                                 + 0.935419 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.0928 w_k1_ADM_s004
                                 + 1.45004 w_k1_AXP_s004
                                 + 1.26416 w_k1_DIS_s004
                                 + 1.22113 w_k1_FDX_s004
                                 + 1.21856 w_k1_IBM_s004
                                 + 1.08973 w_k1_INTC_s004
                                 + 1.14343 w_k1_KO_s004 + 1.15592 w_k1_MAT_s004
                                 + 1.07988 w_k1_MSFT_s004 + 1.1284 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.20723 w_k1_ADM_s005
                                 + 1.17819 w_k1_AXP_s005
                                 + 1.21677 w_k1_DIS_s005
                                 + 1.23614 w_k1_FDX_s005 + 1.0609 w_k1_IBM_s005
                                 + 1.07676 w_k1_INTC_s005
                                 + 1.07528 w_k1_KO_s005 + 1.29592 w_k1_MAT_s005
                                 + 1.27899 w_k1_MSFT_s005
                                 + 1.18039 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.643756 w_k1_ADM_s005
                                 + 0.539323 w_k1_AXP_s005
                                 + 0.496827 w_k1_DIS_s005
                                 + 0.649557 w_k1_FDX_s005
                                 + 0.716159 w_k1_IBM_s005
                                 + 0.544759 w_k1_INTC_s005
                                 + 0.769055 w_k1_KO_s005
                                 + 0.743476 w_k1_MAT_s005
                                 + 0.784185 w_k1_MSFT_s005
                                 + 0.8321 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.998871 w_k1_ADM_s005
                                 + 1.08552 w_k1_AXP_s005
                                 + 0.654896 w_k1_DIS_s005
                                 + 0.901553 w_k1_FDX_s005
                                 + 0.869648 w_k1_IBM_s005
                                 + 0.733106 w_k1_INTC_s005
                                 + 0.746227 w_k1_KO_s005
                                 + 1.00504 w_k1_MAT_s005
                                 + 0.803012 w_k1_MSFT_s005
                                 + 1.0023 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.878341 w_k1_ADM_s005
                                 + 0.776173 w_k1_AXP_s005
                                 + 0.891264 w_k1_DIS_s005
                                 + 0.92339 w_k1_FDX_s005
                                 + 1.14361 w_k1_IBM_s005
                                 + 0.766638 w_k1_INTC_s005
                                 + 0.942569 w_k1_KO_s005
                                 + 0.841868 w_k1_MAT_s005
                                 + 0.885382 w_k1_MSFT_s005
                                 + 1.01487 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.1278 w_k1_ADM_s005
                                 + 0.96088 w_k1_AXP_s005
                                 + 0.956107 w_k1_DIS_s005
                                 + 1.10673 w_k1_FDX_s005
                                 + 0.768561 w_k1_IBM_s005
                                 + 0.873063 w_k1_INTC_s005
                                 + 0.934882 w_k1_KO_s005
                                 + 0.911147 w_k1_MAT_s005
                                 + 1.0449 w_k1_MSFT_s005
                                 + 0.977615 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.851678 w_k1_ADM_s005
                                 + 0.659649 w_k1_AXP_s005
                                 + 0.735939 w_k1_DIS_s005
                                 + 0.806261 w_k1_FDX_s005
                                 + 0.848864 w_k1_IBM_s005
                                 + 0.708948 w_k1_INTC_s005
                                 + 0.860164 w_k1_KO_s005
                                 + 0.961685 w_k1_MAT_s005
                                 + 0.999798 w_k1_MSFT_s005
                                 + 0.895184 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.16807 w_k1_ADM_s005
                                 + 1.62843 w_k1_AXP_s005 + 1.6673 w_k1_DIS_s005
                                 + 1.3836 w_k1_FDX_s005 + 1.59959 w_k1_IBM_s005
                                 + 1.15783 w_k1_INTC_s005
                                 + 1.41764 w_k1_KO_s005 + 1.35314 w_k1_MAT_s005
                                 + 1.29955 w_k1_MSFT_s005
                                 + 1.36357 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.638388 w_k1_ADM_s005
                                 + 0.836446 w_k1_AXP_s005
                                 + 0.736433 w_k1_DIS_s005
                                 + 0.7653 w_k1_FDX_s005 + 1.37015 w_k1_IBM_s005
                                 + 0.745561 w_k1_INTC_s005
                                 + 0.943224 w_k1_KO_s005
                                 + 1.55139 w_k1_MAT_s005
                                 + 0.852903 w_k1_MSFT_s005
                                 + 0.900055 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.791648 w_k1_ADM_s006
                                 + 0.706936 w_k1_AXP_s006
                                 + 0.989226 w_k1_DIS_s006
                                 + 0.968687 w_k1_FDX_s006
                                 + 1.05824 w_k1_IBM_s006
                                 + 0.914721 w_k1_INTC_s006
                                 + 1.12031 w_k1_KO_s006
                                 + 0.918423 w_k1_MAT_s006
                                 + 1.03603 w_k1_MSFT_s006
                                 + 0.964797 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.846157 w_k1_ADM_s006
                                 + 0.936313 w_k1_AXP_s006
                                 + 0.950049 w_k1_DIS_s006
                                 + 0.877734 w_k1_FDX_s006
                                 + 1.51283 w_k1_IBM_s006
                                 + 0.762224 w_k1_INTC_s006
                                 + 1.25986 w_k1_KO_s006 + 1.64348 w_k1_MAT_s006
                                 + 0.859432 w_k1_MSFT_s006
                                 + 1.03671 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.24314 w_k1_ADM_s006
                                 + 1.31834 w_k1_AXP_s006
                                 + 1.18807 w_k1_DIS_s006
                                 + 1.30181 w_k1_FDX_s006
                                 + 1.00233 w_k1_IBM_s006
                                 + 0.925972 w_k1_INTC_s006
                                 + 1.06245 w_k1_KO_s006 + 1.25664 w_k1_MAT_s006
                                 + 1.19014 w_k1_MSFT_s006
                                 + 1.20145 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.959939 w_k1_ADM_s006
                                 + 0.776486 w_k1_AXP_s006
                                 + 0.813413 w_k1_DIS_s006
                                 + 0.912165 w_k1_FDX_s006
                                 + 0.70314 w_k1_IBM_s006
                                 + 0.649876 w_k1_INTC_s006
                                 + 0.794853 w_k1_KO_s006
                                 + 0.918276 w_k1_MAT_s006
                                 + 0.950565 w_k1_MSFT_s006
                                 + 0.959251 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.23163 w_k1_ADM_s006
                                 + 1.77397 w_k1_AXP_s006
                                 + 2.17024 w_k1_DIS_s006
                                 + 1.61497 w_k1_FDX_s006
                                 + 1.31793 w_k1_IBM_s006
                                 + 1.16281 w_k1_INTC_s006
                                 + 1.87164 w_k1_KO_s006 + 1.89601 w_k1_MAT_s006
                                 + 1.6179 w_k1_MSFT_s006 + 1.6713 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.717484 w_k1_ADM_s006
                                 + 0.572951 w_k1_AXP_s006
                                 + 0.559127 w_k1_DIS_s006
                                 + 0.620626 w_k1_FDX_s006
                                 + 1.01323 w_k1_IBM_s006
                                 + 0.624424 w_k1_INTC_s006
                                 + 0.863065 w_k1_KO_s006
                                 + 0.811688 w_k1_MAT_s006
                                 + 0.681266 w_k1_MSFT_s006
                                 + 0.844515 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.829323 w_k1_ADM_s006
                                 + 0.873517 w_k1_AXP_s006
                                 + 0.664126 w_k1_DIS_s006
                                 + 0.82523 w_k1_FDX_s006
                                 + 0.959037 w_k1_IBM_s006
                                 + 0.871367 w_k1_INTC_s006
                                 + 0.961582 w_k1_KO_s006
                                 + 1.17667 w_k1_MAT_s006
                                 + 0.762557 w_k1_MSFT_s006
                                 + 0.88197 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.3308 w_k1_ADM_s006
                                 + 1.67926 w_k1_AXP_s006
                                 + 1.40708 w_k1_DIS_s006
                                 + 1.31802 w_k1_FDX_s006 + 1.368 w_k1_IBM_s006
                                 + 0.838637 w_k1_INTC_s006
                                 + 1.47021 w_k1_KO_s006 + 2.20693 w_k1_MAT_s006
                                 + 1.12042 w_k1_MSFT_s006
                                 + 1.36826 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.13474 w_k1_ADM_s007
                                 + 1.47526 w_k1_AXP_s007
                                 + 1.42064 w_k1_DIS_s007 + 1.3105 w_k1_FDX_s007
                                 + 1.22199 w_k1_IBM_s007
                                 + 1.55541 w_k1_INTC_s007
                                 + 1.30426 w_k1_KO_s007 + 1.1729 w_k1_MAT_s007
                                 + 1.35336 w_k1_MSFT_s007
                                 + 1.27017 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.865976 w_k1_ADM_s007
                                 + 0.828395 w_k1_AXP_s007
                                 + 0.743173 w_k1_DIS_s007
                                 + 0.810346 w_k1_FDX_s007 + 1.085 w_k1_IBM_s007
                                 + 0.853595 w_k1_INTC_s007
                                 + 0.849785 w_k1_KO_s007 + 1.3451 w_k1_MAT_s007
                                 + 0.877265 w_k1_MSFT_s007
                                 + 0.868007 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.836251 w_k1_ADM_s007
                                 + 0.686699 w_k1_AXP_s007
                                 + 1.05962 w_k1_DIS_s007
                                 + 0.849372 w_k1_FDX_s007
                                 + 1.0046 w_k1_IBM_s007
                                 + 0.895252 w_k1_INTC_s007
                                 + 1.01356 w_k1_KO_s007
                                 + 0.779443 w_k1_MAT_s007
                                 + 1.06227 w_k1_MSFT_s007
                                 + 0.944262 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.83187 w_k1_ADM_s007
                                 + 0.782398 w_k1_AXP_s007
                                 + 0.656268 w_k1_DIS_s007
                                 + 0.782197 w_k1_FDX_s007
                                 + 0.810493 w_k1_IBM_s007
                                 + 0.820431 w_k1_INTC_s007
                                 + 0.787155 w_k1_KO_s007 + 0.7489 w_k1_MAT_s007
                                 + 0.796009 w_k1_MSFT_s007
                                 + 0.879828 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.08636 w_k1_ADM_s007
                                 + 1.09494 w_k1_AXP_s007 + 1.3588 w_k1_DIS_s007
                                 + 1.00791 w_k1_FDX_s007
                                 + 1.53088 w_k1_IBM_s007
                                 + 1.05225 w_k1_INTC_s007
                                 + 1.42216 w_k1_KO_s007 + 1.5253 w_k1_MAT_s007
                                 + 1.2386 w_k1_MSFT_s007 + 1.19996 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.902598 w_k1_ADM_s007
                                 + 0.579885 w_k1_AXP_s007
                                 + 0.701424 w_k1_DIS_s007
                                 + 0.803138 w_k1_FDX_s007
                                 + 0.651896 w_k1_IBM_s007
                                 + 0.926245 w_k1_INTC_s007
                                 + 0.697706 w_k1_KO_s007
                                 + 0.547239 w_k1_MAT_s007
                                 + 0.844819 w_k1_MSFT_s007
                                 + 0.914577 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.22965 w_k1_ADM_s007
                                 + 1.1149 w_k1_AXP_s007
                                 + 0.914138 w_k1_DIS_s007
                                 + 1.0967 w_k1_FDX_s007
                                 + 0.920246 w_k1_IBM_s007
                                 + 0.986658 w_k1_INTC_s007
                                 + 1.02302 w_k1_KO_s007 + 1.2533 w_k1_MAT_s007
                                 + 1.07924 w_k1_MSFT_s007
                                 + 1.02754 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.806092 w_k1_ADM_s007
                                 + 1.05061 w_k1_AXP_s007
                                 + 0.794098 w_k1_DIS_s007
                                 + 0.989172 w_k1_FDX_s007
                                 + 0.784308 w_k1_IBM_s007
                                 + 1.2339 w_k1_INTC_s007
                                 + 0.868786 w_k1_KO_s007
                                 + 0.828536 w_k1_MAT_s007
                                 + 1.00069 w_k1_MSFT_s007
                                 + 0.971329 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.08618 w_k1_ADM_s010
                                 + 1.30187 w_k1_AXP_s010
                                 + 1.76862 w_k1_DIS_s010
                                 + 1.25081 w_k1_FDX_s010
                                 + 1.53082 w_k1_IBM_s010
                                 + 1.21078 w_k1_INTC_s010 + 1.5686 w_k1_KO_s010
                                 + 1.97182 w_k1_MAT_s010
                                 + 1.34729 w_k1_MSFT_s010
                                 + 1.26042 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.44701 w_k1_ADM_s010
                                 + 1.56103 w_k1_AXP_s010
                                 + 1.56171 w_k1_DIS_s010
                                 + 1.47326 w_k1_FDX_s010
                                 + 0.974995 w_k1_IBM_s010
                                 + 1.15926 w_k1_INTC_s010
                                 + 1.27898 w_k1_KO_s010 + 1.2779 w_k1_MAT_s010
                                 + 1.37218 w_k1_MSFT_s010
                                 + 1.20404 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.985926 w_k1_ADM_s010
                                 + 0.956332 w_k1_AXP_s010
                                 + 0.817258 w_k1_DIS_s010
                                 + 0.963232 w_k1_FDX_s010
                                 + 0.856529 w_k1_IBM_s010
                                 + 0.875587 w_k1_INTC_s010
                                 + 0.842711 w_k1_KO_s010
                                 + 0.913714 w_k1_MAT_s010
                                 + 0.89284 w_k1_MSFT_s010
                                 + 0.871692 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.844761 w_k1_ADM_s010
                                 + 0.692109 w_k1_AXP_s010
                                 + 0.726813 w_k1_DIS_s010
                                 + 0.797427 w_k1_FDX_s010
                                 + 0.840144 w_k1_IBM_s010
                                 + 0.753734 w_k1_INTC_s010
                                 + 0.802742 w_k1_KO_s010
                                 + 0.683744 w_k1_MAT_s010
                                 + 0.85562 w_k1_MSFT_s010
                                 + 0.810448 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.07509 w_k1_ADM_s010
                                 + 1.00184 w_k1_AXP_s010
                                 + 1.26585 w_k1_DIS_s010
                                 + 1.00383 w_k1_FDX_s010
                                 + 1.34036 w_k1_IBM_s010
                                 + 0.887999 w_k1_INTC_s010 + 1.37 w_k1_KO_s010
                                 + 1.53072 w_k1_MAT_s010
                                 + 1.02711 w_k1_MSFT_s010
                                 + 1.12657 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.977429 w_k1_ADM_s010
                                 + 1.01855 w_k1_AXP_s010 + 1.4363 w_k1_DIS_s010
                                 + 1.22941 w_k1_FDX_s010
                                 + 0.889205 w_k1_IBM_s010
                                 + 1.23989 w_k1_INTC_s010
                                 + 1.26221 w_k1_KO_s010
                                 + 0.866279 w_k1_MAT_s010
                                 + 1.28891 w_k1_MSFT_s010
                                 + 1.14682 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.10177 w_k1_ADM_s010
                                 + 1.105 w_k1_AXP_s010 + 1.08718 w_k1_DIS_s010
                                 + 1.18884 w_k1_FDX_s010
                                 + 0.703474 w_k1_IBM_s010
                                 + 1.08906 w_k1_INTC_s010
                                 + 0.948302 w_k1_KO_s010
                                 + 0.663494 w_k1_MAT_s010
                                 + 1.04752 w_k1_MSFT_s010
                                 + 0.964394 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.928289 w_k1_ADM_s010
                                 + 1.34242 w_k1_AXP_s010
                                 + 0.950799 w_k1_DIS_s010
                                 + 1.08536 w_k1_FDX_s010
                                 + 1.02444 w_k1_IBM_s010
                                 + 0.966183 w_k1_INTC_s010
                                 + 1.15599 w_k1_KO_s010 + 1.60087 w_k1_MAT_s010
                                 + 1.06151 w_k1_MSFT_s010 + 1.0141 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.2235 w_k1_ADM_s011
                                 + 0.942769 w_k1_AXP_s011
                                 + 1.19143 w_k1_DIS_s011
                                 + 1.08179 w_k1_FDX_s011
                                 + 1.13993 w_k1_IBM_s011
                                 + 1.23198 w_k1_INTC_s011
                                 + 1.08023 w_k1_KO_s011
                                 + 0.975766 w_k1_MAT_s011
                                 + 1.18714 w_k1_MSFT_s011
                                 + 1.00161 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.858575 w_k1_ADM_s011
                                 + 1.58149 w_k1_AXP_s011
                                 + 1.54136 w_k1_DIS_s011
                                 + 1.34346 w_k1_FDX_s011
                                 + 1.44778 w_k1_IBM_s011 + 1.498 w_k1_INTC_s011
                                 + 1.44881 w_k1_KO_s011 + 2.06936 w_k1_MAT_s011
                                 + 1.44316 w_k1_MSFT_s011
                                 + 1.12325 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.96462 w_k1_ADM_s011
                                 + 0.873603 w_k1_AXP_s011
                                 + 0.710926 w_k1_DIS_s011
                                 + 0.874672 w_k1_FDX_s011
                                 + 0.845074 w_k1_IBM_s011
                                 + 0.925835 w_k1_INTC_s011
                                 + 0.927682 w_k1_KO_s011
                                 + 0.876651 w_k1_MAT_s011
                                 + 0.839664 w_k1_MSFT_s011
                                 + 0.857574 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.846275 w_k1_ADM_s011
                                 + 1.01701 w_k1_AXP_s011
                                 + 1.00489 w_k1_DIS_s011
                                 + 1.01153 w_k1_FDX_s011
                                 + 1.26342 w_k1_IBM_s011
                                 + 1.08279 w_k1_INTC_s011
                                 + 1.11965 w_k1_KO_s011
                                 + 0.985694 w_k1_MAT_s011
                                 + 1.01747 w_k1_MSFT_s011
                                 + 0.99533 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.862265 w_k1_ADM_s011
                                 + 1.17542 w_k1_AXP_s011
                                 + 1.29323 w_k1_DIS_s011
                                 + 1.20309 w_k1_FDX_s011
                                 + 1.03594 w_k1_IBM_s011
                                 + 1.45676 w_k1_INTC_s011
                                 + 1.13437 w_k1_KO_s011 + 1.18721 w_k1_MAT_s011
                                 + 1.19008 w_k1_MSFT_s011
                                 + 1.04231 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.789926 w_k1_ADM_s011
                                 + 0.612645 w_k1_AXP_s011
                                 + 0.703397 w_k1_DIS_s011
                                 + 0.860453 w_k1_FDX_s011
                                 + 0.716263 w_k1_IBM_s011
                                 + 0.941945 w_k1_INTC_s011
                                 + 0.831373 w_k1_KO_s011
                                 + 0.491004 w_k1_MAT_s011
                                 + 0.890039 w_k1_MSFT_s011
                                 + 0.760738 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.04054 w_k1_ADM_s011
                                 + 1.16404 w_k1_AXP_s011
                                 + 0.976675 w_k1_DIS_s011
                                 + 1.06955 w_k1_FDX_s011
                                 + 0.912899 w_k1_IBM_s011
                                 + 1.15358 w_k1_INTC_s011
                                 + 1.01339 w_k1_KO_s011 + 1.07415 w_k1_MAT_s011
                                 + 1.00069 w_k1_MSFT_s011
                                 + 1.02809 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.849765 w_k1_ADM_s011
                                 + 0.710994 w_k1_AXP_s011
                                 + 0.840988 w_k1_DIS_s011
                                 + 0.858238 w_k1_FDX_s011
                                 + 1.1226 w_k1_IBM_s011
                                 + 1.07394 w_k1_INTC_s011
                                 + 1.02903 w_k1_KO_s011
                                 + 0.800492 w_k1_MAT_s011
                                 + 0.964995 w_k1_MSFT_s011
                                 + 0.890721 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.04358 w_k1_ADM_s012
                                 + 0.878929 w_k1_AXP_s012
                                 + 0.817041 w_k1_DIS_s012
                                 + 0.934655 w_k1_FDX_s012
                                 + 0.83231 w_k1_IBM_s012
                                 + 0.766598 w_k1_INTC_s012
                                 + 0.769042 w_k1_KO_s012
                                 + 1.15362 w_k1_MAT_s012
                                 + 0.896867 w_k1_MSFT_s012
                                 + 0.928841 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.01113 w_k1_ADM_s012
                                 + 0.762404 w_k1_AXP_s012
                                 + 0.909947 w_k1_DIS_s012
                                 + 0.912843 w_k1_FDX_s012
                                 + 0.606447 w_k1_IBM_s012
                                 + 0.777763 w_k1_INTC_s012
                                 + 0.752054 w_k1_KO_s012
                                 + 0.669946 w_k1_MAT_s012
                                 + 0.915756 w_k1_MSFT_s012
                                 + 0.870532 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.37661 w_k1_ADM_s012
                                 + 1.18184 w_k1_AXP_s012
                                 + 1.80142 w_k1_DIS_s012
                                 + 1.33349 w_k1_FDX_s012
                                 + 1.41241 w_k1_IBM_s012
                                 + 0.943867 w_k1_INTC_s012
                                 + 1.43007 w_k1_KO_s012 + 2.20987 w_k1_MAT_s012
                                 + 1.3544 w_k1_MSFT_s012 + 1.28168 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.858207 w_k1_ADM_s012
                                 + 1.00196 w_k1_AXP_s012
                                 + 0.803708 w_k1_DIS_s012
                                 + 0.793164 w_k1_FDX_s012
                                 + 1.5814 w_k1_IBM_s012
                                 + 0.755432 w_k1_INTC_s012
                                 + 0.989937 w_k1_KO_s012
                                 + 1.70334 w_k1_MAT_s012
                                 + 0.809844 w_k1_MSFT_s012
                                 + 0.98295 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.11717 w_k1_ADM_s012
                                 + 1.32426 w_k1_AXP_s012 + 1.1906 w_k1_DIS_s012
                                 + 1.16271 w_k1_FDX_s012
                                 + 1.14025 w_k1_IBM_s012
                                 + 0.926962 w_k1_INTC_s012
                                 + 1.0708 w_k1_KO_s012 + 1.98142 w_k1_MAT_s012
                                 + 0.963615 w_k1_MSFT_s012
                                 + 1.0999 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.1592 w_k1_ADM_s012
                                 + 1.17911 w_k1_AXP_s012
                                 + 1.15815 w_k1_DIS_s012
                                 + 1.30263 w_k1_FDX_s012
                                 + 0.727078 w_k1_IBM_s012
                                 + 0.96147 w_k1_INTC_s012
                                 + 0.793235 w_k1_KO_s012
                                 + 0.639371 w_k1_MAT_s012
                                 + 0.998226 w_k1_MSFT_s012
                                 + 1.10793 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.930705 w_k1_ADM_s012
                                 + 0.756189 w_k1_AXP_s012
                                 + 0.982166 w_k1_DIS_s012
                                 + 0.862406 w_k1_FDX_s012
                                 + 1.29349 w_k1_IBM_s012
                                 + 0.759261 w_k1_INTC_s012
                                 + 0.977004 w_k1_KO_s012 + 1.094 w_k1_MAT_s012
                                 + 0.847765 w_k1_MSFT_s012
                                 + 0.947132 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.64355 w_k1_ADM_s012
                                 + 1.53518 w_k1_AXP_s012
                                 + 1.68581 w_k1_DIS_s012
                                 + 1.56502 w_k1_FDX_s012
                                 + 0.968867 w_k1_IBM_s012
                                 + 1.19156 w_k1_INTC_s012
                                 + 1.05397 w_k1_KO_s012
                                 + 0.909877 w_k1_MAT_s012
                                 + 1.30521 w_k1_MSFT_s012
                                 + 1.22538 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.786403 w_k1_ADM_s013
                                 + 1.00642 w_k1_AXP_s013
                                 + 0.825949 w_k1_DIS_s013
                                 + 0.858733 w_k1_FDX_s013
                                 + 1.13945 w_k1_IBM_s013
                                 + 1.05678 w_k1_INTC_s013
                                 + 1.04332 w_k1_KO_s013 + 1.08587 w_k1_MAT_s013
                                 + 1.00841 w_k1_MSFT_s013
                                 + 1.00054 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.985916 w_k1_ADM_s013
                                 + 1.09792 w_k1_AXP_s013
                                 + 1.18498 w_k1_DIS_s013
                                 + 1.07658 w_k1_FDX_s013
                                 + 1.23802 w_k1_IBM_s013
                                 + 1.20928 w_k1_INTC_s013 + 1.0848 w_k1_KO_s013
                                 + 0.927812 w_k1_MAT_s013
                                 + 1.14659 w_k1_MSFT_s013
                                 + 1.03068 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.990928 w_k1_ADM_s013
                                 + 0.868151 w_k1_AXP_s013
                                 + 1.04042 w_k1_DIS_s013
                                 + 1.00304 w_k1_FDX_s013
                                 + 0.74259 w_k1_IBM_s013
                                 + 1.19444 w_k1_INTC_s013
                                 + 0.932665 w_k1_KO_s013
                                 + 0.735733 w_k1_MAT_s013
                                 + 1.14656 w_k1_MSFT_s013
                                 + 0.941174 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.775357 w_k1_ADM_s013
                                 + 0.577055 w_k1_AXP_s013
                                 + 0.691909 w_k1_DIS_s013
                                 + 0.72713 w_k1_FDX_s013
                                 + 0.848952 w_k1_IBM_s013
                                 + 0.948099 w_k1_INTC_s013
                                 + 0.747913 w_k1_KO_s013
                                 + 0.602876 w_k1_MAT_s013
                                 + 0.919212 w_k1_MSFT_s013
                                 + 0.794551 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.896304 w_k1_ADM_s013
                                 + 0.856846 w_k1_AXP_s013
                                 + 0.673357 w_k1_DIS_s013
                                 + 0.88636 w_k1_FDX_s013
                                 + 0.758206 w_k1_IBM_s013
                                 + 0.93919 w_k1_INTC_s013
                                 + 0.768559 w_k1_KO_s013
                                 + 0.622734 w_k1_MAT_s013
                                 + 0.81805 w_k1_MSFT_s013
                                 + 0.838399 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.06356 w_k1_ADM_s013
                                 + 1.57864 w_k1_AXP_s013
                                 + 1.24848 w_k1_DIS_s013
                                 + 1.17558 w_k1_FDX_s013 + 1.192 w_k1_IBM_s013
                                 + 0.97313 w_k1_INTC_s013
                                 + 1.08254 w_k1_KO_s013 + 1.51301 w_k1_MAT_s013
                                 + 1.13419 w_k1_MSFT_s013
                                 + 1.08375 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.975997 w_k1_ADM_s013
                                 + 1.13964 w_k1_AXP_s013
                                 + 0.790107 w_k1_DIS_s013
                                 + 0.999517 w_k1_FDX_s013
                                 + 0.957111 w_k1_IBM_s013
                                 + 0.978138 w_k1_INTC_s013
                                 + 0.799439 w_k1_KO_s013
                                 + 1.19949 w_k1_MAT_s013
                                 + 1.05574 w_k1_MSFT_s013
                                 + 1.01505 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.964672 w_k1_ADM_s013
                                 + 1.54358 w_k1_AXP_s013 + 1.6753 w_k1_DIS_s013
                                 + 1.35461 w_k1_FDX_s013
                                 + 1.10974 w_k1_IBM_s013
                                 + 1.47155 w_k1_INTC_s013
                                 + 1.28577 w_k1_KO_s013 + 1.49861 w_k1_MAT_s013
                                 + 1.68241 w_k1_MSFT_s013
                                 + 1.12166 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.1439 w_k1_ADM_s014
                                 + 1.05551 w_k1_AXP_s014
                                 + 1.44741 w_k1_DIS_s014
                                 + 1.24834 w_k1_FDX_s014
                                 + 1.30429 w_k1_IBM_s014
                                 + 1.0857 w_k1_INTC_s014 + 1.33762 w_k1_KO_s014
                                 + 1.14593 w_k1_MAT_s014
                                 + 1.22164 w_k1_MSFT_s014
                                 + 1.25546 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.09942 w_k1_ADM_s014
                                 + 0.874282 w_k1_AXP_s014
                                 + 1.18689 w_k1_DIS_s014
                                 + 1.11429 w_k1_FDX_s014
                                 + 0.984656 w_k1_IBM_s014
                                 + 0.96978 w_k1_INTC_s014
                                 + 1.06362 w_k1_KO_s014 + 1.06819 w_k1_MAT_s014
                                 + 1.0452 w_k1_MSFT_s014 + 1.1864 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.02519 w_k1_ADM_s014
                                 + 0.882431 w_k1_AXP_s014
                                 + 0.712591 w_k1_DIS_s014
                                 + 0.982255 w_k1_FDX_s014
                                 + 0.752688 w_k1_IBM_s014
                                 + 0.893103 w_k1_INTC_s014
                                 + 0.82914 w_k1_KO_s014
                                 + 0.930826 w_k1_MAT_s014
                                 + 0.933882 w_k1_MSFT_s014
                                 + 1.04564 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.82954 w_k1_ADM_s014
                                 + 0.537097 w_k1_AXP_s014
                                 + 0.573993 w_k1_DIS_s014
                                 + 0.707859 w_k1_FDX_s014
                                 + 0.656532 w_k1_IBM_s014
                                 + 0.793371 w_k1_INTC_s014
                                 + 0.800695 w_k1_KO_s014
                                 + 0.620471 w_k1_MAT_s014
                                 + 0.823969 w_k1_MSFT_s014
                                 + 0.806817 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.09747 w_k1_ADM_s014
                                 + 1.00553 w_k1_AXP_s014
                                 + 1.02718 w_k1_DIS_s014
                                 + 0.961997 w_k1_FDX_s014
                                 + 1.23797 w_k1_IBM_s014
                                 + 0.917043 w_k1_INTC_s014
                                 + 1.07243 w_k1_KO_s014 + 1.13676 w_k1_MAT_s014
                                 + 1.03089 w_k1_MSFT_s014
                                 + 1.17139 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.982504 w_k1_ADM_s014
                                 + 1.45859 w_k1_AXP_s014
                                 + 1.25577 w_k1_DIS_s014
                                 + 1.20075 w_k1_FDX_s014
                                 + 1.66199 w_k1_IBM_s014
                                 + 1.24954 w_k1_INTC_s014
                                 + 1.34328 w_k1_KO_s014 + 2.02391 w_k1_MAT_s014
                                 + 1.18323 w_k1_MSFT_s014
                                 + 1.16676 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.10094 w_k1_ADM_s014
                                 + 1.20235 w_k1_AXP_s014
                                 + 1.07614 w_k1_DIS_s014
                                 + 1.15549 w_k1_FDX_s014
                                 + 0.706691 w_k1_IBM_s014
                                 + 0.927203 w_k1_INTC_s014
                                 + 0.984092 w_k1_KO_s014
                                 + 0.918514 w_k1_MAT_s014
                                 + 1.01828 w_k1_MSFT_s014
                                 + 1.13822 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.845961 w_k1_ADM_s014
                                 + 0.723145 w_k1_AXP_s014
                                 + 0.658826 w_k1_DIS_s014
                                 + 0.771959 w_k1_FDX_s014
                                 + 0.978457 w_k1_IBM_s014
                                 + 0.687737 w_k1_INTC_s014
                                 + 0.903323 w_k1_KO_s014
                                 + 0.925944 w_k1_MAT_s014
                                 + 0.778886 w_k1_MSFT_s014
                                 + 0.932404 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.04413 w_k1_ADM_s015
                                 + 1.16179 w_k1_AXP_s015
                                 + 1.18479 w_k1_DIS_s015
                                 + 0.993648 w_k1_FDX_s015
                                 + 1.49751 w_k1_IBM_s015
                                 + 1.01442 w_k1_INTC_s015
                                 + 1.24541 w_k1_KO_s015 + 2.04397 w_k1_MAT_s015
                                 + 0.909389 w_k1_MSFT_s015
                                 + 1.14231 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.837213 w_k1_ADM_s015
                                 + 1.1621 w_k1_AXP_s015
                                 + 0.738123 w_k1_DIS_s015
                                 + 0.871795 w_k1_FDX_s015
                                 + 1.04004 w_k1_IBM_s015
                                 + 0.877144 w_k1_INTC_s015
                                 + 0.958099 w_k1_KO_s015
                                 + 1.55035 w_k1_MAT_s015
                                 + 0.775788 w_k1_MSFT_s015
                                 + 0.959779 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.931092 w_k1_ADM_s015
                                 + 1.06712 w_k1_AXP_s015
                                 + 0.966933 w_k1_DIS_s015
                                 + 1.0478 w_k1_FDX_s015 + 0.75988 w_k1_IBM_s015
                                 + 0.867087 w_k1_INTC_s015
                                 + 0.840304 w_k1_KO_s015
                                 + 0.922057 w_k1_MAT_s015
                                 + 0.899628 w_k1_MSFT_s015
                                 + 0.921934 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.15654 w_k1_ADM_s015
                                 + 1.16405 w_k1_AXP_s015
                                 + 1.58441 w_k1_DIS_s015 + 1.2467 w_k1_FDX_s015
                                 + 1.21072 w_k1_IBM_s015
                                 + 1.17112 w_k1_INTC_s015
                                 + 1.42425 w_k1_KO_s015 + 1.4914 w_k1_MAT_s015
                                 + 1.17659 w_k1_MSFT_s015
                                 + 1.16311 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.769467 w_k1_ADM_s015
                                 + 0.757357 w_k1_AXP_s015
                                 + 0.860759 w_k1_DIS_s015
                                 + 0.712186 w_k1_FDX_s015
                                 + 1.37333 w_k1_IBM_s015
                                 + 0.73832 w_k1_INTC_s015
                                 + 1.05342 w_k1_KO_s015 + 1.98058 w_k1_MAT_s015
                                 + 0.731589 w_k1_MSFT_s015
                                 + 0.900598 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.949445 w_k1_ADM_s015
                                 + 1.12132 w_k1_AXP_s015
                                 + 1.13938 w_k1_DIS_s015
                                 + 1.04069 w_k1_FDX_s015
                                 + 0.987568 w_k1_IBM_s015
                                 + 0.878895 w_k1_INTC_s015
                                 + 0.947679 w_k1_KO_s015
                                 + 1.43287 w_k1_MAT_s015
                                 + 0.913571 w_k1_MSFT_s015
                                 + 1.01819 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.13127 w_k1_ADM_s015
                                 + 1.56583 w_k1_AXP_s015
                                 + 1.55731 w_k1_DIS_s015
                                 + 1.38174 w_k1_FDX_s015
                                 + 1.03415 w_k1_IBM_s015
                                 + 1.10896 w_k1_INTC_s015
                                 + 1.20456 w_k1_KO_s015 + 2.10375 w_k1_MAT_s015
                                 + 1.17937 w_k1_MSFT_s015
                                 + 1.29711 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.674145 w_k1_ADM_s015
                                 + 0.788085 w_k1_AXP_s015
                                 + 0.736971 w_k1_DIS_s015
                                 + 0.742822 w_k1_FDX_s015
                                 + 0.810982 w_k1_IBM_s015
                                 + 0.763078 w_k1_INTC_s015
                                 + 0.819357 w_k1_KO_s015
                                 + 0.972625 w_k1_MAT_s015
                                 + 0.766623 w_k1_MSFT_s015
                                 + 0.866067 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.46817 w_k1_ADM_s016
                                 + 1.51317 w_k1_AXP_s016
                                 + 1.81679 w_k1_DIS_s016
                                 + 1.55257 w_k1_FDX_s016
                                 + 0.993123 w_k1_IBM_s016
                                 + 1.3577 w_k1_INTC_s016 + 1.37639 w_k1_KO_s016
                                 + 1.22059 w_k1_MAT_s016
                                 + 1.36935 w_k1_MSFT_s016
                                 + 1.30999 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.750164 w_k1_ADM_s016
                                 + 0.734974 w_k1_AXP_s016
                                 + 0.737175 w_k1_DIS_s016
                                 + 0.778876 w_k1_FDX_s016
                                 + 0.93157 w_k1_IBM_s016
                                 + 0.7009 w_k1_INTC_s016
                                 + 0.926132 w_k1_KO_s016
                                 + 1.01232 w_k1_MAT_s016
                                 + 0.779879 w_k1_MSFT_s016
                                 + 0.845977 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.15764 w_k1_ADM_s016
                                 + 1.07945 w_k1_AXP_s016
                                 + 1.28295 w_k1_DIS_s016 + 1.2884 w_k1_FDX_s016
                                 + 0.719713 w_k1_IBM_s016
                                 + 1.09168 w_k1_INTC_s016
                                 + 1.10846 w_k1_KO_s016
                                 + 0.913853 w_k1_MAT_s016
                                 + 1.15184 w_k1_MSFT_s016
                                 + 1.02226 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.774518 w_k1_ADM_s016
                                 + 0.733604 w_k1_AXP_s016
                                 + 0.707818 w_k1_DIS_s016
                                 + 0.774595 w_k1_FDX_s016
                                 + 0.614565 w_k1_IBM_s016
                                 + 0.685453 w_k1_INTC_s016
                                 + 0.851322 w_k1_KO_s016
                                 + 0.979006 w_k1_MAT_s016
                                 + 0.805253 w_k1_MSFT_s016
                                 + 0.864174 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.16899 w_k1_ADM_s016
                                 + 1.18428 w_k1_AXP_s016
                                 + 1.33286 w_k1_DIS_s016
                                 + 1.05396 w_k1_FDX_s016
                                 + 1.78193 w_k1_IBM_s016
                                 + 0.802514 w_k1_INTC_s016
                                 + 1.47865 w_k1_KO_s016 + 1.89472 w_k1_MAT_s016
                                 + 0.888 w_k1_MSFT_s016 + 1.08435 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.005 w_k1_ADM_s016
                                 + 0.981482 w_k1_AXP_s016
                                 + 0.955302 w_k1_DIS_s016
                                 + 0.869869 w_k1_FDX_s016
                                 + 1.32716 w_k1_IBM_s016
                                 + 0.786844 w_k1_INTC_s016
                                 + 1.17423 w_k1_KO_s016 + 1.47832 w_k1_MAT_s016
                                 + 0.921624 w_k1_MSFT_s016
                                 + 0.952986 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.899165 w_k1_ADM_s016
                                 + 1.10568 w_k1_AXP_s016 + 1.1167 w_k1_DIS_s016
                                 + 1.08099 w_k1_FDX_s016
                                 + 0.889198 w_k1_IBM_s016
                                 + 1.04304 w_k1_INTC_s016
                                 + 1.16438 w_k1_KO_s016 + 1.31089 w_k1_MAT_s016
                                 + 1.02362 w_k1_MSFT_s016
                                 + 1.00441 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.086 w_k1_ADM_s016
                                 + 1.57501 w_k1_AXP_s016
                                 + 1.80983 w_k1_DIS_s016
                                 + 1.32182 w_k1_FDX_s016
                                 + 1.27007 w_k1_IBM_s016
                                 + 1.20759 w_k1_INTC_s016
                                 + 1.45523 w_k1_KO_s016 + 1.82738 w_k1_MAT_s016
                                 + 1.29484 w_k1_MSFT_s016
                                 + 1.14178 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.13538 w_k1_ADM_s017
                                 + 1.27453 w_k1_AXP_s017
                                 + 1.58051 w_k1_DIS_s017
                                 + 1.20957 w_k1_FDX_s017 + 1.5286 w_k1_IBM_s017
                                 + 1.29213 w_k1_INTC_s017
                                 + 1.44923 w_k1_KO_s017 + 1.73392 w_k1_MAT_s017
                                 + 1.32922 w_k1_MSFT_s017 + 1.219 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.702662 w_k1_ADM_s017
                                 + 0.79617 w_k1_AXP_s017
                                 + 0.756635 w_k1_DIS_s017
                                 + 0.756307 w_k1_FDX_s017
                                 + 1.03605 w_k1_IBM_s017
                                 + 0.779865 w_k1_INTC_s017
                                 + 0.854544 w_k1_KO_s017
                                 + 0.933821 w_k1_MAT_s017
                                 + 0.824985 w_k1_MSFT_s017
                                 + 0.985156 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.11551 w_k1_ADM_s017
                                 + 1.09944 w_k1_AXP_s017
                                 + 1.33423 w_k1_DIS_s017
                                 + 1.22997 w_k1_FDX_s017
                                 + 0.796568 w_k1_IBM_s017
                                 + 1.3381 w_k1_INTC_s017
                                 + 0.924922 w_k1_KO_s017
                                 + 0.651972 w_k1_MAT_s017
                                 + 1.06089 w_k1_MSFT_s017
                                 + 1.01618 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.00787 w_k1_ADM_s017
                                 + 1.12475 w_k1_AXP_s017
                                 + 0.849527 w_k1_DIS_s017
                                 + 1.04178 w_k1_FDX_s017
                                 + 0.851914 w_k1_IBM_s017
                                 + 1.16224 w_k1_INTC_s017
                                 + 0.79937 w_k1_KO_s017
                                 + 0.918853 w_k1_MAT_s017
                                 + 0.920524 w_k1_MSFT_s017
                                 + 0.942896 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.944525 w_k1_ADM_s017
                                 + 1.0592 w_k1_AXP_s017
                                 + 0.928606 w_k1_DIS_s017
                                 + 0.90343 w_k1_FDX_s017
                                 + 1.45175 w_k1_IBM_s017
                                 + 1.04627 w_k1_INTC_s017
                                 + 1.10043 w_k1_KO_s017 + 1.45624 w_k1_MAT_s017
                                 + 0.901013 w_k1_MSFT_s017
                                 + 0.971631 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.24001 w_k1_ADM_s017
                                 + 1.53369 w_k1_AXP_s017
                                 + 1.10632 w_k1_DIS_s017
                                 + 1.13487 w_k1_FDX_s017
                                 + 1.18783 w_k1_IBM_s017
                                 + 1.30131 w_k1_INTC_s017
                                 + 0.945377 w_k1_KO_s017
                                 + 1.25131 w_k1_MAT_s017
                                 + 0.943307 w_k1_MSFT_s017
                                 + 1.05216 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.772154 w_k1_ADM_s017
                                 + 0.698587 w_k1_AXP_s017
                                 + 0.598434 w_k1_DIS_s017
                                 + 0.835337 w_k1_FDX_s017
                                 + 0.620253 w_k1_IBM_s017
                                 + 0.898387 w_k1_INTC_s017
                                 + 0.568125 w_k1_KO_s017
                                 + 0.623436 w_k1_MAT_s017
                                 + 0.773635 w_k1_MSFT_s017
                                 + 0.801031 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.898426 w_k1_ADM_s017
                                 + 0.869398 w_k1_AXP_s017
                                 + 0.967767 w_k1_DIS_s017
                                 + 0.962044 w_k1_FDX_s017
                                 + 0.983456 w_k1_IBM_s017
                                 + 1.0022 w_k1_INTC_s017 + 0.99942 w_k1_KO_s017
                                 + 1.07106 w_k1_MAT_s017
                                 + 1.00184 w_k1_MSFT_s017
                                 + 0.968563 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.04006 w_k1_ADM_s100
                                 + 1.03891 w_k1_AXP_s100 + 1.2089 w_k1_DIS_s100
                                 + 1.22378 w_k1_FDX_s100
                                 + 0.74815 w_k1_IBM_s100
                                 + 0.976565 w_k1_INTC_s100
                                 + 0.965989 w_k1_KO_s100
                                 + 0.850163 w_k1_MAT_s100
                                 + 1.14799 w_k1_MSFT_s100
                                 + 1.13475 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.852413 w_k1_ADM_s100
                                 + 0.96133 w_k1_AXP_s100
                                 + 0.943651 w_k1_DIS_s100
                                 + 0.931228 w_k1_FDX_s100
                                 + 1.11242 w_k1_IBM_s100
                                 + 0.940031 w_k1_INTC_s100
                                 + 0.941351 w_k1_KO_s100
                                 + 1.07793 w_k1_MAT_s100
                                 + 0.878908 w_k1_MSFT_s100
                                 + 1.0628 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.845575 w_k1_ADM_s100
                                 + 0.884175 w_k1_AXP_s100
                                 + 0.736853 w_k1_DIS_s100
                                 + 0.774036 w_k1_FDX_s100
                                 + 1.26308 w_k1_IBM_s100
                                 + 0.697645 w_k1_INTC_s100
                                 + 0.907499 w_k1_KO_s100
                                 + 1.23922 w_k1_MAT_s100
                                 + 0.625363 w_k1_MSFT_s100
                                 + 0.850116 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.953606 w_k1_ADM_s100
                                 + 1.38953 w_k1_AXP_s100 + 1.3406 w_k1_DIS_s100
                                 + 1.07262 w_k1_FDX_s100 + 1.7507 w_k1_IBM_s100
                                 + 1.05428 w_k1_INTC_s100
                                 + 1.35743 w_k1_KO_s100 + 2.1111 w_k1_MAT_s100
                                 + 1.05431 w_k1_MSFT_s100
                                 + 1.11302 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.10961 w_k1_ADM_s100
                                 + 1.10234 w_k1_AXP_s100
                                 + 1.41525 w_k1_DIS_s100
                                 + 1.27266 w_k1_FDX_s100
                                 + 1.04383 w_k1_IBM_s100
                                 + 1.38674 w_k1_INTC_s100
                                 + 1.33058 w_k1_KO_s100 + 1.01291 w_k1_MAT_s100
                                 + 1.27205 w_k1_MSFT_s100
                                 + 1.20729 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.05473 w_k1_ADM_s100
                                 + 1.01055 w_k1_AXP_s100
                                 + 0.811492 w_k1_DIS_s100
                                 + 1.03663 w_k1_FDX_s100
                                 + 0.809537 w_k1_IBM_s100
                                 + 0.879264 w_k1_INTC_s100
                                 + 0.820195 w_k1_KO_s100
                                 + 1.02461 w_k1_MAT_s100
                                 + 0.767495 w_k1_MSFT_s100
                                 + 1.03168 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.698329 w_k1_ADM_s100
                                 + 0.672646 w_k1_AXP_s100
                                 + 0.674981 w_k1_DIS_s100
                                 + 0.813589 w_k1_FDX_s100
                                 + 0.723768 w_k1_IBM_s100
                                 + 0.688586 w_k1_INTC_s100
                                 + 0.790513 w_k1_KO_s100
                                 + 0.772635 w_k1_MAT_s100
                                 + 0.799231 w_k1_MSFT_s100
                                 + 0.790837 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.10143 w_k1_ADM_s100
                                 + 1.46575 w_k1_AXP_s100
                                 + 1.45882 w_k1_DIS_s100
                                 + 1.28175 w_k1_FDX_s100
                                 + 1.35023 w_k1_IBM_s100
                                 + 1.10662 w_k1_INTC_s100
                                 + 1.37163 w_k1_KO_s100 + 1.29271 w_k1_MAT_s100
                                 + 1.10448 w_k1_MSFT_s100
                                 + 1.18794 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.32463 w_k1_ADM_s101
                                 + 1.46834 w_k1_AXP_s101
                                 + 1.47039 w_k1_DIS_s101
                                 + 1.35663 w_k1_FDX_s101 + 0.8341 w_k1_IBM_s101
                                 + 0.989003 w_k1_INTC_s101
                                 + 1.10067 w_k1_KO_s101 + 1.16336 w_k1_MAT_s101
                                 + 1.21144 w_k1_MSFT_s101
                                 + 1.19996 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.811694 w_k1_ADM_s101
                                 + 0.672505 w_k1_AXP_s101
                                 + 0.68928 w_k1_DIS_s101
                                 + 0.783428 w_k1_FDX_s101
                                 + 0.661453 w_k1_IBM_s101
                                 + 0.859309 w_k1_INTC_s101
                                 + 0.773437 w_k1_KO_s101
                                 + 0.507452 w_k1_MAT_s101
                                 + 0.764978 w_k1_MSFT_s101
                                 + 0.746928 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.884853 w_k1_ADM_s101
                                 + 0.901599 w_k1_AXP_s101
                                 + 0.913626 w_k1_DIS_s101
                                 + 0.801054 w_k1_FDX_s101
                                 + 1.33748 w_k1_IBM_s101
                                 + 0.745872 w_k1_INTC_s101
                                 + 1.04222 w_k1_KO_s101 + 1.3871 w_k1_MAT_s101
                                 + 0.884483 w_k1_MSFT_s101
                                 + 0.979705 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.30245 w_k1_ADM_s101
                                 + 1.3837 w_k1_AXP_s101 + 1.54194 w_k1_DIS_s101
                                 + 1.44806 w_k1_FDX_s101
                                 + 1.09619 w_k1_IBM_s101
                                 + 0.979857 w_k1_INTC_s101
                                 + 1.15107 w_k1_KO_s101
                                 + 0.994658 w_k1_MAT_s101
                                 + 1.11977 w_k1_MSFT_s101 + 1.1639 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.990279 w_k1_ADM_s101
                                 + 1.1184 w_k1_AXP_s101 + 0.99325 w_k1_DIS_s101
                                 + 0.974915 w_k1_FDX_s101
                                 + 1.16741 w_k1_IBM_s101
                                 + 1.0784 w_k1_INTC_s101 + 1.07888 w_k1_KO_s101
                                 + 1.05366 w_k1_MAT_s101
                                 + 0.989164 w_k1_MSFT_s101
                                 + 1.09701 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.09323 w_k1_ADM_s101
                                 + 0.858538 w_k1_AXP_s101
                                 + 1.15677 w_k1_DIS_s101
                                 + 1.03525 w_k1_FDX_s101
                                 + 0.836939 w_k1_IBM_s101
                                 + 0.900913 w_k1_INTC_s101
                                 + 1.00344 w_k1_KO_s101
                                 + 0.773156 w_k1_MAT_s101
                                 + 1.02931 w_k1_MSFT_s101
                                 + 1.00357 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.25346 w_k1_ADM_s101
                                 + 1.45732 w_k1_AXP_s101
                                 + 1.44005 w_k1_DIS_s101
                                 + 1.22185 w_k1_FDX_s101
                                 + 1.45907 w_k1_IBM_s101
                                 + 1.03149 w_k1_INTC_s101
                                 + 1.32406 w_k1_KO_s101 + 1.25404 w_k1_MAT_s101
                                 + 1.15529 w_k1_MSFT_s101
                                 + 1.30844 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.03627 w_k1_ADM_s101
                                 + 0.967288 w_k1_AXP_s101
                                 + 0.763618 w_k1_DIS_s101
                                 + 0.934961 w_k1_FDX_s101
                                 + 0.774557 w_k1_IBM_s101
                                 + 0.890386 w_k1_INTC_s101
                                 + 0.823655 w_k1_KO_s101
                                 + 0.749116 w_k1_MAT_s101
                                 + 0.828391 w_k1_MSFT_s101
                                 + 0.971406 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.04842 w_k1_ADM_s102
                                 + 0.855137 w_k1_AXP_s102
                                 + 0.868084 w_k1_DIS_s102
                                 + 0.956592 w_k1_FDX_s102
                                 + 0.824171 w_k1_IBM_s102
                                 + 0.752188 w_k1_INTC_s102
                                 + 0.835775 w_k1_KO_s102
                                 + 0.83418 w_k1_MAT_s102
                                 + 0.777344 w_k1_MSFT_s102
                                 + 0.976498 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.956877 w_k1_ADM_s102
                                 + 1.3115 w_k1_AXP_s102
                                 + 0.906292 w_k1_DIS_s102
                                 + 0.955452 w_k1_FDX_s102
                                 + 0.914706 w_k1_IBM_s102
                                 + 0.781056 w_k1_INTC_s102
                                 + 0.91789 w_k1_KO_s102 + 1.25878 w_k1_MAT_s102
                                 + 0.840911 w_k1_MSFT_s102
                                 + 0.932413 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.18389 w_k1_ADM_s102
                                 + 1.21217 w_k1_AXP_s102
                                 + 1.36894 w_k1_DIS_s102
                                 + 1.30164 w_k1_FDX_s102
                                 + 0.936198 w_k1_IBM_s102
                                 + 1.09071 w_k1_INTC_s102 + 1.1144 w_k1_KO_s102
                                 + 0.981934 w_k1_MAT_s102
                                 + 1.16273 w_k1_MSFT_s102
                                 + 1.11975 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.6941 w_k1_ADM_s102
                                 + 0.720049 w_k1_AXP_s102
                                 + 0.618216 w_k1_DIS_s102
                                 + 0.740452 w_k1_FDX_s102
                                 + 0.853277 w_k1_IBM_s102
                                 + 0.70895 w_k1_INTC_s102
                                 + 0.674795 w_k1_KO_s102
                                 + 0.799028 w_k1_MAT_s102
                                 + 0.719974 w_k1_MSFT_s102
                                 + 0.75219 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.14651 w_k1_ADM_s102
                                 + 1.32975 w_k1_AXP_s102
                                 + 1.22145 w_k1_DIS_s102
                                 + 1.06617 w_k1_FDX_s102
                                 + 1.40775 w_k1_IBM_s102
                                 + 1.10702 w_k1_INTC_s102
                                 + 1.15988 w_k1_KO_s102 + 1.4963 w_k1_MAT_s102
                                 + 0.896676 w_k1_MSFT_s102
                                 + 1.09475 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.90736 w_k1_ADM_s102
                                 + 1.00008 w_k1_AXP_s102
                                 + 0.794597 w_k1_DIS_s102
                                 + 0.81925 w_k1_FDX_s102
                                 + 1.19922 w_k1_IBM_s102
                                 + 0.935094 w_k1_INTC_s102
                                 + 0.960086 w_k1_KO_s102
                                 + 1.08241 w_k1_MAT_s102
                                 + 0.916233 w_k1_MSFT_s102
                                 + 0.926217 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.805693 w_k1_ADM_s102
                                 + 0.711095 w_k1_AXP_s102
                                 + 0.968373 w_k1_DIS_s102
                                 + 0.860001 w_k1_FDX_s102
                                 + 0.783605 w_k1_IBM_s102
                                 + 0.788829 w_k1_INTC_s102
                                 + 0.783861 w_k1_KO_s102
                                 + 0.571702 w_k1_MAT_s102
                                 + 0.874331 w_k1_MSFT_s102
                                 + 0.733657 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.888407 w_k1_ADM_s102
                                 + 0.932086 w_k1_AXP_s102
                                 + 0.963983 w_k1_DIS_s102
                                 + 0.958801 w_k1_FDX_s102
                                 + 0.790112 w_k1_IBM_s102
                                 + 1.09224 w_k1_INTC_s102
                                 + 0.835754 w_k1_KO_s102
                                 + 0.755365 w_k1_MAT_s102
                                 + 0.932109 w_k1_MSFT_s102
                                 + 0.897582 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.853968 w_k1_ADM_s103
                                 + 0.80702 w_k1_AXP_s103
                                 + 0.926997 w_k1_DIS_s103
                                 + 1.02644 w_k1_FDX_s103
                                 + 0.767839 w_k1_IBM_s103
                                 + 0.749859 w_k1_INTC_s103
                                 + 0.827463 w_k1_KO_s103
                                 + 0.716936 w_k1_MAT_s103
                                 + 0.886005 w_k1_MSFT_s103
                                 + 0.907496 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.901246 w_k1_ADM_s103
                                 + 1.10058 w_k1_AXP_s103
                                 + 1.03321 w_k1_DIS_s103
                                 + 1.11067 w_k1_FDX_s103
                                 + 1.03378 w_k1_IBM_s103
                                 + 1.02627 w_k1_INTC_s103
                                 + 1.17673 w_k1_KO_s103 + 1.3965 w_k1_MAT_s103
                                 + 1.19438 w_k1_MSFT_s103
                                 + 1.04694 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.10564 w_k1_ADM_s103
                                 + 1.27192 w_k1_AXP_s103
                                 + 1.29358 w_k1_DIS_s103
                                 + 1.13292 w_k1_FDX_s103
                                 + 1.47851 w_k1_IBM_s103
                                 + 1.13084 w_k1_INTC_s103
                                 + 1.28606 w_k1_KO_s103 + 1.14666 w_k1_MAT_s103
                                 + 1.06974 w_k1_MSFT_s103
                                 + 1.13698 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.672897 w_k1_ADM_s103
                                 + 0.650757 w_k1_AXP_s103
                                 + 0.614972 w_k1_DIS_s103
                                 + 0.720273 w_k1_FDX_s103
                                 + 0.844317 w_k1_IBM_s103
                                 + 0.83713 w_k1_INTC_s103
                                 + 0.855832 w_k1_KO_s103
                                 + 0.740505 w_k1_MAT_s103
                                 + 0.810851 w_k1_MSFT_s103
                                 + 0.778853 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.839044 w_k1_ADM_s103
                                 + 1.05036 w_k1_AXP_s103
                                 + 1.32357 w_k1_DIS_s103
                                 + 1.18928 w_k1_FDX_s103
                                 + 0.863501 w_k1_IBM_s103
                                 + 1.14958 w_k1_INTC_s103
                                 + 1.04031 w_k1_KO_s103 + 1.10217 w_k1_MAT_s103
                                 + 1.07744 w_k1_MSFT_s103
                                 + 0.95077 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.17037 w_k1_ADM_s103
                                 + 1.06537 w_k1_AXP_s103
                                 + 1.38589 w_k1_DIS_s103
                                 + 1.26899 w_k1_FDX_s103 + 1.0911 w_k1_IBM_s103
                                 + 0.921969 w_k1_INTC_s103
                                 + 1.16795 w_k1_KO_s103 + 1.13428 w_k1_MAT_s103
                                 + 1.20176 w_k1_MSFT_s103 + 1.1906 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.8811 w_k1_ADM_s103
                                 + 0.956306 w_k1_AXP_s103
                                 + 0.719523 w_k1_DIS_s103
                                 + 0.900377 w_k1_FDX_s103
                                 + 1.02308 w_k1_IBM_s103
                                 + 0.797032 w_k1_INTC_s103
                                 + 1.05461 w_k1_KO_s103
                                 + 0.888426 w_k1_MAT_s103
                                 + 0.780096 w_k1_MSFT_s103
                                 + 0.974151 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.917396 w_k1_ADM_s103
                                 + 0.837311 w_k1_AXP_s103
                                 + 0.865891 w_k1_DIS_s103
                                 + 0.956394 w_k1_FDX_s103
                                 + 0.899377 w_k1_IBM_s103
                                 + 0.987985 w_k1_INTC_s103
                                 + 0.783505 w_k1_KO_s103
                                 + 0.680027 w_k1_MAT_s103
                                 + 1.02006 w_k1_MSFT_s103
                                 + 0.806031 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.897595 w_k1_ADM_s104
                                 + 0.811248 w_k1_AXP_s104
                                 + 0.892334 w_k1_DIS_s104
                                 + 0.900011 w_k1_FDX_s104
                                 + 0.751709 w_k1_IBM_s104
                                 + 1.13432 w_k1_INTC_s104
                                 + 0.932821 w_k1_KO_s104
                                 + 0.685837 w_k1_MAT_s104
                                 + 0.927576 w_k1_MSFT_s104
                                 + 0.879221 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.92788 w_k1_ADM_s104
                                 + 0.732896 w_k1_AXP_s104
                                 + 0.601806 w_k1_DIS_s104
                                 + 0.756323 w_k1_FDX_s104
                                 + 0.827627 w_k1_IBM_s104
                                 + 0.812979 w_k1_INTC_s104
                                 + 0.863758 w_k1_KO_s104
                                 + 1.21198 w_k1_MAT_s104
                                 + 0.868418 w_k1_MSFT_s104
                                 + 0.837411 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.2478 w_k1_ADM_s104
                                 + 0.860043 w_k1_AXP_s104
                                 + 1.32554 w_k1_DIS_s104
                                 + 1.10637 w_k1_FDX_s104
                                 + 0.937399 w_k1_IBM_s104
                                 + 1.06473 w_k1_INTC_s104
                                 + 1.26832 w_k1_KO_s104 + 1.01158 w_k1_MAT_s104
                                 + 1.22193 w_k1_MSFT_s104 + 1.0884 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.850114 w_k1_ADM_s104
                                 + 1.24086 w_k1_AXP_s104
                                 + 1.20247 w_k1_DIS_s104
                                 + 1.12392 w_k1_FDX_s104
                                 + 1.28835 w_k1_IBM_s104
                                 + 1.38759 w_k1_INTC_s104
                                 + 1.40904 w_k1_KO_s104 + 2.04029 w_k1_MAT_s104
                                 + 1.31785 w_k1_MSFT_s104
                                 + 1.18861 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.02788 w_k1_ADM_s104
                                 + 1.26354 w_k1_AXP_s104
                                 + 1.32375 w_k1_DIS_s104
                                 + 1.35529 w_k1_FDX_s104
                                 + 0.939915 w_k1_IBM_s104
                                 + 1.37031 w_k1_INTC_s104
                                 + 1.09316 w_k1_KO_s104 + 1.07 w_k1_MAT_s104
                                 + 1.31528 w_k1_MSFT_s104 + 1.0222 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.935828 w_k1_ADM_s104
                                 + 1.12463 w_k1_AXP_s104
                                 + 1.50354 w_k1_DIS_s104
                                 + 1.04984 w_k1_FDX_s104
                                 + 1.61587 w_k1_IBM_s104
                                 + 1.22863 w_k1_INTC_s104
                                 + 1.62108 w_k1_KO_s104 + 2.22524 w_k1_MAT_s104
                                 + 1.47978 w_k1_MSFT_s104 + 1.209 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.962597 w_k1_ADM_s104
                                 + 0.876575 w_k1_AXP_s104
                                 + 1.00511 w_k1_DIS_s104
                                 + 0.977926 w_k1_FDX_s104
                                 + 1.10987 w_k1_IBM_s104
                                 + 1.07111 w_k1_INTC_s104
                                 + 1.20029 w_k1_KO_s104 + 1.53269 w_k1_MAT_s104
                                 + 1.07459 w_k1_MSFT_s104 + 1.1765 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.973142 w_k1_ADM_s104
                                 + 0.841874 w_k1_AXP_s104
                                 + 0.810443 w_k1_DIS_s104
                                 + 0.801475 w_k1_FDX_s104
                                 + 1.16839 w_k1_IBM_s104
                                 + 1.10767 w_k1_INTC_s104
                                 + 1.09043 w_k1_KO_s104 + 1.13266 w_k1_MAT_s104
                                 + 1.05278 w_k1_MSFT_s104
                                 + 1.00819 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.39121 w_k1_ADM_s105
                                 + 1.08676 w_k1_AXP_s105
                                 + 1.28244 w_k1_DIS_s105
                                 + 1.24326 w_k1_FDX_s105
                                 + 0.996027 w_k1_IBM_s105
                                 + 1.25574 w_k1_INTC_s105
                                 + 0.999882 w_k1_KO_s105
                                 + 1.11024 w_k1_MAT_s105
                                 + 1.24623 w_k1_MSFT_s105
                                 + 1.06761 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.767646 w_k1_ADM_s105
                                 + 0.656444 w_k1_AXP_s105
                                 + 0.681115 w_k1_DIS_s105
                                 + 0.786375 w_k1_FDX_s105
                                 + 1.19383 w_k1_IBM_s105
                                 + 0.987069 w_k1_INTC_s105
                                 + 0.888027 w_k1_KO_s105
                                 + 1.15963 w_k1_MAT_s105
                                 + 0.814972 w_k1_MSFT_s105
                                 + 0.91957 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.974142 w_k1_ADM_s105
                                 + 0.878991 w_k1_AXP_s105
                                 + 0.852006 w_k1_DIS_s105
                                 + 0.933632 w_k1_FDX_s105
                                 + 0.759379 w_k1_IBM_s105
                                 + 1.06042 w_k1_INTC_s105
                                 + 0.797984 w_k1_KO_s105
                                 + 0.700854 w_k1_MAT_s105
                                 + 1.00067 w_k1_MSFT_s105
                                 + 0.963948 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.12077 w_k1_ADM_s105
                                 + 1.54177 w_k1_AXP_s105
                                 + 1.66707 w_k1_DIS_s105 + 1.5186 w_k1_FDX_s105
                                 + 1.27405 w_k1_IBM_s105
                                 + 1.48093 w_k1_INTC_s105
                                 + 1.32667 w_k1_KO_s105 + 1.65591 w_k1_MAT_s105
                                 + 1.42605 w_k1_MSFT_s105
                                 + 1.38136 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.11014 w_k1_ADM_s105
                                 + 0.781228 w_k1_AXP_s105
                                 + 0.705927 w_k1_DIS_s105
                                 + 0.909821 w_k1_FDX_s105
                                 + 0.91662 w_k1_IBM_s105
                                 + 0.852882 w_k1_INTC_s105
                                 + 0.863252 w_k1_KO_s105
                                 + 1.08874 w_k1_MAT_s105
                                 + 0.827007 w_k1_MSFT_s105
                                 + 0.902706 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.35388 w_k1_ADM_s105
                                 + 1.18903 w_k1_AXP_s105
                                 + 1.17998 w_k1_DIS_s105
                                 + 1.14278 w_k1_FDX_s105
                                 + 1.55388 w_k1_IBM_s105
                                 + 0.99079 w_k1_INTC_s105 + 1.1552 w_k1_KO_s105
                                 + 1.62097 w_k1_MAT_s105
                                 + 1.01486 w_k1_MSFT_s105
                                 + 1.04754 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.25359 w_k1_ADM_s105
                                 + 1.08813 w_k1_AXP_s105
                                 + 1.10323 w_k1_DIS_s105
                                 + 1.22546 w_k1_FDX_s105
                                 + 0.794069 w_k1_IBM_s105
                                 + 1.30934 w_k1_INTC_s105
                                 + 1.05483 w_k1_KO_s105
                                 + 0.831405 w_k1_MAT_s105
                                 + 1.15584 w_k1_MSFT_s105
                                 + 1.13122 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.17777 w_k1_ADM_s105
                                 + 1.0695 w_k1_AXP_s105 + 1.041 w_k1_DIS_s105
                                 + 1.05897 w_k1_FDX_s105
                                 + 1.21764 w_k1_IBM_s105
                                 + 0.928955 w_k1_INTC_s105
                                 + 1.01696 w_k1_KO_s105 + 1.22754 w_k1_MAT_s105
                                 + 0.920638 w_k1_MSFT_s105
                                 + 0.989593 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.993355 w_k1_ADM_s106
                                 + 1.29933 w_k1_AXP_s106
                                 + 1.18279 w_k1_DIS_s106
                                 + 1.13092 w_k1_FDX_s106
                                 + 1.06738 w_k1_IBM_s106
                                 + 1.18592 w_k1_INTC_s106
                                 + 1.00463 w_k1_KO_s106 + 1.29392 w_k1_MAT_s106
                                 + 1.16185 w_k1_MSFT_s106
                                 + 0.968677 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.780823 w_k1_ADM_s106
                                 + 0.982444 w_k1_AXP_s106
                                 + 1.02371 w_k1_DIS_s106
                                 + 1.04313 w_k1_FDX_s106 + 0.6681 w_k1_IBM_s106
                                 + 1.2411 w_k1_INTC_s106
                                 + 0.713672 w_k1_KO_s106
                                 + 0.854592 w_k1_MAT_s106
                                 + 0.991379 w_k1_MSFT_s106
                                 + 0.852622 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.789932 w_k1_ADM_s106
                                 + 0.874185 w_k1_AXP_s106
                                 + 0.950613 w_k1_DIS_s106
                                 + 0.978733 w_k1_FDX_s106
                                 + 0.753265 w_k1_IBM_s106
                                 + 1.1714 w_k1_INTC_s106
                                 + 0.810264 w_k1_KO_s106
                                 + 0.805088 w_k1_MAT_s106
                                 + 1.05641 w_k1_MSFT_s106
                                 + 0.853019 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.760731 w_k1_ADM_s106
                                 + 0.688108 w_k1_AXP_s106
                                 + 0.837709 w_k1_DIS_s106
                                 + 0.695911 w_k1_FDX_s106
                                 + 1.23256 w_k1_IBM_s106
                                 + 0.65531 w_k1_INTC_s106
                                 + 0.972737 w_k1_KO_s106
                                 + 1.03726 w_k1_MAT_s106
                                 + 0.679909 w_k1_MSFT_s106
                                 + 0.779732 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.04127 w_k1_ADM_s106
                                 + 1.17377 w_k1_AXP_s106 + 1.539 w_k1_DIS_s106
                                 + 1.30307 w_k1_FDX_s106
                                 + 0.904165 w_k1_IBM_s106
                                 + 1.17977 w_k1_INTC_s106
                                 + 1.06126 w_k1_KO_s106 + 1.03447 w_k1_MAT_s106
                                 + 1.37391 w_k1_MSFT_s106 + 1.1099 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.780451 w_k1_ADM_s106
                                 + 0.846138 w_k1_AXP_s106
                                 + 0.648179 w_k1_DIS_s106
                                 + 0.78635 w_k1_FDX_s106
                                 + 0.658707 w_k1_IBM_s106
                                 + 1.15129 w_k1_INTC_s106
                                 + 0.582706 w_k1_KO_s106
                                 + 0.618232 w_k1_MAT_s106
                                 + 0.80959 w_k1_MSFT_s106
                                 + 0.680982 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.08429 w_k1_ADM_s106
                                 + 1.08625 w_k1_AXP_s106
                                 + 1.30153 w_k1_DIS_s106
                                 + 1.12546 w_k1_FDX_s106
                                 + 1.36547 w_k1_IBM_s106
                                 + 1.27892 w_k1_INTC_s106
                                 + 1.34591 w_k1_KO_s106 + 1.58213 w_k1_MAT_s106
                                 + 1.22488 w_k1_MSFT_s106
                                 + 1.08676 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.859742 w_k1_ADM_s106
                                 + 0.759922 w_k1_AXP_s106
                                 + 0.845954 w_k1_DIS_s106
                                 + 0.877566 w_k1_FDX_s106
                                 + 0.971493 w_k1_IBM_s106
                                 + 1.07737 w_k1_INTC_s106
                                 + 1.04754 w_k1_KO_s106
                                 + 0.914045 w_k1_MAT_s106
                                 + 1.05476 w_k1_MSFT_s106
                                 + 0.824548 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.839669 w_k1_ADM_s107
                                 + 0.852431 w_k1_AXP_s107
                                 + 0.950257 w_k1_DIS_s107
                                 + 0.874084 w_k1_FDX_s107
                                 + 1.4218 w_k1_IBM_s107
                                 + 0.812194 w_k1_INTC_s107
                                 + 0.874803 w_k1_KO_s107
                                 + 1.25048 w_k1_MAT_s107
                                 + 0.788915 w_k1_MSFT_s107
                                 + 0.907781 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.34938 w_k1_ADM_s107
                                 + 1.74127 w_k1_AXP_s107
                                 + 1.72619 w_k1_DIS_s107
                                 + 1.46751 w_k1_FDX_s107
                                 + 1.49993 w_k1_IBM_s107
                                 + 1.12693 w_k1_INTC_s107 + 1.23 w_k1_KO_s107
                                 + 2.27926 w_k1_MAT_s107
                                 + 1.21747 w_k1_MSFT_s107
                                 + 1.23644 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.792559 w_k1_ADM_s107
                                 + 0.939663 w_k1_AXP_s107
                                 + 0.865939 w_k1_DIS_s107
                                 + 1.00779 w_k1_FDX_s107
                                 + 0.643622 w_k1_IBM_s107
                                 + 0.956961 w_k1_INTC_s107
                                 + 0.701692 w_k1_KO_s107
                                 + 0.942274 w_k1_MAT_s107
                                 + 1.04214 w_k1_MSFT_s107
                                 + 0.865598 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.814152 w_k1_ADM_s107
                                 + 1.06813 w_k1_AXP_s107
                                 + 1.19987 w_k1_DIS_s107 + 1.1638 w_k1_FDX_s107
                                 + 1.37823 w_k1_IBM_s107
                                 + 0.88689 w_k1_INTC_s107
                                 + 1.11418 w_k1_KO_s107 + 1.64481 w_k1_MAT_s107
                                 + 1.11613 w_k1_MSFT_s107
                                 + 1.02853 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.18276 w_k1_ADM_s107
                                 + 1.45929 w_k1_AXP_s107
                                 + 1.34657 w_k1_DIS_s107 + 1.4022 w_k1_FDX_s107
                                 + 0.931703 w_k1_IBM_s107
                                 + 1.33098 w_k1_INTC_s107 + 1.1678 w_k1_KO_s107
                                 + 1.64689 w_k1_MAT_s107
                                 + 1.47128 w_k1_MSFT_s107
                                 + 1.10063 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.695719 w_k1_ADM_s107
                                 + 0.923009 w_k1_AXP_s107
                                 + 0.784247 w_k1_DIS_s107
                                 + 0.906704 w_k1_FDX_s107
                                 + 0.95252 w_k1_IBM_s107
                                 + 0.879589 w_k1_INTC_s107
                                 + 0.901014 w_k1_KO_s107
                                 + 1.27452 w_k1_MAT_s107
                                 + 0.875949 w_k1_MSFT_s107
                                 + 0.836128 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.50762 w_k1_ADM_s107
                                 + 1.29642 w_k1_AXP_s107
                                 + 1.44439 w_k1_DIS_s107
                                 + 1.48664 w_k1_FDX_s107
                                 + 0.694932 w_k1_IBM_s107
                                 + 0.985362 w_k1_INTC_s107
                                 + 0.868109 w_k1_KO_s107
                                 + 1.12254 w_k1_MAT_s107
                                 + 1.17853 w_k1_MSFT_s107 + 1.1513 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.1073 w_k1_ADM_s107
                                 + 0.95828 w_k1_AXP_s107
                                 + 1.23872 w_k1_DIS_s107
                                 + 1.11912 w_k1_FDX_s107
                                 + 1.08574 w_k1_IBM_s107
                                 + 0.966231 w_k1_INTC_s107
                                 + 1.19961 w_k1_KO_s107 + 1.22318 w_k1_MAT_s107
                                 + 1.06964 w_k1_MSFT_s107
                                 + 1.13823 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.887097 w_k1_ADM_s110
                                 + 0.977897 w_k1_AXP_s110
                                 + 0.839901 w_k1_DIS_s110
                                 + 0.76989 w_k1_FDX_s110
                                 + 1.14145 w_k1_IBM_s110
                                 + 0.819361 w_k1_INTC_s110
                                 + 0.963066 w_k1_KO_s110
                                 + 1.08575 w_k1_MAT_s110
                                 + 0.916953 w_k1_MSFT_s110
                                 + 0.823635 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.1238 w_k1_ADM_s110
                                 + 0.977423 w_k1_AXP_s110
                                 + 1.09141 w_k1_DIS_s110
                                 + 0.940895 w_k1_FDX_s110
                                 + 0.970972 w_k1_IBM_s110
                                 + 0.892258 w_k1_INTC_s110
                                 + 1.09776 w_k1_KO_s110 + 0.92316 w_k1_MAT_s110
                                 + 1.07899 w_k1_MSFT_s110 + 1.0641 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.1985 w_k1_ADM_s110
                                 + 1.26665 w_k1_AXP_s110
                                 + 1.14126 w_k1_DIS_s110
                                 + 1.06068 w_k1_FDX_s110
                                 + 1.05726 w_k1_IBM_s110
                                 + 1.18423 w_k1_INTC_s110
                                 + 1.01263 w_k1_KO_s110
                                 + 0.918677 w_k1_MAT_s110
                                 + 1.09243 w_k1_MSFT_s110 + 1.0011 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.29754 w_k1_ADM_s110
                                 + 1.60598 w_k1_AXP_s110
                                 + 1.63171 w_k1_DIS_s110 + 1.4531 w_k1_FDX_s110
                                 + 1.18681 w_k1_IBM_s110
                                 + 1.17633 w_k1_INTC_s110 + 1.1498 w_k1_KO_s110
                                 + 1.19239 w_k1_MAT_s110
                                 + 1.37221 w_k1_MSFT_s110
                                 + 1.17585 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.30604 w_k1_ADM_s110
                                 + 0.880555 w_k1_AXP_s110
                                 + 1.02566 w_k1_DIS_s110 + 1.1189 w_k1_FDX_s110
                                 + 0.724581 w_k1_IBM_s110
                                 + 0.887912 w_k1_INTC_s110
                                 + 0.813537 w_k1_KO_s110
                                 + 0.600733 w_k1_MAT_s110
                                 + 0.965267 w_k1_MSFT_s110
                                 + 0.973478 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.885075 w_k1_ADM_s110
                                 + 0.678155 w_k1_AXP_s110
                                 + 0.857267 w_k1_DIS_s110
                                 + 0.839596 w_k1_FDX_s110
                                 + 0.713198 w_k1_IBM_s110
                                 + 0.855321 w_k1_INTC_s110
                                 + 0.924694 w_k1_KO_s110
                                 + 0.737986 w_k1_MAT_s110
                                 + 0.89809 w_k1_MSFT_s110
                                 + 0.788926 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.17381 w_k1_ADM_s110
                                 + 1.21293 w_k1_AXP_s110
                                 + 1.14832 w_k1_DIS_s110
                                 + 1.03254 w_k1_FDX_s110
                                 + 1.54356 w_k1_IBM_s110
                                 + 0.987862 w_k1_INTC_s110
                                 + 1.18919 w_k1_KO_s110 + 1.27182 w_k1_MAT_s110
                                 + 0.985986 w_k1_MSFT_s110
                                 + 1.08215 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.934493 w_k1_ADM_s110
                                 + 0.679953 w_k1_AXP_s110
                                 + 0.675687 w_k1_DIS_s110
                                 + 0.72849 w_k1_FDX_s110
                                 + 0.805831 w_k1_IBM_s110
                                 + 0.654541 w_k1_INTC_s110
                                 + 0.761404 w_k1_KO_s110
                                 + 0.872251 w_k1_MAT_s110
                                 + 0.807055 w_k1_MSFT_s110
                                 + 0.835616 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.903322 w_k1_ADM_s111
                                 + 1.70348 w_k1_AXP_s111
                                 + 1.53503 w_k1_DIS_s111 + 1.2463 w_k1_FDX_s111
                                 + 1.61835 w_k1_IBM_s111
                                 + 1.41143 w_k1_INTC_s111
                                 + 1.29037 w_k1_KO_s111 + 1.31989 w_k1_MAT_s111
                                 + 1.24198 w_k1_MSFT_s111
                                 + 1.15398 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.22374 w_k1_ADM_s111
                                 + 1.38785 w_k1_AXP_s111
                                 + 1.61151 w_k1_DIS_s111
                                 + 1.42589 w_k1_FDX_s111
                                 + 0.915686 w_k1_IBM_s111
                                 + 1.3227 w_k1_INTC_s111 + 1.18984 w_k1_KO_s111
                                 + 1.04276 w_k1_MAT_s111
                                 + 1.34275 w_k1_MSFT_s111
                                 + 1.20077 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.881487 w_k1_ADM_s111
                                 + 1.0686 w_k1_AXP_s111
                                 + 0.836149 w_k1_DIS_s111
                                 + 0.961241 w_k1_FDX_s111
                                 + 0.909961 w_k1_IBM_s111
                                 + 1.00552 w_k1_INTC_s111
                                 + 0.869237 w_k1_KO_s111
                                 + 1.02561 w_k1_MAT_s111
                                 + 0.911925 w_k1_MSFT_s111
                                 + 0.969822 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.865844 w_k1_ADM_s111
                                 + 1.44958 w_k1_AXP_s111
                                 + 1.10505 w_k1_DIS_s111
                                 + 1.21674 w_k1_FDX_s111
                                 + 0.976206 w_k1_IBM_s111
                                 + 1.15835 w_k1_INTC_s111
                                 + 0.889584 w_k1_KO_s111
                                 + 1.02351 w_k1_MAT_s111
                                 + 1.07954 w_k1_MSFT_s111
                                 + 0.932123 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.712363 w_k1_ADM_s111
                                 + 0.748561 w_k1_AXP_s111
                                 + 0.621313 w_k1_DIS_s111
                                 + 0.811895 w_k1_FDX_s111
                                 + 0.718114 w_k1_IBM_s111
                                 + 0.888325 w_k1_INTC_s111
                                 + 0.690819 w_k1_KO_s111
                                 + 0.662156 w_k1_MAT_s111
                                 + 0.811196 w_k1_MSFT_s111
                                 + 0.82025 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.944034 w_k1_ADM_s111
                                 + 1.47814 w_k1_AXP_s111
                                 + 1.07265 w_k1_DIS_s111 + 1.133 w_k1_FDX_s111
                                 + 1.24477 w_k1_IBM_s111
                                 + 1.14464 w_k1_INTC_s111 + 1.095 w_k1_KO_s111
                                 + 1.44475 w_k1_MAT_s111
                                 + 1.00586 w_k1_MSFT_s111
                                 + 1.14557 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.895126 w_k1_ADM_s111
                                 + 0.875113 w_k1_AXP_s111
                                 + 0.93409 w_k1_DIS_s111
                                 + 0.914388 w_k1_FDX_s111
                                 + 1.0153 w_k1_IBM_s111
                                 + 1.17325 w_k1_INTC_s111
                                 + 1.03434 w_k1_KO_s111
                                 + 0.704605 w_k1_MAT_s111
                                 + 0.896886 w_k1_MSFT_s111
                                 + 1.00369 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.905171 w_k1_ADM_s111
                                 + 1.22538 w_k1_AXP_s111
                                 + 0.812395 w_k1_DIS_s111
                                 + 0.853812 w_k1_FDX_s111
                                 + 1.33814 w_k1_IBM_s111
                                 + 1.0734 w_k1_INTC_s111 + 1.0428 w_k1_KO_s111
                                 + 1.18751 w_k1_MAT_s111
                                 + 0.936512 w_k1_MSFT_s111
                                 + 0.881536 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.46121 w_k1_ADM_s112
                                 + 1.09067 w_k1_AXP_s112
                                 + 1.10808 w_k1_DIS_s112
                                 + 1.25108 w_k1_FDX_s112
                                 + 0.823758 w_k1_IBM_s112
                                 + 1.13809 w_k1_INTC_s112
                                 + 0.987219 w_k1_KO_s112
                                 + 0.661696 w_k1_MAT_s112
                                 + 1.11658 w_k1_MSFT_s112
                                 + 1.08154 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.33805 w_k1_ADM_s112
                                 + 1.65985 w_k1_AXP_s112
                                 + 1.31232 w_k1_DIS_s112
                                 + 1.39507 w_k1_FDX_s112
                                 + 0.983942 w_k1_IBM_s112
                                 + 1.42925 w_k1_INTC_s112
                                 + 0.970509 w_k1_KO_s112
                                 + 1.16645 w_k1_MAT_s112
                                 + 1.23632 w_k1_MSFT_s112
                                 + 1.17516 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.56742 w_k1_ADM_s112
                                 + 1.77007 w_k1_AXP_s112
                                 + 1.65494 w_k1_DIS_s112
                                 + 1.45621 w_k1_FDX_s112
                                 + 1.47226 w_k1_IBM_s112
                                 + 1.4728 w_k1_INTC_s112 + 1.41499 w_k1_KO_s112
                                 + 1.4235 w_k1_MAT_s112
                                 + 1.39313 w_k1_MSFT_s112
                                 + 1.31066 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.01738 w_k1_ADM_s112
                                 + 0.730436 w_k1_AXP_s112
                                 + 0.937393 w_k1_DIS_s112
                                 + 0.971501 w_k1_FDX_s112
                                 + 0.944512 w_k1_IBM_s112
                                 + 0.916984 w_k1_INTC_s112
                                 + 0.82604 w_k1_KO_s112
                                 + 0.645399 w_k1_MAT_s112
                                 + 0.982024 w_k1_MSFT_s112
                                 + 0.908313 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.1142 w_k1_ADM_s112
                                 + 0.964787 w_k1_AXP_s112
                                 + 1.12881 w_k1_DIS_s112
                                 + 1.11542 w_k1_FDX_s112
                                 + 1.12662 w_k1_IBM_s112
                                 + 1.24456 w_k1_INTC_s112
                                 + 0.990333 w_k1_KO_s112
                                 + 1.12791 w_k1_MAT_s112
                                 + 1.22417 w_k1_MSFT_s112
                                 + 1.15229 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.90925 w_k1_ADM_s112
                                 + 0.50177 w_k1_AXP_s112
                                 + 0.56604 w_k1_DIS_s112
                                 + 0.754026 w_k1_FDX_s112
                                 + 0.553279 w_k1_IBM_s112
                                 + 0.841583 w_k1_INTC_s112
                                 + 0.604557 w_k1_KO_s112
                                 + 0.462196 w_k1_MAT_s112
                                 + 0.813263 w_k1_MSFT_s112
                                 + 0.79205 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.961892 w_k1_ADM_s112
                                 + 0.755743 w_k1_AXP_s112
                                 + 0.705064 w_k1_DIS_s112
                                 + 0.848828 w_k1_FDX_s112
                                 + 0.784687 w_k1_IBM_s112
                                 + 0.847554 w_k1_INTC_s112
                                 + 0.66695 w_k1_KO_s112
                                 + 0.765111 w_k1_MAT_s112
                                 + 0.913935 w_k1_MSFT_s112
                                 + 0.824793 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.07329 w_k1_ADM_s112
                                 + 0.969003 w_k1_AXP_s112
                                 + 0.852408 w_k1_DIS_s112
                                 + 0.942455 w_k1_FDX_s112
                                 + 1.38632 w_k1_IBM_s112
                                 + 1.08263 w_k1_INTC_s112 + 1.0238 w_k1_KO_s112
                                 + 1.35696 w_k1_MAT_s112
                                 + 0.944767 w_k1_MSFT_s112
                                 + 1.0382 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.750474 w_k1_ADM_s113
                                 + 0.803902 w_k1_AXP_s113
                                 + 1.00379 w_k1_DIS_s113
                                 + 0.836738 w_k1_FDX_s113 + 1.016 w_k1_IBM_s113
                                 + 1.01071 w_k1_INTC_s113
                                 + 0.999897 w_k1_KO_s113
                                 + 0.588326 w_k1_MAT_s113
                                 + 0.98592 w_k1_MSFT_s113
                                 + 0.94741 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.933315 w_k1_ADM_s113
                                 + 1.14813 w_k1_AXP_s113
                                 + 1.30901 w_k1_DIS_s113
                                 + 1.21255 w_k1_FDX_s113
                                 + 1.23342 w_k1_IBM_s113
                                 + 0.992983 w_k1_INTC_s113
                                 + 1.22021 w_k1_KO_s113 + 0.91473 w_k1_MAT_s113
                                 + 1.23504 w_k1_MSFT_s113
                                 + 1.10607 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.700103 w_k1_ADM_s113
                                 + 0.852871 w_k1_AXP_s113
                                 + 0.909564 w_k1_DIS_s113
                                 + 0.931725 w_k1_FDX_s113
                                 + 1.36809 w_k1_IBM_s113
                                 + 1.0202 w_k1_INTC_s113 + 1.11977 w_k1_KO_s113
                                 + 1.12271 w_k1_MAT_s113
                                 + 1.08956 w_k1_MSFT_s113
                                 + 1.00972 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.695669 w_k1_ADM_s113
                                 + 0.796677 w_k1_AXP_s113
                                 + 0.787946 w_k1_DIS_s113
                                 + 0.930781 w_k1_FDX_s113
                                 + 0.833095 w_k1_IBM_s113
                                 + 0.784677 w_k1_INTC_s113
                                 + 0.871623 w_k1_KO_s113
                                 + 0.612258 w_k1_MAT_s113
                                 + 0.837484 w_k1_MSFT_s113
                                 + 0.861651 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.918706 w_k1_ADM_s113
                                 + 0.682644 w_k1_AXP_s113
                                 + 1.16593 w_k1_DIS_s113
                                 + 1.03905 w_k1_FDX_s113
                                 + 0.883835 w_k1_IBM_s113
                                 + 1.00582 w_k1_INTC_s113
                                 + 1.17024 w_k1_KO_s113
                                 + 0.480637 w_k1_MAT_s113
                                 + 1.16807 w_k1_MSFT_s113
                                 + 1.04586 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.00574 w_k1_ADM_s113
                                 + 0.929503 w_k1_AXP_s113
                                 + 0.934762 w_k1_DIS_s113
                                 + 1.06158 w_k1_FDX_s113
                                 + 0.672974 w_k1_IBM_s113
                                 + 0.793228 w_k1_INTC_s113
                                 + 0.924569 w_k1_KO_s113
                                 + 0.68484 w_k1_MAT_s113
                                 + 0.933014 w_k1_MSFT_s113
                                 + 1.03036 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.740504 w_k1_ADM_s113
                                 + 0.616746 w_k1_AXP_s113
                                 + 0.817462 w_k1_DIS_s113
                                 + 0.804182 w_k1_FDX_s113
                                 + 0.666344 w_k1_IBM_s113
                                 + 0.909006 w_k1_INTC_s113
                                 + 1.00176 w_k1_KO_s113
                                 + 0.412951 w_k1_MAT_s113
                                 + 1.02954 w_k1_MSFT_s113
                                 + 0.88811 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.42167 w_k1_ADM_s113
                                 + 1.24026 w_k1_AXP_s113 + 1.4149 w_k1_DIS_s113
                                 + 1.45549 w_k1_FDX_s113
                                 + 0.894784 w_k1_IBM_s113
                                 + 1.09297 w_k1_INTC_s113 + 1.1138 w_k1_KO_s113
                                 + 0.700489 w_k1_MAT_s113
                                 + 1.40192 w_k1_MSFT_s113
                                 + 1.12664 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.844284 w_k1_ADM_s114
                                 + 0.931645 w_k1_AXP_s114
                                 + 1.06369 w_k1_DIS_s114
                                 + 0.907034 w_k1_FDX_s114
                                 + 1.26231 w_k1_IBM_s114
                                 + 0.929402 w_k1_INTC_s114
                                 + 1.09982 w_k1_KO_s114 + 1.53491 w_k1_MAT_s114
                                 + 1.00902 w_k1_MSFT_s114
                                 + 1.01824 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.41923 w_k1_ADM_s114
                                 + 1.6348 w_k1_AXP_s114 + 1.40466 w_k1_DIS_s114
                                 + 1.23577 w_k1_FDX_s114
                                 + 1.15739 w_k1_IBM_s114
                                 + 1.12474 w_k1_INTC_s114
                                 + 1.18683 w_k1_KO_s114 + 1.79862 w_k1_MAT_s114
                                 + 1.14999 w_k1_MSFT_s114
                                 + 1.11851 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.837575 w_k1_ADM_s114
                                 + 0.756962 w_k1_AXP_s114
                                 + 0.884077 w_k1_DIS_s114
                                 + 0.823715 w_k1_FDX_s114
                                 + 1.06799 w_k1_IBM_s114
                                 + 0.937302 w_k1_INTC_s114
                                 + 1.13176 w_k1_KO_s114 + 1.0365 w_k1_MAT_s114
                                 + 0.888318 w_k1_MSFT_s114
                                 + 1.00743 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.821468 w_k1_ADM_s114
                                 + 0.770398 w_k1_AXP_s114
                                 + 0.786394 w_k1_DIS_s114
                                 + 0.884956 w_k1_FDX_s114
                                 + 0.619589 w_k1_IBM_s114
                                 + 0.960372 w_k1_INTC_s114
                                 + 0.785469 w_k1_KO_s114
                                 + 0.961125 w_k1_MAT_s114
                                 + 1.02165 w_k1_MSFT_s114
                                 + 0.88637 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.06258 w_k1_ADM_s114
                                 + 0.996418 w_k1_AXP_s114
                                 + 0.978909 w_k1_DIS_s114
                                 + 1.04352 w_k1_FDX_s114
                                 + 0.836189 w_k1_IBM_s114
                                 + 1.20647 w_k1_INTC_s114 + 1.0674 w_k1_KO_s114
                                 + 1.08314 w_k1_MAT_s114
                                 + 1.13803 w_k1_MSFT_s114
                                 + 1.03517 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.28851 w_k1_ADM_s114
                                 + 0.961153 w_k1_AXP_s114
                                 + 1.52119 w_k1_DIS_s114
                                 + 1.12011 w_k1_FDX_s114
                                 + 1.13587 w_k1_IBM_s114
                                 + 1.10599 w_k1_INTC_s114
                                 + 1.30113 w_k1_KO_s114
                                 + 0.959158 w_k1_MAT_s114
                                 + 1.24155 w_k1_MSFT_s114
                                 + 1.12873 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.873138 w_k1_ADM_s114
                                 + 0.579037 w_k1_AXP_s114
                                 + 0.750469 w_k1_DIS_s114
                                 + 0.713153 w_k1_FDX_s114
                                 + 0.746005 w_k1_IBM_s114
                                 + 0.75269 w_k1_INTC_s114
                                 + 0.799759 w_k1_KO_s114
                                 + 0.602134 w_k1_MAT_s114
                                 + 0.885993 w_k1_MSFT_s114
                                 + 0.862833 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.37067 w_k1_ADM_s114
                                 + 1.11971 w_k1_AXP_s114
                                 + 1.17744 w_k1_DIS_s114 + 1.1377 w_k1_FDX_s114
                                 + 1.55093 w_k1_IBM_s114
                                 + 0.878777 w_k1_INTC_s114
                                 + 1.23332 w_k1_KO_s114 + 2.0064 w_k1_MAT_s114
                                 + 1.02866 w_k1_MSFT_s114
                                 + 1.17796 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.818778 w_k1_ADM_s115
                                 + 0.966784 w_k1_AXP_s115
                                 + 0.902407 w_k1_DIS_s115
                                 + 0.938834 w_k1_FDX_s115
                                 + 0.757696 w_k1_IBM_s115
                                 + 1.14796 w_k1_INTC_s115
                                 + 0.867067 w_k1_KO_s115
                                 + 0.62652 w_k1_MAT_s115
                                 + 0.882498 w_k1_MSFT_s115
                                 + 0.943446 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.989657 w_k1_ADM_s115
                                 + 1.62422 w_k1_AXP_s115
                                 + 1.41327 w_k1_DIS_s115
                                 + 1.18321 w_k1_FDX_s115
                                 + 1.45754 w_k1_IBM_s115
                                 + 1.28358 w_k1_INTC_s115
                                 + 1.24282 w_k1_KO_s115 + 1.52144 w_k1_MAT_s115
                                 + 1.05615 w_k1_MSFT_s115
                                 + 1.12858 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.06994 w_k1_ADM_s115
                                 + 1.14079 w_k1_AXP_s115 + 1.5969 w_k1_DIS_s115
                                 + 1.21251 w_k1_FDX_s115
                                 + 1.23539 w_k1_IBM_s115
                                 + 1.1432 w_k1_INTC_s115 + 1.42444 w_k1_KO_s115
                                 + 0.945561 w_k1_MAT_s115
                                 + 1.07834 w_k1_MSFT_s115
                                 + 1.16823 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.00935 w_k1_ADM_s115
                                 + 1.34387 w_k1_AXP_s115
                                 + 1.07416 w_k1_DIS_s115
                                 + 0.966184 w_k1_FDX_s115
                                 + 1.32591 w_k1_IBM_s115
                                 + 1.07802 w_k1_INTC_s115
                                 + 1.16956 w_k1_KO_s115 + 1.06639 w_k1_MAT_s115
                                 + 0.829907 w_k1_MSFT_s115
                                 + 1.00201 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.779016 w_k1_ADM_s115
                                 + 0.891216 w_k1_AXP_s115
                                 + 0.834775 w_k1_DIS_s115
                                 + 0.824852 w_k1_FDX_s115
                                 + 0.987797 w_k1_IBM_s115
                                 + 0.972518 w_k1_INTC_s115
                                 + 0.900133 w_k1_KO_s115
                                 + 0.773469 w_k1_MAT_s115
                                 + 0.797267 w_k1_MSFT_s115
                                 + 0.874983 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.06611 w_k1_ADM_s115
                                 + 1.14186 w_k1_AXP_s115
                                 + 1.28539 w_k1_DIS_s115
                                 + 1.21152 w_k1_FDX_s115
                                 + 0.803094 w_k1_IBM_s115
                                 + 1.1252 w_k1_INTC_s115
                                 + 0.891599 w_k1_KO_s115
                                 + 0.658688 w_k1_MAT_s115
                                 + 1.20105 w_k1_MSFT_s115
                                 + 1.02852 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.939499 w_k1_ADM_s115
                                 + 0.764235 w_k1_AXP_s115
                                 + 0.65125 w_k1_DIS_s115
                                 + 0.764778 w_k1_FDX_s115
                                 + 0.541891 w_k1_IBM_s115
                                 + 0.937398 w_k1_INTC_s115
                                 + 0.688153 w_k1_KO_s115
                                 + 0.435302 w_k1_MAT_s115
                                 + 0.78027 w_k1_MSFT_s115
                                 + 0.778494 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.02648 w_k1_ADM_s115
                                 + 1.0272 w_k1_AXP_s115
                                 + 0.883914 w_k1_DIS_s115
                                 + 0.903475 w_k1_FDX_s115
                                 + 1.22621 w_k1_IBM_s115
                                 + 0.921359 w_k1_INTC_s115
                                 + 1.05115 w_k1_KO_s115
                                 + 0.849484 w_k1_MAT_s115
                                 + 0.758381 w_k1_MSFT_s115
                                 + 0.999589 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.652102 w_k1_ADM_s116
                                 + 0.667398 w_k1_AXP_s116
                                 + 0.560492 w_k1_DIS_s116
                                 + 0.681457 w_k1_FDX_s116
                                 + 0.673141 w_k1_IBM_s116
                                 + 0.754349 w_k1_INTC_s116
                                 + 0.653986 w_k1_KO_s116
                                 + 0.688256 w_k1_MAT_s116
                                 + 0.753475 w_k1_MSFT_s116
                                 + 0.794412 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.30755 w_k1_ADM_s116
                                 + 0.993988 w_k1_AXP_s116
                                 + 1.2403 w_k1_DIS_s116 + 1.22315 w_k1_FDX_s116
                                 + 0.903028 w_k1_IBM_s116
                                 + 1.3158 w_k1_INTC_s116 + 1.1163 w_k1_KO_s116
                                 + 0.864432 w_k1_MAT_s116
                                 + 1.12853 w_k1_MSFT_s116
                                 + 1.06397 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.88292 w_k1_ADM_s116
                                 + 0.895584 w_k1_AXP_s116
                                 + 0.925853 w_k1_DIS_s116
                                 + 0.889877 w_k1_FDX_s116
                                 + 1.14407 w_k1_IBM_s116
                                 + 1.01144 w_k1_INTC_s116
                                 + 1.00214 w_k1_KO_s116 + 1.13161 w_k1_MAT_s116
                                 + 0.924174 w_k1_MSFT_s116
                                 + 0.855338 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.939209 w_k1_ADM_s116
                                 + 0.644646 w_k1_AXP_s116
                                 + 0.570426 w_k1_DIS_s116
                                 + 0.678216 w_k1_FDX_s116
                                 + 0.936056 w_k1_IBM_s116
                                 + 0.782095 w_k1_INTC_s116
                                 + 0.742292 w_k1_KO_s116
                                 + 0.805638 w_k1_MAT_s116
                                 + 0.730177 w_k1_MSFT_s116
                                 + 0.725302 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.934123 w_k1_ADM_s116
                                 + 0.996308 w_k1_AXP_s116
                                 + 0.793104 w_k1_DIS_s116
                                 + 0.969299 w_k1_FDX_s116
                                 + 0.752545 w_k1_IBM_s116
                                 + 0.754757 w_k1_INTC_s116
                                 + 0.757064 w_k1_KO_s116
                                 + 1.05761 w_k1_MAT_s116
                                 + 0.882173 w_k1_MSFT_s116
                                 + 0.800974 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.849297 w_k1_ADM_s116
                                 + 1.10771 w_k1_AXP_s116 + 1.2751 w_k1_DIS_s116
                                 + 0.980612 w_k1_FDX_s116
                                 + 1.56145 w_k1_IBM_s116
                                 + 0.977471 w_k1_INTC_s116
                                 + 1.28166 w_k1_KO_s116 + 1.58967 w_k1_MAT_s116
                                 + 0.969134 w_k1_MSFT_s116
                                 + 0.992655 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.820397 w_k1_ADM_s116
                                 + 0.712482 w_k1_AXP_s116
                                 + 0.85967 w_k1_DIS_s116
                                 + 0.790856 w_k1_FDX_s116
                                 + 0.84337 w_k1_IBM_s116
                                 + 0.847332 w_k1_INTC_s116
                                 + 0.85364 w_k1_KO_s116
                                 + 0.788589 w_k1_MAT_s116
                                 + 0.892512 w_k1_MSFT_s116
                                 + 0.813965 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.941966 w_k1_ADM_s116
                                 + 0.81485 w_k1_AXP_s116
                                 + 0.843596 w_k1_DIS_s116
                                 + 1.01451 w_k1_FDX_s116
                                 + 0.617062 w_k1_IBM_s116
                                 + 1.02052 w_k1_INTC_s116
                                 + 0.771812 w_k1_KO_s116
                                 + 0.534398 w_k1_MAT_s116
                                 + 0.961583 w_k1_MSFT_s116
                                 + 0.846127 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.900167 w_k1_ADM_s117
                                 + 0.881832 w_k1_AXP_s117
                                 + 0.926571 w_k1_DIS_s117
                                 + 0.92146 w_k1_FDX_s117
                                 + 1.04882 w_k1_IBM_s117
                                 + 1.07337 w_k1_INTC_s117
                                 + 0.904381 w_k1_KO_s117
                                 + 0.927261 w_k1_MAT_s117
                                 + 0.939314 w_k1_MSFT_s117
                                 + 0.958611 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.22651 w_k1_ADM_s117
                                 + 0.751516 w_k1_AXP_s117
                                 + 1.01495 w_k1_DIS_s117
                                 + 1.08088 w_k1_FDX_s117
                                 + 0.673002 w_k1_IBM_s117
                                 + 1.12971 w_k1_INTC_s117
                                 + 0.881964 w_k1_KO_s117
                                 + 0.350484 w_k1_MAT_s117
                                 + 1.2313 w_k1_MSFT_s117 + 1.07447 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.43338 w_k1_ADM_s117
                                 + 1.0624 w_k1_AXP_s117 + 1.4842 w_k1_DIS_s117
                                 + 1.33173 w_k1_FDX_s117
                                 + 0.993093 w_k1_IBM_s117
                                 + 1.48968 w_k1_INTC_s117
                                 + 1.32967 w_k1_KO_s117
                                 + 0.923569 w_k1_MAT_s117
                                 + 1.43422 w_k1_MSFT_s117
                                 + 1.05781 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.815453 w_k1_ADM_s117
                                 + 0.595939 w_k1_AXP_s117
                                 + 0.658519 w_k1_DIS_s117
                                 + 0.797893 w_k1_FDX_s117
                                 + 0.696443 w_k1_IBM_s117
                                 + 0.88793 w_k1_INTC_s117
                                 + 0.798879 w_k1_KO_s117
                                 + 0.572623 w_k1_MAT_s117
                                 + 0.899432 w_k1_MSFT_s117
                                 + 0.821001 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.28237 w_k1_ADM_s117
                                 + 1.50402 w_k1_AXP_s117
                                 + 1.62647 w_k1_DIS_s117
                                 + 1.25812 w_k1_FDX_s117
                                 + 1.56334 w_k1_IBM_s117
                                 + 1.37155 w_k1_INTC_s117
                                 + 1.52838 w_k1_KO_s117 + 1.45279 w_k1_MAT_s117
                                 + 1.41689 w_k1_MSFT_s117
                                 + 1.22311 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.941281 w_k1_ADM_s117
                                 + 0.817623 w_k1_AXP_s117
                                 + 0.899741 w_k1_DIS_s117
                                 + 0.999541 w_k1_FDX_s117
                                 + 0.789973 w_k1_IBM_s117
                                 + 1.07461 w_k1_INTC_s117
                                 + 0.839418 w_k1_KO_s117
                                 + 0.63002 w_k1_MAT_s117
                                 + 1.14639 w_k1_MSFT_s117
                                 + 0.87279 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.850158 w_k1_ADM_s117
                                 + 0.942401 w_k1_AXP_s117
                                 + 1.14393 w_k1_DIS_s117
                                 + 1.09886 w_k1_FDX_s117
                                 + 0.902093 w_k1_IBM_s117
                                 + 1.12762 w_k1_INTC_s117
                                 + 0.928491 w_k1_KO_s117
                                 + 0.694326 w_k1_MAT_s117
                                 + 1.14286 w_k1_MSFT_s117 + 1.0643 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.926291 w_k1_ADM_s117
                                 + 0.916935 w_k1_AXP_s117
                                 + 0.961793 w_k1_DIS_s117
                                 + 0.870854 w_k1_FDX_s117
                                 + 1.34232 w_k1_IBM_s117
                                 + 0.825433 w_k1_INTC_s117
                                 + 1.19778 w_k1_KO_s117 + 1.34313 w_k1_MAT_s117
                                 + 0.945005 w_k1_MSFT_s117
                                 + 1.05403 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0365020727013448 w_k1_ADM_s000
                                 + 0.0272404580204683 w_k1_ADM_s001
                                 + 0.0412552582459337 w_k1_ADM_s002
                                 + 0.0271300080342018 w_k1_ADM_s003
                                 + 0.021106294862896 w_k1_ADM_s004
                                 + 0.0641545118432483 w_k1_ADM_s005
                                 + 0.0298032942395382 w_k1_ADM_s006
                                 + 0.0863532701984582 w_k1_ADM_s007
                                 + 0.0314371230935175 w_k1_ADM_s010
                                 + 0.0175658283839968 w_k1_ADM_s011
                                 + 0.0593849479022833 w_k1_ADM_s012
                                 + 0.0278651700581513 w_k1_ADM_s013
                                 + 0.0471147841085645 w_k1_ADM_s014
                                 + 0.0727108996631167 w_k1_ADM_s015
                                 + 0.0158273048823495 w_k1_ADM_s016
                                 + 0.0575264930110416 w_k1_ADM_s017
                                 + 0.0417466624612281 w_k1_ADM_s100
                                 + 0.0228596676064618 w_k1_ADM_s101
                                 + 0.00944525286628293 w_k1_ADM_s102
                                 + 0.0105484379681158 w_k1_ADM_s103
                                 + 0.0212220998597971 w_k1_ADM_s104
                                 + 0.0210092949495285 w_k1_ADM_s105
                                 + 0.0158611176636651 w_k1_ADM_s106
                                 + 0.0171700927551587 w_k1_ADM_s107
                                 + 0.0151502681127137 w_k1_ADM_s110
                                 + 0.0194013453160895 w_k1_ADM_s111
                                 + 0.0345325258138331 w_k1_ADM_s112
                                 + 0.0104197662114354 w_k1_ADM_s113
                                 + 0.0112660306631846 w_k1_ADM_s114
                                 + 0.0387417732190608 w_k1_ADM_s115
                                 + 0.0202573268797208 w_k1_ADM_s116
                                 + 0.0157187660832943 w_k1_ADM_s117
                                 + 0.0388929943497783 w_k1_AXP_s000
                                 + 0.0276557660358049 w_k1_AXP_s001
                                 + 0.0413404583357346 w_k1_AXP_s002
                                 + 0.0264358305059087 w_k1_AXP_s003
                                 + 0.0243759989329297 w_k1_AXP_s004
                                 + 0.0629726040757976 w_k1_AXP_s005
                                 + 0.0304532752726264 w_k1_AXP_s006
                                 + 0.0907026114772147 w_k1_AXP_s007
                                 + 0.0317350639856963 w_k1_AXP_s010
                                 + 0.0194725178637519 w_k1_AXP_s011
                                 + 0.0545248714723218 w_k1_AXP_s012
                                 + 0.0329858147670546 w_k1_AXP_s013
                                 + 0.0428415226605434 w_k1_AXP_s014
                                 + 0.0848374526068122 w_k1_AXP_s015
                                 + 0.0172599825464705 w_k1_AXP_s016
                                 + 0.0607693503720199 w_k1_AXP_s017
                                 + 0.0457022679388627 w_k1_AXP_s100
                                 + 0.0229688221344911 w_k1_AXP_s101
                                 + 0.0101910175224938 w_k1_AXP_s102
                                 + 0.0111174560302301 w_k1_AXP_s103
                                 + 0.0220124585661435 w_k1_AXP_s104
                                 + 0.0178901703426544 w_k1_AXP_s105
                                 + 0.0168897562403752 w_k1_AXP_s106
                                 + 0.0198962044963354 w_k1_AXP_s107
                                 + 0.0147331098407353 w_k1_AXP_s110
                                 + 0.0256757945434884 w_k1_AXP_s111
                                 + 0.0306606844163147 w_k1_AXP_s112
                                 + 0.0108546496247574 w_k1_AXP_s113
                                 + 0.0100316881824297 w_k1_AXP_s114
                                 + 0.0463168283902497 w_k1_AXP_s115
                                 + 0.0193613913115549 w_k1_AXP_s116
                                 + 0.0144727723240042 w_k1_AXP_s117
                                 + 0.0394083829914093 w_k1_DIS_s000
                                 + 0.0278575777083056 w_k1_DIS_s001
                                 + 0.0415000913960036 w_k1_DIS_s002
                                 + 0.0254087798055372 w_k1_DIS_s003
                                 + 0.0244837325273493 w_k1_DIS_s004
                                 + 0.0632365962527447 w_k1_DIS_s005
                                 + 0.0302387875780635 w_k1_DIS_s006
                                 + 0.0871324511141998 w_k1_DIS_s007
                                 + 0.033455088360687 w_k1_DIS_s010
                                 + 0.0196972545687923 w_k1_DIS_s011
                                 + 0.0577044854724282 w_k1_DIS_s012
                                 + 0.0306578581912852 w_k1_DIS_s013
                                 + 0.0450466689438356 w_k1_DIS_s014
                                 + 0.0848060103956381 w_k1_DIS_s015
                                 + 0.0183586360736147 w_k1_DIS_s016
                                 + 0.0610064677857621 w_k1_DIS_s017
                                 + 0.0459235169434156 w_k1_DIS_s100
                                 + 0.0228838462308437 w_k1_DIS_s101
                                 + 0.00991524815655593 w_k1_DIS_s102
                                 + 0.0114764520430028 w_k1_DIS_s103
                                 + 0.0240883485111801 w_k1_DIS_s104
                                 + 0.0182471494185866 w_k1_DIS_s105
                                 + 0.0188256191883798 w_k1_DIS_s106
                                 + 0.0198072034807168 w_k1_DIS_s107
                                 + 0.0146051173162829 w_k1_DIS_s110
                                 + 0.0214058673205437 w_k1_DIS_s111
                                 + 0.030092400598592 w_k1_DIS_s112
                                 + 0.0125914633865776 w_k1_DIS_s113
                                 + 0.0111425911594893 w_k1_DIS_s114
                                 + 0.0452559222751427 w_k1_DIS_s115
                                 + 0.0209964760190191 w_k1_DIS_s116
                                 + 0.016708826046523 w_k1_DIS_s117
                                 + 0.0377000388362865 w_k1_FDX_s000
                                 + 0.0280081678719118 w_k1_FDX_s001
                                 + 0.0400799814141614 w_k1_FDX_s002
                                 + 0.026085137542452 w_k1_FDX_s003
                                 + 0.0234025004613376 w_k1_FDX_s004
                                 + 0.0659162988526622 w_k1_FDX_s005
                                 + 0.0305888714643603 w_k1_FDX_s006
                                 + 0.0876955450332751 w_k1_FDX_s007
                                 + 0.0322726257057049 w_k1_FDX_s010
                                 + 0.0198832580021402 w_k1_FDX_s011
                                 + 0.056512901823156 w_k1_FDX_s012
                                 + 0.0302974497615264 w_k1_FDX_s013
                                 + 0.0463285159395914 w_k1_FDX_s014
                                 + 0.0787031297038437 w_k1_FDX_s015
                                 + 0.0168605521470692 w_k1_FDX_s016
                                 + 0.0604279681141033 w_k1_FDX_s017
                                 + 0.0458897148831386 w_k1_FDX_s100
                                 + 0.0220799800253201 w_k1_FDX_s101
                                 + 0.00948783324973279 w_k1_FDX_s102
                                 + 0.0117216207572136 w_k1_FDX_s103
                                 + 0.0228837840632313 w_k1_FDX_s104
                                 + 0.0193208605329533 w_k1_FDX_s105
                                 + 0.0175988170878276 w_k1_FDX_s106
                                 + 0.0202771010351487 w_k1_FDX_s107
                                 + 0.0136651360635377 w_k1_FDX_s110
                                 + 0.0223717183193303 w_k1_FDX_s111
                                 + 0.0317941381229457 w_k1_FDX_s112
                                 + 0.0123988670991966 w_k1_FDX_s113
                                 + 0.0104584288593529 w_k1_FDX_s114
                                 + 0.0413505772302358 w_k1_FDX_s115
                                 + 0.020245227106312 w_k1_FDX_s116
                                 + 0.0157408966933413 w_k1_FDX_s117
                                 + 0.0381814661049485 w_k1_IBM_s000
                                 + 0.0301076949945102 w_k1_IBM_s001
                                 + 0.0419678625592965 w_k1_IBM_s002
                                 + 0.0270883166132917 w_k1_IBM_s003
                                 + 0.0242383775069805 w_k1_IBM_s004
                                 + 0.0673529623374635 w_k1_IBM_s005
                                 + 0.0315676637364904 w_k1_IBM_s006
                                 + 0.0899377912405289 w_k1_IBM_s007
                                 + 0.0313190690983716 w_k1_IBM_s010
                                 + 0.0201628652197232 w_k1_IBM_s011
                                 + 0.056113228161953 w_k1_IBM_s012
                                 + 0.0314735266050627 w_k1_IBM_s013
                                 + 0.046243703507643 w_k1_IBM_s014
                                 + 0.0804607474451743 w_k1_IBM_s015
                                 + 0.0167419042607117 w_k1_IBM_s016
                                 + 0.0618816791273629 w_k1_IBM_s017
                                 + 0.0471804381985619 w_k1_IBM_s100
                                 + 0.0216948947299242 w_k1_IBM_s101
                                 + 0.00962827325743791 w_k1_IBM_s102
                                 + 0.0117890636195565 w_k1_IBM_s103
                                 + 0.023148364491183 w_k1_IBM_s104
                                 + 0.0186040351639217 w_k1_IBM_s105
                                 + 0.0174569490599938 w_k1_IBM_s106
                                 + 0.01880988953987 w_k1_IBM_s107
                                 + 0.0149024399178363 w_k1_IBM_s110
                                 + 0.0230704139210748 w_k1_IBM_s111
                                 + 0.0318646031830944 w_k1_IBM_s112
                                 + 0.011975181314982 w_k1_IBM_s113
                                 + 0.0109403669530375 w_k1_IBM_s114
                                 + 0.043242218778096 w_k1_IBM_s115
                                 + 0.0222131448586209 w_k1_IBM_s116
                                 + 0.015564611903526 w_k1_IBM_s117
                                 + 0.0384400736728496 w_k1_INTC_s000
                                 + 0.0273382911883348 w_k1_INTC_s001
                                 + 0.0388343645696833 w_k1_INTC_s002
                                 + 0.025262228034084 w_k1_INTC_s003
                                 + 0.0213109214856278 w_k1_INTC_s004
                                 + 0.0560319125726505 w_k1_INTC_s005
                                 + 0.0242260217985746 w_k1_INTC_s006
                                 + 0.0962290510671174 w_k1_INTC_s007
                                 + 0.0295612141424474 w_k1_INTC_s010
                                 + 0.0223159504672698 w_k1_INTC_s011
                                 + 0.0460414853407278 w_k1_INTC_s012
                                 + 0.0333199927352294 w_k1_INTC_s013
                                 + 0.0419842632757623 w_k1_INTC_s014
                                 + 0.0716949421738192 w_k1_INTC_s015
                                 + 0.0151768245653339 w_k1_INTC_s016
                                 + 0.0653929940089441 w_k1_INTC_s017
                                 + 0.0422315163569895 w_k1_INTC_s100
                                 + 0.020039052538784 w_k1_INTC_s101
                                 + 0.0089634452260904 w_k1_INTC_s102
                                 + 0.0107906320653298 w_k1_INTC_s103
                                 + 0.0260000482973961 w_k1_INTC_s104
                                 + 0.0192990986084557 w_k1_INTC_s105
                                 + 0.0196660147561895 w_k1_INTC_s106
                                 + 0.0178660044114982 w_k1_INTC_s107
                                 + 0.012949517109173 w_k1_INTC_s110
                                 + 0.0242887535884833 w_k1_INTC_s111
                                 + 0.0330973545335802 w_k1_INTC_s112
                                 + 0.0112748194752633 w_k1_INTC_s113
                                 + 0.0108756423761253 w_k1_INTC_s114
                                 + 0.0447753534378133 w_k1_INTC_s115
                                 + 0.0213219491700369 w_k1_INTC_s116
                                 + 0.0171371902598402 w_k1_INTC_s117
                                 + 0.0360498228063926 w_k1_KO_s000
                                 + 0.0274999947666105 w_k1_KO_s001
                                 + 0.0461635737599581 w_k1_KO_s002
                                 + 0.0271840272935513 w_k1_KO_s003
                                 + 0.0243872413343481 w_k1_KO_s004
                                 + 0.0634706323470097 w_k1_KO_s005
                                 + 0.0327826606665119 w_k1_KO_s006
                                 + 0.0893191348354141 w_k1_KO_s007
                                 + 0.033404120742942 w_k1_KO_s010
                                 + 0.0204481739791518 w_k1_KO_s011
                                 + 0.0503905613849255 w_k1_KO_s012
                                 + 0.0301902032820501 w_k1_KO_s013
                                 + 0.0468381505037836 w_k1_KO_s014
                                 + 0.0799879548915777 w_k1_KO_s015
                                 + 0.0187636891618838 w_k1_KO_s016
                                 + 0.0575475264263984 w_k1_KO_s017
                                 + 0.045824129926778 w_k1_KO_s100
                                 + 0.0218812842154005 w_k1_KO_s101
                                 + 0.0091197062090898 w_k1_KO_s102
                                 + 0.0118265607255408 w_k1_KO_s103
                                 + 0.0257806677166111 w_k1_KO_s104
                                 + 0.0174600420076154 w_k1_KO_s105
                                 + 0.0176632760854305 w_k1_KO_s106
                                 + 0.0176111498836547 w_k1_KO_s107
                                 + 0.0138974109197695 w_k1_KO_s110
                                 + 0.0212184399783144 w_k1_KO_s111
                                 + 0.0281592064240172 w_k1_KO_s112
                                 + 0.0126433425674391 w_k1_KO_s113
                                 + 0.0116473537500322 w_k1_KO_s114
                                 + 0.0428117968408107 w_k1_KO_s115
                                 + 0.0211346649270017 w_k1_KO_s116
                                 + 0.0162104979376338 w_k1_KO_s117
                                 + 0.0325090943008811 w_k1_MAT_s000
                                 + 0.0329383531129939 w_k1_MAT_s001
                                 + 0.0539823442466486 w_k1_MAT_s002
                                 + 0.0274838656084594 w_k1_MAT_s003
                                 + 0.022999297479338 w_k1_MAT_s004
                                 + 0.0698522779548224 w_k1_MAT_s005
                                 + 0.0381518899394649 w_k1_MAT_s006
                                 + 0.0958278801235509 w_k1_MAT_s007
                                 + 0.0345845292467261 w_k1_MAT_s010
                                 + 0.0201466781369434 w_k1_MAT_s011
                                 + 0.0671072004679809 w_k1_MAT_s012
                                 + 0.0314728373408883 w_k1_MAT_s013
                                 + 0.0482475824271042 w_k1_MAT_s014
                                 + 0.11256738690691 w_k1_MAT_s015
                                 + 0.0212303629724775 w_k1_MAT_s016
                                 + 0.0629649213788711 w_k1_MAT_s017
                                 + 0.0498954662455477 w_k1_MAT_s100
                                 + 0.02063990908338 w_k1_MAT_s101
                                 + 0.00992876167173449 w_k1_MAT_s102
                                 + 0.0111192590478151 w_k1_MAT_s103
                                 + 0.0294971608906097 w_k1_MAT_s104
                                 + 0.0200312808349484 w_k1_MAT_s105
                                 + 0.0190629346297201 w_k1_MAT_s106
                                 + 0.0247673402885643 w_k1_MAT_s107
                                 + 0.0138183812835264 w_k1_MAT_s110
                                 + 0.0221957685870171 w_k1_MAT_s111
                                 + 0.0300889608871785 w_k1_MAT_s112
                                 + 0.00884043672990981 w_k1_MAT_s113
                                 + 0.0126305341540697 w_k1_MAT_s114
                                 + 0.0366099095958764 w_k1_MAT_s115
                                 + 0.0221294793432364 w_k1_MAT_s116
                                 + 0.0136431950790476 w_k1_MAT_s117
                                 + 0.0370267484855892 w_k1_MSFT_s000
                                 + 0.026220695974839 w_k1_MSFT_s001
                                 + 0.041062133969393 w_k1_MSFT_s002
                                 + 0.0256938720456819 w_k1_MSFT_s003
                                 + 0.0223460375771578 w_k1_MSFT_s004
                                 + 0.0675955640725834 w_k1_MSFT_s005
                                 + 0.0299753609571691 w_k1_MSFT_s006
                                 + 0.0927766407999479 w_k1_MSFT_s007
                                 + 0.0322423311765858 w_k1_MSFT_s010
                                 + 0.0202900973215965 w_k1_MSFT_s011
                                 + 0.0521982270908117 w_k1_MSFT_s012
                                 + 0.0334704619385628 w_k1_MSFT_s013
                                 + 0.0454450779939234 w_k1_MSFT_s014
                                 + 0.0715926221017815 w_k1_MSFT_s015
                                 + 0.0163114225740138 w_k1_MSFT_s016
                                 + 0.0588420129347865 w_k1_MSFT_s017
                                 + 0.0419027767204268 w_k1_MSFT_s100
                                 + 0.0209647991901339 w_k1_MSFT_s101
                                 + 0.00859649930730285 w_k1_MSFT_s102
                                 + 0.0113793365341784 w_k1_MSFT_s103
                                 + 0.025530475174964 w_k1_MSFT_s104
                                 + 0.0185581789129011 w_k1_MSFT_s105
                                 + 0.0189443578757067 w_k1_MSFT_s106
                                 + 0.0193258296077166 w_k1_MSFT_s107
                                 + 0.0140158601789944 w_k1_MSFT_s110
                                 + 0.0215131846535452 w_k1_MSFT_s111
                                 + 0.0319913553783497 w_k1_MSFT_s112
                                 + 0.012954068100356 w_k1_MSFT_s113
                                 + 0.0111915999243949 w_k1_MSFT_s114
                                 + 0.0379204603977274 w_k1_MSFT_s115
                                 + 0.0201765463125606 w_k1_MSFT_s116
                                 + 0.0171548185882929 w_k1_MSFT_s117
                                 + 0.0351497153925415 w_k1_PG_s000
                                 + 0.0271699781313059 w_k1_PG_s001
                                 + 0.0448356258500833 w_k1_PG_s002
                                 + 0.0259740076352272 w_k1_PG_s003
                                 + 0.0233629883561984 w_k1_PG_s004
                                 + 0.0676451371991289 w_k1_PG_s005
                                 + 0.0323807528722506 w_k1_PG_s006
                                 + 0.0898891442855672 w_k1_PG_s007
                                 + 0.0309394477191859 w_k1_PG_s010
                                 + 0.0184235559146846 w_k1_PG_s011
                                 + 0.0549330439799603 w_k1_PG_s012
                                 + 0.0300059475297019 w_k1_PG_s013
                                 + 0.0503394798098006 w_k1_PG_s014
                                 + 0.0795342746710404 w_k1_PG_s015
                                 + 0.0162946911728691 w_k1_PG_s016
                                 + 0.0597442766234513 w_k1_PG_s017
                                 + 0.046040055089033 w_k1_PG_s100
                                 + 0.0225142012809723 w_k1_PG_s101
                                 + 0.00910548278591274 w_k1_PG_s102
                                 + 0.0111982231769726 w_k1_PG_s103
                                 + 0.0230485577122019 w_k1_PG_s104
                                 + 0.0183785673359607 w_k1_PG_s105
                                 + 0.01600664848574 w_k1_PG_s106
                                 + 0.0179988939486846 w_k1_PG_s107
                                 + 0.013477625666482 w_k1_PG_s110
                                 + 0.0216594326616377 w_k1_PG_s111
                                 + 0.0308896553736194 w_k1_PG_s112
                                 + 0.0119144605895758 w_k1_PG_s113
                                 + 0.0110507866111656 w_k1_PG_s114
                                 + 0.0409134618371483 w_k1_PG_s115
                                 + 0.0192291606703201 w_k1_PG_s116
                                 + 0.0152953262420049 w_k1_PG_s117 >= 1.05
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
