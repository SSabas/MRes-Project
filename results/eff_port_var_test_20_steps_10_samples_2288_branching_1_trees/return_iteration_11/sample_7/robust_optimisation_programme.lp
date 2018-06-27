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
 w_k1_ADM_s0_return_constraint:  1.19987 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.961538 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.02172 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.981352 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.1481 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.859956 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.904962 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.03132 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.0481 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.20156 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.45439 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.91081 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.19202 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.953093 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.683617 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.855599 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.714217 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.923882 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.977811 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.965649 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.23663 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.30919 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.0336 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.00183 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.01939 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.25768 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.34616 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.780713 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.974867 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.1074 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.815166 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.15447 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.41222 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.956223 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.858475 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.990138 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.31706 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.14059 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.19614 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.972333 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.22816 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.943359 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.39618 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.835334 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.54801 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.981925 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.0888 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.04308 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.6021 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.767949 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.918998 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.732578 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.567915 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.809686 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.643872 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.15554 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.03542 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.890412 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.93447 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.30263 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.13779 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.701998 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.840153 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.28251 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.48883 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.623762 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.879536 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.02541 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.732902 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.14478 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.27758 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.03491 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.754314 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.974506 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.56449 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.30032 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.31628 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.956501 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.1389 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.959654 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.42839 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.813867 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.535333 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.963708 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.0951 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.47799 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.73501 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.804794 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.16892 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.720301 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.532762 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.657813 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.637871 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.852617 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.943675 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.968359 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.8562 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.41975 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.968785 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.837868 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.06299 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.49972 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.46774 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.546252 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.873768 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.31108 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.58665 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.12407 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.42059 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.936161 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.627441 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.96722 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.92594 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.38782 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.24532 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.937186 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.13547 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.88864 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.27322 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.897799 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.725068 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.01811 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.962655 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.1851 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.45157 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.794352 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.04149 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.850879 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.735161 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.880008 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.695157 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.950559 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.994642 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.06884 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.45822 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.30476 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.08545 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.879151 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.15918 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.19758 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.4603 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.736954 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.836165 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.16604 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.708493 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.14632 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.40955 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.916637 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.890266 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.02945 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.41252 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.183 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.998486 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.02385 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.902096 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.12794 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.31186 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.920036 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.526221 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.01808 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.32355 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.20785 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.62946 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.07192 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.781931 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.719298 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.430143 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.628679 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.729803 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.08966 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.06768 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.657701 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.97659 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.14442 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.902849 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.726754 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.616096 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.6198 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.936416 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.641577 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.2734 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.899941 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.811186 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.90411 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.769948 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.31551 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.529503 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.738521 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.62525 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.28993 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11243 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.899835 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.0814 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.923206 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.34124 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.994185 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.79601 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.2239 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.989488 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.35814 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.34693 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.928908 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.05579 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.873252 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.813903 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.915282 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.836778 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.984944 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.07885 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.18984 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.59631 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.20255 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.06822 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.910194 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.09877 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.09955 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.28551 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.741658 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.840957 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.07349 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.742823 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.988811 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.19767 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.983793 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.880927 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.03924 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.35646 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.09457 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.10886 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.975302 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.00862 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.02509 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.29532 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.881946 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.639696 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.958628 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.15218 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.41516 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.4511 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.00002 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.06329 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.828454 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.486503 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.723829 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.723279 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.904686 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.06348 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.910182 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.66731 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.19592 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.908466 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.8469 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.831109 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.40878 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.15912 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.706038 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.01459 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.17248 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.713099 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.08022 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.0163 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.21071 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.662027 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.814346 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.52295 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.28492 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.19503 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.0715 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.916268 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.91097 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.37655 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.839422 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.384338 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.912856 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.5539 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.18027 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.36871 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.12741 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.731027 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.626114 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.477056 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.759289 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.666707 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.43593 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.14087 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.593967 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.54972 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.22466 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.985324 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.518031 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.551774 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.97108 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.999517 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.557325 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.08915 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.846972 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.711689 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.20303 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.709108 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.26492 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.566536 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.652482 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.87361 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.40737 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18238 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.928056 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.04144 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.862334 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.27126 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.938012 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.776592 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.051 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.05173 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.39731 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.41602 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.977437 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.17223 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.823521 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.805679 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.01586 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.812519 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.05425 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.14839 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.14988 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.50803 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.2829 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.03945 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.93535 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.15795 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.08951 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.27701 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.7978 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.820827 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.14715 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.724054 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.1192 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.1501 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.945812 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.856869 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.952621 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.31253 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.11866 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12841 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.954754 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.04048 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.926258 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.22424 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.960145 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.754019 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.00321 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.08399 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.04933 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.22888 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.945831 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.09624 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.904766 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.680725 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.916329 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.809025 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.0081 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.03127 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.01749 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.35135 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.09986 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.04496 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.888753 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.902817 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.24324 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.24935 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.74636 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.02249 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.979103 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.755156 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.13941 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.09155 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.983052 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.824887 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.955098 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.25518 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.07657 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0649999932504499 z_k1_s0000
                                 - 0.12700004952013 z_k1_s0001
                                 - 0.19499997975135 z_k1_s0002
                                 - 0.15800007765497 z_k1_s0003
                                 - 0.0689999771581648 z_k1_s0004
                                 - 0.239000414136413 z_k1_s0005
                                 - 0.113999643269907 z_k1_s0006
                                 - 0.0519999946003599 z_k1_s0007
                                 - 0.23700054096696 z_k1_s0010
                                 - 0.1570004240374 z_k1_s0011
                                 - 0.13200040518447 z_k1_s0012
                                 - 0.178000507794083 z_k1_s0013
                                 - 0.156000015761948 z_k1_s0014
                                 - 0.266001060917989 z_k1_s0015
                                 - 0.112000305201848 z_k1_s0016
                                 - 0.177000099518631 z_k1_s0017
                                 - 0.225000529540076 z_k1_s0020
                                 - 0.113999565180113 z_k1_s0021
                                 - 0.0770000364603419 z_k1_s0022
                                 - 0.119999868821261 z_k1_s0023
                                 - 0.120999747094738 z_k1_s0024
                                 - 0.147999562479479 z_k1_s0025
                                 - 0.120999747094738 z_k1_s0026
                                 - 0.108000295539248 z_k1_s0027
                                 - 0.124999845034716 z_k1_s0030
                                 - 0.137999920678168 z_k1_s0031
                                 - 0.0769998127815427 z_k1_s0032
                                 - 0.0779998803617021 z_k1_s0033
                                 - 0.104999869829161 z_k1_s0034
                                 - 0.0989998084476133 z_k1_s0035
                                 - 0.158999160565263 z_k1_s0036
                                 - 0.365999041582516 z_k1_s0037
                                 - 0.457998104644224 z_k1_s0040
                                 - 0.249998590553421 z_k1_s0041
                                 - 0.135999592244202 z_k1_s0042
                                 - 0.141999683281185 z_k1_s0043
                                 - 0.288998987194279 z_k1_s0044
                                 - 0.238998878884529 z_k1_s0045
                                 - 0.181999574829452 z_k1_s0046
                                 - 0.254998991583462 z_k1_s0047
                                 - 0.16599981099614 z_k1_s0050
                                 - 0.0599998082737177 z_k1_s0051
                                 - 0.163998901710089 z_k1_s0052
                                 - 0.197999227623737 z_k1_s0053
                                 - 0.178999319376956 z_k1_s0054
                                 - 0.0479998233390523 z_k1_s0055
                                 - 0.124999212571547 z_k1_s0056
                                 - 0.223999408381463 z_k1_s0057
                                 - 0.0609999521898311 z_k1_s0060
                                 - 0.168000119883456 z_k1_s0061
                                 - 0.162999696154804 z_k1_s0062
                                 - 0.141000082348616 z_k1_s0063
                                 - 0.185999803906701 z_k1_s0064
                                 - 0.0669999491642406 z_k1_s0065
                                 - 0.106000185247889 z_k1_s0066
                                 - 0.131000257891257 z_k1_s0067
                                 - 0.526998373413793 z_k1_s0070
                                 - 0.373999249518716 z_k1_s0071
                                 - 0.149999838264467 z_k1_s0072
                                 - 0.208999727907244 z_k1_s0073
                                 - 0.420999242244168 z_k1_s0074
                                 - 0.343998881226566 z_k1_s0075
                                 - 0.164999772011007 z_k1_s0076
                                 - 0.313998512934417 z_k1_s0077
                                 - 0.136999573975493 z_k1_s0100
                                 - 0.0440000306125049 z_k1_s0101
                                 - 0.126999797699838 z_k1_s0102
                                 - 0.149999689133694 z_k1_s0103
                                 - 0.0860000044699165 z_k1_s0104
                                 - 0.198999810883951 z_k1_s0105
                                 - 0.153000028016239 z_k1_s0106
                                 - 0.118999570679465 z_k1_s0107
                                 - 0.192999900905599 z_k1_s0110
                                 - 0.249999959525431 z_k1_s0111
                                 - 0.218999575960083 z_k1_s0112
                                 - 0.322000601677084 z_k1_s0113
                                 - 0.0900001087892172 z_k1_s0114
                                 - 0.289999459609252 z_k1_s0115
                                 - 0.108000038027014 z_k1_s0116
                                 - 0.0700000503471517 z_k1_s0117
                                 - 0.287000895597971 z_k1_s0120
                                 - 0.111000239102065 z_k1_s0121
                                 - 0.13800039812716 z_k1_s0122
                                 - 0.101000318426472 z_k1_s0123
                                 - 0.353001491659868 z_k1_s0124
                                 - 0.267000432245126 z_k1_s0125
                                 - 0.118000245775146 z_k1_s0126
                                 - 0.180001060167304 z_k1_s0127
                                 - 0.221999896646831 z_k1_s0130
                                 - 0.0760000362863338 z_k1_s0131
                                 - 0.178000163491711 z_k1_s0132
                                 - 0.207000609116947 z_k1_s0133
                                 - 0.0720000605449592 z_k1_s0134
                                 - 0.0570000272147504 z_k1_s0135
                                 - 0.148000345431401 z_k1_s0136
                                 - 0.283000148203065 z_k1_s0137
                                 - 0.390000256320729 z_k1_s0140
                                 - 0.500998714222429 z_k1_s0141
                                 - 0.359000240185497 z_k1_s0142
                                 - 0.476002300924041 z_k1_s0143
                                 - 0.602001086607679 z_k1_s0144
                                 - 1.11500204689798 z_k1_s0145
                                 - 0.367000497662628 z_k1_s0146
                                 - 0.323000528090102 z_k1_s0147
                                 - 0.0870000123883505 z_k1_s0150
                                 - 0.0720000362938158 z_k1_s0151
                                 - 0.082999999883135 z_k1_s0152
                                 - 0.12900047409322 z_k1_s0153
                                 - 0.0890000658409736 z_k1_s0154
                                 - 0.280999627690974 z_k1_s0155
                                 - 0.0900000453672697 z_k1_s0156
                                 - 0.112999668872112 z_k1_s0157
                                 - 0.257000131817456 z_k1_s0160
                                 - 0.289000148230524 z_k1_s0161
                                 - 0.209000107197853 z_k1_s0162
                                 - 0.0830000425713963 z_k1_s0163
                                 - 0.059000030261595 z_k1_s0164
                                 - 0.0640000328261369 z_k1_s0165
                                 - 0.121000062061915 z_k1_s0166
                                 - 0.168000086168609 z_k1_s0167
                                 - 0.771000603536106 z_k1_s0170
                                 - 0.179000146338479 z_k1_s0171
                                 - 0.207000329922311 z_k1_s0172
                                 - 0.161000256606242 z_k1_s0173
                                 - 0.643000677439492 z_k1_s0174
                                 - 0.336999355988298 z_k1_s0175
                                 - 0.431000200591385 z_k1_s0176
                                 - 0.466999980055861 z_k1_s0177
                                 - 0.294001609985824 z_k1_s1000
                                 - 0.267000342013724 z_k1_s1001
                                 - 0.456000347402796 z_k1_s1002
                                 - 0.385000986312792 z_k1_s1003
                                 - 0.222000446330797 z_k1_s1004
                                 - 0.306001619128003 z_k1_s1005
                                 - 0.325000940601898 z_k1_s1006
                                 - 0.51700108687676 z_k1_s1007
                                 - 0.38599936224349 z_k1_s1010
                                 - 0.639999744713077 z_k1_s1011
                                 - 0.589999637345109 z_k1_s1012
                                 - 0.284999389799732 z_k1_s1013
                                 - 0.544000190405344 z_k1_s1014
                                 - 0.694000512509249 z_k1_s1015
                                 - 0.500001073679682 z_k1_s1016
                                 - 0.435000526702094 z_k1_s1017
                                 - 0.179000298278174 z_k1_s1020
                                 - 0.253999329176983 z_k1_s1021
                                 - 0.0950000524716694 z_k1_s1022
                                 - 0.18099965803555 z_k1_s1023
                                 - 0.176999658520953 z_k1_s1024
                                 - 0.119000049559251 z_k1_s1025
                                 - 0.0839999898065356 z_k1_s1026
                                 - 0.191000296821965 z_k1_s1027
                                 - 0.185999614755662 z_k1_s1030
                                 - 0.405999463100457 z_k1_s1031
                                 - 0.213999522017874 z_k1_s1032
                                 - 0.422999579895023 z_k1_s1033
                                 - 0.436998725727755 z_k1_s1034
                                 - 0.785999935512716 z_k1_s1035
                                 - 0.913999626968312 z_k1_s1036
                                 - 0.672997020882547 z_k1_s1037
                                 - 0.117999920113319 z_k1_s1040
                                 - 0.109999925529365 z_k1_s1041
                                 - 0.274999813823414 z_k1_s1042
                                 - 0.0669999546406135 z_k1_s1043
                                 - 0.319999783358154 z_k1_s1044
                                 - 0.081999944485527 z_k1_s1045
                                 - 0.0769999478705558 z_k1_s1046
                                 - 0.20099986392184 z_k1_s1047
                                 - 0.433999160229155 z_k1_s1050
                                 - 0.121999918935395 z_k1_s1051
                                 - 0.20200029113896 z_k1_s1052
                                 - 0.0930000122491171 z_k1_s1053
                                 - 0.230999429025187 z_k1_s1054
                                 - 0.415000188993381 z_k1_s1055
                                 - 0.220999262374734 z_k1_s1056
                                 - 0.203999171268975 z_k1_s1057
                                 - 0.506000287833285 z_k1_s1060
                                 - 0.980998178691176 z_k1_s1061
                                 - 1.10400129912734 z_k1_s1062
                                 - 0.539999939999061 z_k1_s1063
                                 - 0.314000008402428 z_k1_s1064
                                 - 0.496000209196244 z_k1_s1065
                                 - 0.707000330437397 z_k1_s1066
                                 - 1.50399829260735 z_k1_s1067
                                 - 0.203999519555871 z_k1_s1070
                                 - 0.236000902486929 z_k1_s1071
                                 - 0.203999519555871 z_k1_s1072
                                 - 0.260000509434511 z_k1_s1073
                                 - 0.411999927103479 z_k1_s1074
                                 - 0.260999777931971 z_k1_s1075
                                 - 0.192999938080014 z_k1_s1076
                                 - 0.137000092401943 z_k1_s1077
                                 - 0.951000751816047 z_k1_s1100
                                 - 0.759000033286695 z_k1_s1101
                                 - 0.82799519705141 z_k1_s1102
                                 - 0.74399529856884 z_k1_s1103
                                 - 1.04599968643547 z_k1_s1104
                                 - 1.15099955953868 z_k1_s1105
                                 - 0.543998487041508 z_k1_s1106
                                 - 0.630998667069213 z_k1_s1107
                                 - 0.296000299076629 z_k1_s1110
                                 - 0.708998833726315 z_k1_s1111
                                 - 0.34700024727558 z_k1_s1112
                                 - 0.574998501752315 z_k1_s1113
                                 - 0.231000160313136 z_k1_s1114
                                 - 0.674999862927623 z_k1_s1115
                                 - 0.289000203794358 z_k1_s1116
                                 - 0.608001561941764 z_k1_s1117
                                 - 0.262999949985459 z_k1_s1120
                                 - 0.367000175695626 z_k1_s1121
                                 - 0.187999868187725 z_k1_s1122
                                 - 0.526000523770785 z_k1_s1123
                                 - 0.441999633693494 z_k1_s1124
                                 - 0.262999949985459 z_k1_s1125
                                 - 0.255999979778996 z_k1_s1126
                                 - 0.813999565322356 z_k1_s1127
                                 - 1.77799841525593 z_k1_s1130
                                 - 1.779998711673 z_k1_s1131
                                 - 1.49699291859192 z_k1_s1132
                                 - 1.45599889202026 z_k1_s1133
                                 - 0.879997873751119 z_k1_s1134
                                 - 1.00199788022517 z_k1_s1135
                                 - 1.18999802848001 z_k1_s1136
                                 - 2.46699208113306 z_k1_s1137
                                 - 0.468000752836294 z_k1_s1140
                                 - 0.498000794016818 z_k1_s1141
                                 - 0.497000387770377 z_k1_s1142
                                 - 1.52900390236261 z_k1_s1143
                                 - 1.05200004540892 z_k1_s1144
                                 - 0.603998031791629 z_k1_s1145
                                 - 0.482000367180115 z_k1_s1146
                                 - 0.391000389492679 z_k1_s1147
                                 - 1.54500245233795 z_k1_s1150
                                 - 0.389000078868579 z_k1_s1151
                                 - 1.36000072702153 z_k1_s1152
                                 - 0.793999092336753 z_k1_s1153
                                 - 0.346000133330446 z_k1_s1154
                                 - 1.23800042429445 z_k1_s1155
                                 - 0.400999982018706 z_k1_s1156
                                 - 0.949002815278936 z_k1_s1157
                                 - 0.633998164805546 z_k1_s1160
                                 - 0.425000216572648 z_k1_s1161
                                 - 0.333000104772665 z_k1_s1162
                                 - 0.626001498171866 z_k1_s1163
                                 - 0.932000640951839 z_k1_s1164
                                 - 0.649000960746675 z_k1_s1165
                                 - 0.734001727741443 z_k1_s1166
                                 - 0.189999985075971 z_k1_s1167
                                 - 1.16699650873829 z_k1_s1170
                                 - 0.387000226573655 z_k1_s1171
                                 - 0.614000209370112 z_k1_s1172
                                 - 0.807003417738378 z_k1_s1173
                                 - 0.484999610512313 z_k1_s1174
                                 - 1.73299711597517 z_k1_s1175
                                 - 1.88499664106362 z_k1_s1176
                                 - 0.771999488475252 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.09718 w_k1_ADM_s000
                                 + 1.30582 w_k1_AXP_s000
                                 + 1.02906 w_k1_DIS_s000
                                 + 1.24289 w_k1_FDX_s000
                                 + 0.630509 w_k1_IBM_s000
                                 + 1.08134 w_k1_INTC_s000
                                 + 0.816622 w_k1_KO_s000
                                 + 1.08233 w_k1_MAT_s000
                                 + 1.08193 w_k1_MSFT_s000
                                 + 0.965051 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.34052 w_k1_ADM_s000
                                 + 1.54417 w_k1_AXP_s000
                                 + 1.61915 w_k1_DIS_s000
                                 + 1.40446 w_k1_FDX_s000
                                 + 1.12635 w_k1_IBM_s000
                                 + 1.21338 w_k1_INTC_s000
                                 + 1.16327 w_k1_KO_s000 + 1.41031 w_k1_MAT_s000
                                 + 1.31678 w_k1_MSFT_s000
                                 + 1.25182 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.823828 w_k1_ADM_s000
                                 + 1.00584 w_k1_AXP_s000
                                 + 0.921743 w_k1_DIS_s000
                                 + 1.05298 w_k1_FDX_s000
                                 + 0.855825 w_k1_IBM_s000
                                 + 1.21676 w_k1_INTC_s000
                                 + 0.860238 w_k1_KO_s000
                                 + 0.956974 w_k1_MAT_s000
                                 + 1.05289 w_k1_MSFT_s000
                                 + 0.995326 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.710442 w_k1_ADM_s000
                                 + 1.20902 w_k1_AXP_s000
                                 + 0.815578 w_k1_DIS_s000
                                 + 0.889146 w_k1_FDX_s000
                                 + 1.02846 w_k1_IBM_s000
                                 + 0.884894 w_k1_INTC_s000
                                 + 0.867364 w_k1_KO_s000
                                 + 1.60812 w_k1_MAT_s000
                                 + 0.904162 w_k1_MSFT_s000
                                 + 0.941117 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.0394 w_k1_ADM_s000
                                 + 1.18546 w_k1_AXP_s000
                                 + 1.01166 w_k1_DIS_s000
                                 + 1.10237 w_k1_FDX_s000
                                 + 1.53146 w_k1_IBM_s000
                                 + 1.07553 w_k1_INTC_s000
                                 + 1.07696 w_k1_KO_s000 + 1.59739 w_k1_MAT_s000
                                 + 1.05604 w_k1_MSFT_s000
                                 + 0.958722 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.624789 w_k1_ADM_s000
                                 + 0.745914 w_k1_AXP_s000
                                 + 0.68694 w_k1_DIS_s000
                                 + 0.756895 w_k1_FDX_s000
                                 + 0.647914 w_k1_IBM_s000
                                 + 0.862668 w_k1_INTC_s000
                                 + 0.606423 w_k1_KO_s000
                                 + 0.842205 w_k1_MAT_s000
                                 + 0.78361 w_k1_MSFT_s000
                                 + 0.795156 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.17347 w_k1_ADM_s000
                                 + 1.02581 w_k1_AXP_s000
                                 + 1.54224 w_k1_DIS_s000 + 1.1217 w_k1_FDX_s000
                                 + 1.0848 w_k1_IBM_s000 + 1.1941 w_k1_INTC_s000
                                 + 1.22558 w_k1_KO_s000
                                 + 0.840617 w_k1_MAT_s000
                                 + 1.17791 w_k1_MSFT_s000
                                 + 1.07359 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.664034 w_k1_ADM_s000
                                 + 1.03321 w_k1_AXP_s000
                                 + 0.994304 w_k1_DIS_s000
                                 + 0.87603 w_k1_FDX_s000
                                 + 1.34219 w_k1_IBM_s000
                                 + 0.897597 w_k1_INTC_s000
                                 + 0.988404 w_k1_KO_s000
                                 + 1.73162 w_k1_MAT_s000
                                 + 0.930338 w_k1_MSFT_s000
                                 + 0.960107 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.704328 w_k1_ADM_s001
                                 + 0.73846 w_k1_AXP_s001
                                 + 0.801346 w_k1_DIS_s001
                                 + 0.797749 w_k1_FDX_s001
                                 + 1.20966 w_k1_IBM_s001
                                 + 0.841488 w_k1_INTC_s001
                                 + 1.13128 w_k1_KO_s001 + 1.28858 w_k1_MAT_s001
                                 + 0.911656 w_k1_MSFT_s001
                                 + 0.893284 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.25093 w_k1_ADM_s001
                                 + 1.35806 w_k1_AXP_s001
                                 + 1.32008 w_k1_DIS_s001
                                 + 1.44749 w_k1_FDX_s001
                                 + 1.02842 w_k1_IBM_s001
                                 + 1.09135 w_k1_INTC_s001
                                 + 1.09493 w_k1_KO_s001 + 1.44935 w_k1_MAT_s001
                                 + 1.25852 w_k1_MSFT_s001
                                 + 1.04558 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.974792 w_k1_ADM_s001
                                 + 0.912855 w_k1_AXP_s001
                                 + 1.08165 w_k1_DIS_s001
                                 + 1.11137 w_k1_FDX_s001
                                 + 0.716241 w_k1_IBM_s001
                                 + 1.00917 w_k1_INTC_s001
                                 + 1.01233 w_k1_KO_s001
                                 + 0.890454 w_k1_MAT_s001
                                 + 1.20317 w_k1_MSFT_s001
                                 + 0.916005 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.831335 w_k1_ADM_s001
                                 + 0.918627 w_k1_AXP_s001
                                 + 0.93306 w_k1_DIS_s001
                                 + 0.856028 w_k1_FDX_s001
                                 + 0.81477 w_k1_IBM_s001
                                 + 0.683952 w_k1_INTC_s001
                                 + 0.936629 w_k1_KO_s001
                                 + 0.996695 w_k1_MAT_s001
                                 + 0.873052 w_k1_MSFT_s001
                                 + 0.984991 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.20096 w_k1_ADM_s001
                                 + 1.05355 w_k1_AXP_s001
                                 + 1.36247 w_k1_DIS_s001
                                 + 1.18036 w_k1_FDX_s001 + 1.1788 w_k1_IBM_s001
                                 + 0.928121 w_k1_INTC_s001
                                 + 1.09656 w_k1_KO_s001 + 1.14592 w_k1_MAT_s001
                                 + 1.13422 w_k1_MSFT_s001
                                 + 1.08414 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.16063 w_k1_ADM_s001
                                 + 1.09398 w_k1_AXP_s001
                                 + 0.803328 w_k1_DIS_s001
                                 + 1.00207 w_k1_FDX_s001
                                 + 0.999207 w_k1_IBM_s001
                                 + 0.791752 w_k1_INTC_s001
                                 + 1.10308 w_k1_KO_s001 + 1.17264 w_k1_MAT_s001
                                 + 0.928123 w_k1_MSFT_s001
                                 + 1.05008 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.785982 w_k1_ADM_s001
                                 + 0.66537 w_k1_AXP_s001
                                 + 0.591387 w_k1_DIS_s001
                                 + 0.744912 w_k1_FDX_s001
                                 + 0.688673 w_k1_IBM_s001
                                 + 0.772123 w_k1_INTC_s001
                                 + 0.735439 w_k1_KO_s001
                                 + 0.641787 w_k1_MAT_s001
                                 + 0.808421 w_k1_MSFT_s001
                                 + 0.817369 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.23859 w_k1_ADM_s001
                                 + 1.65913 w_k1_AXP_s001
                                 + 1.06111 w_k1_DIS_s001
                                 + 1.13084 w_k1_FDX_s001
                                 + 1.42775 w_k1_IBM_s001
                                 + 1.08841 w_k1_INTC_s001
                                 + 1.24531 w_k1_KO_s001 + 2.17461 w_k1_MAT_s001
                                 + 1.10265 w_k1_MSFT_s001
                                 + 1.16132 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.04815 w_k1_ADM_s002
                                 + 0.914406 w_k1_AXP_s002
                                 + 0.951861 w_k1_DIS_s002
                                 + 0.946393 w_k1_FDX_s002
                                 + 0.945764 w_k1_IBM_s002
                                 + 0.821211 w_k1_INTC_s002
                                 + 0.94572 w_k1_KO_s002
                                 + 0.808414 w_k1_MAT_s002
                                 + 0.839168 w_k1_MSFT_s002
                                 + 0.895076 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.995177 w_k1_ADM_s002
                                 + 1.19851 w_k1_AXP_s002
                                 + 0.942158 w_k1_DIS_s002
                                 + 1.03874 w_k1_FDX_s002
                                 + 1.51888 w_k1_IBM_s002
                                 + 0.938665 w_k1_INTC_s002
                                 + 1.11522 w_k1_KO_s002 + 1.43521 w_k1_MAT_s002
                                 + 0.871138 w_k1_MSFT_s002
                                 + 1.17637 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.14595 w_k1_ADM_s002
                                 + 1.6935 w_k1_AXP_s002 + 1.86737 w_k1_DIS_s002
                                 + 1.54602 w_k1_FDX_s002
                                 + 1.07896 w_k1_IBM_s002
                                 + 1.42273 w_k1_INTC_s002
                                 + 1.43155 w_k1_KO_s002 + 1.34922 w_k1_MAT_s002
                                 + 1.52478 w_k1_MSFT_s002
                                 + 1.31098 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.810346 w_k1_ADM_s002
                                 + 0.792639 w_k1_AXP_s002
                                 + 0.636648 w_k1_DIS_s002
                                 + 0.784556 w_k1_FDX_s002
                                 + 1.16723 w_k1_IBM_s002
                                 + 0.939775 w_k1_INTC_s002
                                 + 0.990577 w_k1_KO_s002
                                 + 0.622362 w_k1_MAT_s002
                                 + 0.75402 w_k1_MSFT_s002
                                 + 0.887065 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.664049 w_k1_ADM_s002
                                 + 0.7836 w_k1_AXP_s002
                                 + 0.860808 w_k1_DIS_s002
                                 + 0.921013 w_k1_FDX_s002
                                 + 0.667588 w_k1_IBM_s002
                                 + 1.12663 w_k1_INTC_s002
                                 + 0.743824 w_k1_KO_s002
                                 + 0.547532 w_k1_MAT_s002
                                 + 1.0429 w_k1_MSFT_s002
                                 + 0.851762 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.05201 w_k1_ADM_s002
                                 + 0.920459 w_k1_AXP_s002
                                 + 1.12663 w_k1_DIS_s002
                                 + 1.17692 w_k1_FDX_s002
                                 + 0.767088 w_k1_IBM_s002
                                 + 1.26703 w_k1_INTC_s002
                                 + 0.834403 w_k1_KO_s002
                                 + 0.770032 w_k1_MAT_s002
                                 + 1.07065 w_k1_MSFT_s002
                                 + 0.890172 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.932445 w_k1_ADM_s002
                                 + 1.24493 w_k1_AXP_s002
                                 + 1.17797 w_k1_DIS_s002
                                 + 1.18456 w_k1_FDX_s002
                                 + 1.00533 w_k1_IBM_s002
                                 + 1.25628 w_k1_INTC_s002
                                 + 1.09684 w_k1_KO_s002
                                 + 0.986763 w_k1_MAT_s002
                                 + 1.13421 w_k1_MSFT_s002
                                 + 1.01878 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.997973 w_k1_ADM_s002
                                 + 1.14091 w_k1_AXP_s002
                                 + 1.39924 w_k1_DIS_s002
                                 + 1.25918 w_k1_FDX_s002
                                 + 1.38744 w_k1_IBM_s002
                                 + 1.29034 w_k1_INTC_s002
                                 + 1.25408 w_k1_KO_s002 + 1.18958 w_k1_MAT_s002
                                 + 1.11678 w_k1_MSFT_s002
                                 + 1.06883 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.835149 w_k1_ADM_s003
                                 + 0.674415 w_k1_AXP_s003
                                 + 0.715337 w_k1_DIS_s003
                                 + 0.844673 w_k1_FDX_s003
                                 + 0.667577 w_k1_IBM_s003
                                 + 0.955076 w_k1_INTC_s003
                                 + 0.70376 w_k1_KO_s003
                                 + 0.575406 w_k1_MAT_s003
                                 + 0.905538 w_k1_MSFT_s003
                                 + 0.830777 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.08025 w_k1_ADM_s003
                                 + 1.2335 w_k1_AXP_s003 + 1.05867 w_k1_DIS_s003
                                 + 1.18017 w_k1_FDX_s003
                                 + 0.775371 w_k1_IBM_s003
                                 + 1.11882 w_k1_INTC_s003
                                 + 0.909039 w_k1_KO_s003
                                 + 0.740263 w_k1_MAT_s003
                                 + 0.993034 w_k1_MSFT_s003
                                 + 1.11349 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.03568 w_k1_ADM_s003
                                 + 1.1425 w_k1_AXP_s003
                                 + 0.817874 w_k1_DIS_s003
                                 + 1.01678 w_k1_FDX_s003
                                 + 0.900729 w_k1_IBM_s003
                                 + 0.984496 w_k1_INTC_s003
                                 + 0.808838 w_k1_KO_s003
                                 + 1.08176 w_k1_MAT_s003
                                 + 0.918887 w_k1_MSFT_s003
                                 + 1.05897 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.02636 w_k1_ADM_s003
                                 + 0.990266 w_k1_AXP_s003
                                 + 1.02099 w_k1_DIS_s003
                                 + 1.26133 w_k1_FDX_s003
                                 + 0.672662 w_k1_IBM_s003
                                 + 1.01157 w_k1_INTC_s003
                                 + 0.757094 w_k1_KO_s003
                                 + 0.563049 w_k1_MAT_s003
                                 + 1.11061 w_k1_MSFT_s003
                                 + 1.06835 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.797268 w_k1_ADM_s003
                                 + 0.789096 w_k1_AXP_s003
                                 + 0.653893 w_k1_DIS_s003
                                 + 0.733523 w_k1_FDX_s003
                                 + 0.923502 w_k1_IBM_s003
                                 + 0.739813 w_k1_INTC_s003
                                 + 0.833082 w_k1_KO_s003
                                 + 1.03777 w_k1_MAT_s003
                                 + 0.718016 w_k1_MSFT_s003
                                 + 1.06213 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.49162 w_k1_ADM_s003
                                 + 1.67461 w_k1_AXP_s003
                                 + 1.59619 w_k1_DIS_s003
                                 + 1.50964 w_k1_FDX_s003
                                 + 1.03092 w_k1_IBM_s003
                                 + 1.33329 w_k1_INTC_s003
                                 + 1.00763 w_k1_KO_s003 + 1.07914 w_k1_MAT_s003
                                 + 1.32534 w_k1_MSFT_s003 + 1.1853 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.0672 w_k1_ADM_s003
                                 + 0.828302 w_k1_AXP_s003
                                 + 0.873037 w_k1_DIS_s003
                                 + 0.861165 w_k1_FDX_s003
                                 + 1.07576 w_k1_IBM_s003
                                 + 0.825553 w_k1_INTC_s003
                                 + 1.00547 w_k1_KO_s003
                                 + 0.969869 w_k1_MAT_s003
                                 + 0.897358 w_k1_MSFT_s003
                                 + 1.00463 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.981837 w_k1_ADM_s003
                                 + 1.14337 w_k1_AXP_s003
                                 + 1.06025 w_k1_DIS_s003
                                 + 1.01971 w_k1_FDX_s003
                                 + 1.41116 w_k1_IBM_s003
                                 + 0.98488 w_k1_INTC_s003
                                 + 1.10915 w_k1_KO_s003 + 1.15634 w_k1_MAT_s003
                                 + 1.01219 w_k1_MSFT_s003
                                 + 1.11619 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.771312 w_k1_ADM_s004
                                 + 0.98403 w_k1_AXP_s004
                                 + 0.803943 w_k1_DIS_s004
                                 + 0.85581 w_k1_FDX_s004
                                 + 1.06944 w_k1_IBM_s004
                                 + 1.00985 w_k1_INTC_s004
                                 + 0.958919 w_k1_KO_s004
                                 + 1.26084 w_k1_MAT_s004
                                 + 0.913549 w_k1_MSFT_s004
                                 + 1.02384 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.819104 w_k1_ADM_s004
                                 + 0.726756 w_k1_AXP_s004
                                 + 0.839212 w_k1_DIS_s004
                                 + 0.891919 w_k1_FDX_s004
                                 + 0.619837 w_k1_IBM_s004
                                 + 0.816685 w_k1_INTC_s004
                                 + 0.858961 w_k1_KO_s004
                                 + 0.740982 w_k1_MAT_s004
                                 + 0.920445 w_k1_MSFT_s004
                                 + 0.846096 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.831771 w_k1_ADM_s004
                                 + 1.25261 w_k1_AXP_s004
                                 + 1.06299 w_k1_DIS_s004
                                 + 0.979512 w_k1_FDX_s004
                                 + 1.9117 w_k1_IBM_s004
                                 + 1.20426 w_k1_INTC_s004
                                 + 1.58998 w_k1_KO_s004 + 3.27422 w_k1_MAT_s004
                                 + 1.11579 w_k1_MSFT_s004
                                 + 1.12666 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.86529 w_k1_ADM_s004
                                 + 0.845585 w_k1_AXP_s004
                                 + 1.05208 w_k1_DIS_s004
                                 + 0.867578 w_k1_FDX_s004
                                 + 1.11729 w_k1_IBM_s004
                                 + 0.856824 w_k1_INTC_s004
                                 + 1.24728 w_k1_KO_s004 + 1.54614 w_k1_MAT_s004
                                 + 0.923695 w_k1_MSFT_s004
                                 + 1.04078 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.27607 w_k1_ADM_s004
                                 + 1.32377 w_k1_AXP_s004
                                 + 1.70221 w_k1_DIS_s004
                                 + 1.19649 w_k1_FDX_s004
                                 + 1.29667 w_k1_IBM_s004
                                 + 1.36591 w_k1_INTC_s004
                                 + 1.65415 w_k1_KO_s004 + 1.41086 w_k1_MAT_s004
                                 + 1.41177 w_k1_MSFT_s004
                                 + 1.40159 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.11957 w_k1_ADM_s004
                                 + 1.26515 w_k1_AXP_s004
                                 + 0.980432 w_k1_DIS_s004
                                 + 1.15138 w_k1_FDX_s004
                                 + 0.790332 w_k1_IBM_s004
                                 + 1.13795 w_k1_INTC_s004
                                 + 0.935781 w_k1_KO_s004
                                 + 1.34032 w_k1_MAT_s004 + 1.052 w_k1_MSFT_s004
                                 + 0.99454 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.07624 w_k1_ADM_s004
                                 + 0.95061 w_k1_AXP_s004
                                 + 1.30879 w_k1_DIS_s004
                                 + 1.22891 w_k1_FDX_s004
                                 + 0.788169 w_k1_IBM_s004
                                 + 0.96718 w_k1_INTC_s004
                                 + 1.00387 w_k1_KO_s004
                                 + 0.819748 w_k1_MAT_s004
                                 + 1.09848 w_k1_MSFT_s004
                                 + 1.08181 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.626249 w_k1_ADM_s004
                                 + 0.718815 w_k1_AXP_s004
                                 + 0.599854 w_k1_DIS_s004
                                 + 0.653114 w_k1_FDX_s004
                                 + 0.891513 w_k1_IBM_s004
                                 + 0.901921 w_k1_INTC_s004
                                 + 0.83547 w_k1_KO_s004 + 1.01608 w_k1_MAT_s004
                                 + 0.819776 w_k1_MSFT_s004
                                 + 0.785201 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.21922 w_k1_ADM_s005
                                 + 1.0977 w_k1_AXP_s005 + 1.31559 w_k1_DIS_s005
                                 + 1.20541 w_k1_FDX_s005
                                 + 1.01167 w_k1_IBM_s005
                                 + 0.96156 w_k1_INTC_s005
                                 + 1.15275 w_k1_KO_s005
                                 + 0.964236 w_k1_MAT_s005
                                 + 1.12634 w_k1_MSFT_s005
                                 + 1.14515 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.34436 w_k1_ADM_s005
                                 + 1.66206 w_k1_AXP_s005
                                 + 1.93988 w_k1_DIS_s005 + 1.4449 w_k1_FDX_s005
                                 + 1.29274 w_k1_IBM_s005
                                 + 1.21611 w_k1_INTC_s005
                                 + 1.46364 w_k1_KO_s005 + 1.20877 w_k1_MAT_s005
                                 + 1.21935 w_k1_MSFT_s005
                                 + 1.21928 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.796752 w_k1_ADM_s005
                                 + 0.805976 w_k1_AXP_s005
                                 + 0.580013 w_k1_DIS_s005
                                 + 0.753897 w_k1_FDX_s005
                                 + 0.854166 w_k1_IBM_s005
                                 + 0.70196 w_k1_INTC_s005
                                 + 0.694252 w_k1_KO_s005
                                 + 0.781637 w_k1_MAT_s005
                                 + 0.707709 w_k1_MSFT_s005
                                 + 0.69588 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.11542 w_k1_ADM_s005
                                 + 1.00705 w_k1_AXP_s005 + 0.9535 w_k1_DIS_s005
                                 + 1.02166 w_k1_FDX_s005
                                 + 1.07343 w_k1_IBM_s005
                                 + 0.992779 w_k1_INTC_s005
                                 + 0.976396 w_k1_KO_s005
                                 + 0.840529 w_k1_MAT_s005
                                 + 0.928683 w_k1_MSFT_s005
                                 + 0.946312 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.990878 w_k1_ADM_s005
                                 + 1.13379 w_k1_AXP_s005
                                 + 1.07001 w_k1_DIS_s005
                                 + 1.06513 w_k1_FDX_s005 + 1.431 w_k1_IBM_s005
                                 + 1.06153 w_k1_INTC_s005
                                 + 1.11587 w_k1_KO_s005 + 1.18009 w_k1_MAT_s005
                                 + 0.892709 w_k1_MSFT_s005
                                 + 1.0308 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.37033 w_k1_ADM_s005
                                 + 1.41602 w_k1_AXP_s005
                                 + 1.47532 w_k1_DIS_s005
                                 + 1.46359 w_k1_FDX_s005
                                 + 0.789982 w_k1_IBM_s005
                                 + 0.864805 w_k1_INTC_s005
                                 + 1.02177 w_k1_KO_s005
                                 + 0.718301 w_k1_MAT_s005
                                 + 1.04084 w_k1_MSFT_s005 + 1.034 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.00162 w_k1_ADM_s005
                                 + 1.52473 w_k1_AXP_s005
                                 + 0.884863 w_k1_DIS_s005
                                 + 1.04861 w_k1_FDX_s005
                                 + 1.18997 w_k1_IBM_s005
                                 + 1.14057 w_k1_INTC_s005
                                 + 1.09599 w_k1_KO_s005 + 1.56027 w_k1_MAT_s005
                                 + 1.02869 w_k1_MSFT_s005
                                 + 1.04404 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.917339 w_k1_ADM_s005
                                 + 0.814499 w_k1_AXP_s005
                                 + 0.974883 w_k1_DIS_s005
                                 + 1.02593 w_k1_FDX_s005
                                 + 0.755452 w_k1_IBM_s005
                                 + 0.925945 w_k1_INTC_s005
                                 + 0.83378 w_k1_KO_s005
                                 + 0.732802 w_k1_MAT_s005
                                 + 0.969235 w_k1_MSFT_s005
                                 + 0.975365 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.980102 w_k1_ADM_s006
                                 + 0.82538 w_k1_AXP_s006
                                 + 0.995464 w_k1_DIS_s006
                                 + 0.892542 w_k1_FDX_s006
                                 + 1.0359 w_k1_IBM_s006 + 1.0136 w_k1_INTC_s006
                                 + 0.959285 w_k1_KO_s006
                                 + 0.928259 w_k1_MAT_s006
                                 + 0.985956 w_k1_MSFT_s006
                                 + 0.910777 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.995114 w_k1_ADM_s006
                                 + 1.11547 w_k1_AXP_s006
                                 + 0.970096 w_k1_DIS_s006
                                 + 1.01887 w_k1_FDX_s006
                                 + 1.27942 w_k1_IBM_s006
                                 + 1.19303 w_k1_INTC_s006
                                 + 1.08742 w_k1_KO_s006
                                 + 0.973505 w_k1_MAT_s006
                                 + 1.02855 w_k1_MSFT_s006
                                 + 0.987016 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.1618 w_k1_ADM_s006
                                 + 1.05855 w_k1_AXP_s006
                                 + 1.40092 w_k1_DIS_s006
                                 + 1.35847 w_k1_FDX_s006 + 1.1903 w_k1_IBM_s006
                                 + 1.24681 w_k1_INTC_s006
                                 + 1.01737 w_k1_KO_s006 + 1.18019 w_k1_MAT_s006
                                 + 1.25101 w_k1_MSFT_s006
                                 + 0.974114 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.900606 w_k1_ADM_s006
                                 + 0.749771 w_k1_AXP_s006
                                 + 0.739771 w_k1_DIS_s006
                                 + 0.907921 w_k1_FDX_s006
                                 + 0.795416 w_k1_IBM_s006
                                 + 0.78819 w_k1_INTC_s006
                                 + 0.782221 w_k1_KO_s006
                                 + 0.904961 w_k1_MAT_s006
                                 + 0.845534 w_k1_MSFT_s006
                                 + 0.75235 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.07806 w_k1_ADM_s006
                                 + 0.877123 w_k1_AXP_s006
                                 + 1.20288 w_k1_DIS_s006
                                 + 1.03441 w_k1_FDX_s006 + 1.1842 w_k1_IBM_s006
                                 + 1.04923 w_k1_INTC_s006 + 1.1283 w_k1_KO_s006
                                 + 0.816177 w_k1_MAT_s006
                                 + 1.029 w_k1_MSFT_s006 + 0.925954 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.801039 w_k1_ADM_s006
                                 + 0.777229 w_k1_AXP_s006
                                 + 0.698837 w_k1_DIS_s006
                                 + 0.686181 w_k1_FDX_s006
                                 + 1.33384 w_k1_IBM_s006
                                 + 0.764508 w_k1_INTC_s006
                                 + 0.94825 w_k1_KO_s006 + 1.31949 w_k1_MAT_s006
                                 + 0.746156 w_k1_MSFT_s006
                                 + 0.790889 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.39669 w_k1_ADM_s006
                                 + 1.42555 w_k1_AXP_s006
                                 + 1.12102 w_k1_DIS_s006
                                 + 1.25219 w_k1_FDX_s006
                                 + 0.972357 w_k1_IBM_s006
                                 + 1.21757 w_k1_INTC_s006
                                 + 1.05028 w_k1_KO_s006 + 1.03492 w_k1_MAT_s006
                                 + 1.00615 w_k1_MSFT_s006
                                 + 0.895612 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.797744 w_k1_ADM_s006
                                 + 0.719875 w_k1_AXP_s006
                                 + 0.702651 w_k1_DIS_s006
                                 + 0.808047 w_k1_FDX_s006
                                 + 1.05297 w_k1_IBM_s006
                                 + 0.73768 w_k1_INTC_s006
                                 + 0.795674 w_k1_KO_s006
                                 + 1.30679 w_k1_MAT_s006
                                 + 0.787339 w_k1_MSFT_s006
                                 + 0.77594 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.839393 w_k1_ADM_s007
                                 + 1.00866 w_k1_AXP_s007
                                 + 0.856889 w_k1_DIS_s007
                                 + 0.926796 w_k1_FDX_s007
                                 + 0.954344 w_k1_IBM_s007
                                 + 1.02913 w_k1_INTC_s007
                                 + 0.835468 w_k1_KO_s007 + 1.1852 w_k1_MAT_s007
                                 + 0.979889 w_k1_MSFT_s007
                                 + 0.868737 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.862612 w_k1_ADM_s007
                                 + 1.22696 w_k1_AXP_s007
                                 + 1.34095 w_k1_DIS_s007
                                 + 1.12619 w_k1_FDX_s007
                                 + 1.21497 w_k1_IBM_s007
                                 + 1.18197 w_k1_INTC_s007
                                 + 1.11944 w_k1_KO_s007 + 1.34903 w_k1_MAT_s007
                                 + 1.30259 w_k1_MSFT_s007
                                 + 1.11111 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.84973 w_k1_ADM_s007
                                 + 1.34059 w_k1_AXP_s007
                                 + 1.05871 w_k1_DIS_s007
                                 + 1.00516 w_k1_FDX_s007
                                 + 1.56819 w_k1_IBM_s007
                                 + 1.1182 w_k1_INTC_s007 + 1.07583 w_k1_KO_s007
                                 + 1.81596 w_k1_MAT_s007
                                 + 1.05014 w_k1_MSFT_s007
                                 + 0.978698 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.17739 w_k1_ADM_s007
                                 + 1.73561 w_k1_AXP_s007
                                 + 1.29088 w_k1_DIS_s007
                                 + 1.20696 w_k1_FDX_s007
                                 + 1.16143 w_k1_IBM_s007
                                 + 1.07343 w_k1_INTC_s007
                                 + 1.07314 w_k1_KO_s007 + 1.41807 w_k1_MAT_s007
                                 + 1.0798 w_k1_MSFT_s007 + 1.03343 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.898044 w_k1_ADM_s007
                                 + 0.735752 w_k1_AXP_s007
                                 + 0.763811 w_k1_DIS_s007
                                 + 0.857011 w_k1_FDX_s007
                                 + 0.726892 w_k1_IBM_s007
                                 + 0.670656 w_k1_INTC_s007
                                 + 0.681501 w_k1_KO_s007
                                 + 0.935323 w_k1_MAT_s007
                                 + 0.89917 w_k1_MSFT_s007
                                 + 0.786154 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.16607 w_k1_ADM_s007
                                 + 1.21709 w_k1_AXP_s007
                                 + 1.47785 w_k1_DIS_s007
                                 + 1.29711 w_k1_FDX_s007 + 0.8556 w_k1_IBM_s007
                                 + 1.16037 w_k1_INTC_s007
                                 + 0.991937 w_k1_KO_s007
                                 + 1.00251 w_k1_MAT_s007
                                 + 1.37294 w_k1_MSFT_s007
                                 + 1.08222 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.688752 w_k1_ADM_s007
                                 + 0.71666 w_k1_AXP_s007
                                 + 0.50406 w_k1_DIS_s007
                                 + 0.649197 w_k1_FDX_s007
                                 + 0.946057 w_k1_IBM_s007
                                 + 0.83512 w_k1_INTC_s007
                                 + 0.716273 w_k1_KO_s007
                                 + 0.951595 w_k1_MAT_s007
                                 + 0.751023 w_k1_MSFT_s007
                                 + 0.736107 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.848853 w_k1_ADM_s007
                                 + 0.900931 w_k1_AXP_s007
                                 + 0.958232 w_k1_DIS_s007
                                 + 0.834889 w_k1_FDX_s007
                                 + 1.29698 w_k1_IBM_s007
                                 + 0.845785 w_k1_INTC_s007
                                 + 1.04176 w_k1_KO_s007
                                 + 0.872317 w_k1_MAT_s007
                                 + 0.876049 w_k1_MSFT_s007
                                 + 0.939357 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.828962 w_k1_ADM_s010
                                 + 0.707736 w_k1_AXP_s010
                                 + 0.850987 w_k1_DIS_s010
                                 + 0.782796 w_k1_FDX_s010
                                 + 1.11534 w_k1_IBM_s010
                                 + 0.842407 w_k1_INTC_s010
                                 + 1.19086 w_k1_KO_s010
                                 + 0.845642 w_k1_MAT_s010
                                 + 0.851374 w_k1_MSFT_s010
                                 + 1.03774 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.28665 w_k1_ADM_s010
                                 + 1.29485 w_k1_AXP_s010
                                 + 1.32343 w_k1_DIS_s010
                                 + 1.36981 w_k1_FDX_s010
                                 + 0.909941 w_k1_IBM_s010
                                 + 1.16856 w_k1_INTC_s010
                                 + 1.28445 w_k1_KO_s010
                                 + 0.849872 w_k1_MAT_s010
                                 + 1.2173 w_k1_MSFT_s010 + 1.24162 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.712131 w_k1_ADM_s010
                                 + 0.622386 w_k1_AXP_s010
                                 + 0.656486 w_k1_DIS_s010
                                 + 0.767867 w_k1_FDX_s010
                                 + 0.78149 w_k1_IBM_s010
                                 + 0.74158 w_k1_INTC_s010
                                 + 0.827355 w_k1_KO_s010
                                 + 0.50858 w_k1_MAT_s010
                                 + 0.841874 w_k1_MSFT_s010
                                 + 0.791189 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.98434 w_k1_ADM_s010
                                 + 1.13143 w_k1_AXP_s010
                                 + 1.25083 w_k1_DIS_s010
                                 + 1.10975 w_k1_FDX_s010 + 1.3084 w_k1_IBM_s010
                                 + 0.917793 w_k1_INTC_s010
                                 + 1.44049 w_k1_KO_s010 + 1.49865 w_k1_MAT_s010
                                 + 1.06642 w_k1_MSFT_s010
                                 + 1.21732 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.45686 w_k1_ADM_s010
                                 + 1.3038 w_k1_AXP_s010 + 1.68042 w_k1_DIS_s010
                                 + 1.37989 w_k1_FDX_s010
                                 + 1.35296 w_k1_IBM_s010
                                 + 1.19692 w_k1_INTC_s010
                                 + 1.55157 w_k1_KO_s010 + 1.33698 w_k1_MAT_s010
                                 + 1.52938 w_k1_MSFT_s010 + 1.4029 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.01464 w_k1_ADM_s010
                                 + 1.14476 w_k1_AXP_s010
                                 + 0.960844 w_k1_DIS_s010
                                 + 1.01326 w_k1_FDX_s010
                                 + 1.07598 w_k1_IBM_s010
                                 + 1.04773 w_k1_INTC_s010
                                 + 1.05373 w_k1_KO_s010 + 1.19452 w_k1_MAT_s010
                                 + 0.997656 w_k1_MSFT_s010
                                 + 0.970551 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.15631 w_k1_ADM_s010
                                 + 1.1535 w_k1_AXP_s010 + 1.11507 w_k1_DIS_s010
                                 + 1.16575 w_k1_FDX_s010
                                 + 0.947903 w_k1_IBM_s010
                                 + 0.91274 w_k1_INTC_s010
                                 + 1.24207 w_k1_KO_s010 + 0.916 w_k1_MAT_s010
                                 + 1.13672 w_k1_MSFT_s010
                                 + 1.28946 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.948262 w_k1_ADM_s010
                                 + 0.714995 w_k1_AXP_s010
                                 + 0.633999 w_k1_DIS_s010
                                 + 0.859141 w_k1_FDX_s010
                                 + 0.740734 w_k1_IBM_s010
                                 + 0.897742 w_k1_INTC_s010
                                 + 0.761975 w_k1_KO_s010
                                 + 0.444918 w_k1_MAT_s010
                                 + 0.761358 w_k1_MSFT_s010
                                 + 0.935593 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.07843 w_k1_ADM_s011
                                 + 0.962067 w_k1_AXP_s011
                                 + 1.00605 w_k1_DIS_s011
                                 + 0.978417 w_k1_FDX_s011
                                 + 1.10858 w_k1_IBM_s011
                                 + 0.876944 w_k1_INTC_s011
                                 + 1.00132 w_k1_KO_s011
                                 + 0.984183 w_k1_MAT_s011
                                 + 0.948481 w_k1_MSFT_s011
                                 + 0.947265 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.923033 w_k1_ADM_s011
                                 + 1.07102 w_k1_AXP_s011
                                 + 1.15017 w_k1_DIS_s011
                                 + 1.13297 w_k1_FDX_s011
                                 + 1.00929 w_k1_IBM_s011
                                 + 0.936557 w_k1_INTC_s011
                                 + 1.04271 w_k1_KO_s011 + 1.03815 w_k1_MAT_s011
                                 + 1.03899 w_k1_MSFT_s011
                                 + 1.04698 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.770709 w_k1_ADM_s011
                                 + 0.728886 w_k1_AXP_s011
                                 + 0.782308 w_k1_DIS_s011
                                 + 0.879184 w_k1_FDX_s011
                                 + 0.744248 w_k1_IBM_s011
                                 + 0.888453 w_k1_INTC_s011
                                 + 0.754408 w_k1_KO_s011
                                 + 0.631785 w_k1_MAT_s011
                                 + 0.843694 w_k1_MSFT_s011
                                 + 0.830692 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.824338 w_k1_ADM_s011
                                 + 1.00858 w_k1_AXP_s011
                                 + 0.772197 w_k1_DIS_s011
                                 + 0.833546 w_k1_FDX_s011
                                 + 1.32099 w_k1_IBM_s011
                                 + 0.846041 w_k1_INTC_s011
                                 + 1.03558 w_k1_KO_s011
                                 + 0.987501 w_k1_MAT_s011
                                 + 0.726148 w_k1_MSFT_s011
                                 + 0.900319 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.09753 w_k1_ADM_s011
                                 + 1.32499 w_k1_AXP_s011
                                 + 1.41832 w_k1_DIS_s011
                                 + 1.14059 w_k1_FDX_s011
                                 + 1.18041 w_k1_IBM_s011
                                 + 1.16838 w_k1_INTC_s011
                                 + 1.07032 w_k1_KO_s011
                                 + 0.744259 w_k1_MAT_s011
                                 + 0.981548 w_k1_MSFT_s011
                                 + 0.98022 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.01991 w_k1_ADM_s011
                                 + 1.40678 w_k1_AXP_s011
                                 + 1.46289 w_k1_DIS_s011
                                 + 1.23929 w_k1_FDX_s011
                                 + 1.54109 w_k1_IBM_s011
                                 + 1.16164 w_k1_INTC_s011
                                 + 1.40401 w_k1_KO_s011 + 1.18521 w_k1_MAT_s011
                                 + 1.10728 w_k1_MSFT_s011 + 1.0978 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.804006 w_k1_ADM_s011
                                 + 1.15598 w_k1_AXP_s011
                                 + 0.88638 w_k1_DIS_s011
                                 + 0.94201 w_k1_FDX_s011
                                 + 0.836664 w_k1_IBM_s011
                                 + 1.03898 w_k1_INTC_s011
                                 + 0.874993 w_k1_KO_s011
                                 + 0.720998 w_k1_MAT_s011
                                 + 0.850514 w_k1_MSFT_s011
                                 + 0.908007 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.948189 w_k1_ADM_s011
                                 + 0.944032 w_k1_AXP_s011
                                 + 0.934417 w_k1_DIS_s011
                                 + 1.04554 w_k1_FDX_s011
                                 + 0.690032 w_k1_IBM_s011
                                 + 0.894805 w_k1_INTC_s011
                                 + 0.679362 w_k1_KO_s011
                                 + 0.690135 w_k1_MAT_s011
                                 + 0.897151 w_k1_MSFT_s011
                                 + 0.857459 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.705604 w_k1_ADM_s012
                                 + 0.627851 w_k1_AXP_s012
                                 + 0.664269 w_k1_DIS_s012
                                 + 0.776685 w_k1_FDX_s012
                                 + 0.729262 w_k1_IBM_s012
                                 + 0.832591 w_k1_INTC_s012
                                 + 0.887807 w_k1_KO_s012
                                 + 0.534856 w_k1_MAT_s012
                                 + 0.837694 w_k1_MSFT_s012
                                 + 0.878211 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.952872 w_k1_ADM_s012
                                 + 0.889136 w_k1_AXP_s012
                                 + 0.71257 w_k1_DIS_s012
                                 + 0.983943 w_k1_FDX_s012
                                 + 0.931944 w_k1_IBM_s012
                                 + 0.982597 w_k1_INTC_s012
                                 + 0.907588 w_k1_KO_s012
                                 + 0.830013 w_k1_MAT_s012
                                 + 1.01487 w_k1_MSFT_s012
                                 + 0.932265 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.05465 w_k1_ADM_s012
                                 + 0.801078 w_k1_AXP_s012
                                 + 0.766971 w_k1_DIS_s012
                                 + 1.00429 w_k1_FDX_s012
                                 + 0.693031 w_k1_IBM_s012
                                 + 0.808734 w_k1_INTC_s012
                                 + 0.767292 w_k1_KO_s012
                                 + 0.532927 w_k1_MAT_s012
                                 + 0.913508 w_k1_MSFT_s012
                                 + 0.969947 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.00541 w_k1_ADM_s012
                                 + 1.06292 w_k1_AXP_s012
                                 + 1.23158 w_k1_DIS_s012
                                 + 1.21045 w_k1_FDX_s012
                                 + 0.666872 w_k1_IBM_s012
                                 + 1.13411 w_k1_INTC_s012
                                 + 0.985327 w_k1_KO_s012
                                 + 0.624947 w_k1_MAT_s012
                                 + 1.2971 w_k1_MSFT_s012 + 1.03337 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.807543 w_k1_ADM_s012
                                 + 1.15795 w_k1_AXP_s012
                                 + 1.22904 w_k1_DIS_s012
                                 + 1.27786 w_k1_FDX_s012
                                 + 1.21096 w_k1_IBM_s012
                                 + 1.18007 w_k1_INTC_s012
                                 + 1.19998 w_k1_KO_s012 + 1.49352 w_k1_MAT_s012
                                 + 1.27057 w_k1_MSFT_s012
                                 + 1.20667 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.785702 w_k1_ADM_s012
                                 + 0.852771 w_k1_AXP_s012
                                 + 0.851686 w_k1_DIS_s012
                                 + 0.888618 w_k1_FDX_s012
                                 + 1.58113 w_k1_IBM_s012
                                 + 0.962469 w_k1_INTC_s012
                                 + 1.12235 w_k1_KO_s012 + 1.0643 w_k1_MAT_s012
                                 + 0.931914 w_k1_MSFT_s012
                                 + 1.14217 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.02078 w_k1_ADM_s012
                                 + 1.13667 w_k1_AXP_s012
                                 + 0.928122 w_k1_DIS_s012
                                 + 0.998535 w_k1_FDX_s012
                                 + 0.904978 w_k1_IBM_s012
                                 + 0.822262 w_k1_INTC_s012
                                 + 0.888899 w_k1_KO_s012
                                 + 0.940605 w_k1_MAT_s012
                                 + 0.949972 w_k1_MSFT_s012
                                 + 0.946805 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.05529 w_k1_ADM_s012
                                 + 1.01657 w_k1_AXP_s012
                                 + 0.789696 w_k1_DIS_s012
                                 + 0.890127 w_k1_FDX_s012
                                 + 1.05489 w_k1_IBM_s012
                                 + 0.868343 w_k1_INTC_s012
                                 + 0.948195 w_k1_KO_s012
                                 + 1.16122 w_k1_MAT_s012
                                 + 0.963361 w_k1_MSFT_s012
                                 + 0.986794 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.05241 w_k1_ADM_s013
                                 + 1.13322 w_k1_AXP_s013
                                 + 0.940636 w_k1_DIS_s013
                                 + 1.11591 w_k1_FDX_s013
                                 + 0.635135 w_k1_IBM_s013
                                 + 0.908363 w_k1_INTC_s013
                                 + 0.985547 w_k1_KO_s013
                                 + 0.989421 w_k1_MAT_s013
                                 + 1.02251 w_k1_MSFT_s013
                                 + 1.04573 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.653529 w_k1_ADM_s013
                                 + 0.64597 w_k1_AXP_s013
                                 + 0.571199 w_k1_DIS_s013
                                 + 0.704129 w_k1_FDX_s013
                                 + 0.669404 w_k1_IBM_s013
                                 + 0.887201 w_k1_INTC_s013
                                 + 0.798315 w_k1_KO_s013
                                 + 0.726647 w_k1_MAT_s013
                                 + 0.778717 w_k1_MSFT_s013
                                 + 0.730207 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.924376 w_k1_ADM_s013
                                 + 1.29871 w_k1_AXP_s013
                                 + 1.54472 w_k1_DIS_s013
                                 + 1.15836 w_k1_FDX_s013 + 1.3596 w_k1_IBM_s013
                                 + 1.10386 w_k1_INTC_s013 + 1.3429 w_k1_KO_s013
                                 + 1.25909 w_k1_MAT_s013
                                 + 1.10808 w_k1_MSFT_s013
                                 + 1.20691 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.847954 w_k1_ADM_s013
                                 + 1.0408 w_k1_AXP_s013
                                 + 0.812442 w_k1_DIS_s013
                                 + 0.918288 w_k1_FDX_s013
                                 + 0.798305 w_k1_IBM_s013
                                 + 0.877536 w_k1_INTC_s013
                                 + 0.837727 w_k1_KO_s013
                                 + 0.853011 w_k1_MAT_s013
                                 + 0.815991 w_k1_MSFT_s013
                                 + 0.965599 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.89175 w_k1_ADM_s013
                                 + 1.2043 w_k1_AXP_s013 + 1.09905 w_k1_DIS_s013
                                 + 1.05478 w_k1_FDX_s013
                                 + 1.80939 w_k1_IBM_s013
                                 + 1.26495 w_k1_INTC_s013
                                 + 1.43578 w_k1_KO_s013 + 2.13649 w_k1_MAT_s013
                                 + 1.027 w_k1_MSFT_s013 + 1.05939 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.625034 w_k1_ADM_s013
                                 + 0.663035 w_k1_AXP_s013
                                 + 0.685258 w_k1_DIS_s013
                                 + 0.700403 w_k1_FDX_s013
                                 + 1.09688 w_k1_IBM_s013
                                 + 0.707251 w_k1_INTC_s013
                                 + 0.869598 w_k1_KO_s013
                                 + 1.07426 w_k1_MAT_s013
                                 + 0.731363 w_k1_MSFT_s013
                                 + 0.795857 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.07043 w_k1_ADM_s013
                                 + 1.36673 w_k1_AXP_s013
                                 + 1.08168 w_k1_DIS_s013
                                 + 1.09459 w_k1_FDX_s013
                                 + 1.26473 w_k1_IBM_s013
                                 + 0.932811 w_k1_INTC_s013
                                 + 1.16103 w_k1_KO_s013 + 1.35779 w_k1_MAT_s013
                                 + 0.884576 w_k1_MSFT_s013
                                 + 1.14061 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.03847 w_k1_ADM_s013
                                 + 1.13879 w_k1_AXP_s013
                                 + 0.912926 w_k1_DIS_s013
                                 + 0.984934 w_k1_FDX_s013
                                 + 1.05035 w_k1_IBM_s013
                                 + 0.962603 w_k1_INTC_s013
                                 + 1.05296 w_k1_KO_s013
                                 + 0.970677 w_k1_MAT_s013
                                 + 0.820896 w_k1_MSFT_s013
                                 + 1.17953 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.804122 w_k1_ADM_s014
                                 + 0.694044 w_k1_AXP_s014
                                 + 0.938279 w_k1_DIS_s014
                                 + 0.915515 w_k1_FDX_s014
                                 + 0.872942 w_k1_IBM_s014
                                 + 1.0133 w_k1_INTC_s014
                                 + 0.804388 w_k1_KO_s014
                                 + 0.594406 w_k1_MAT_s014
                                 + 0.895426 w_k1_MSFT_s014
                                 + 0.960165 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.990932 w_k1_ADM_s014
                                 + 0.9867 w_k1_AXP_s014 + 1.42594 w_k1_DIS_s014
                                 + 1.163 w_k1_FDX_s014 + 1.19055 w_k1_IBM_s014
                                 + 1.14957 w_k1_INTC_s014
                                 + 1.18487 w_k1_KO_s014 + 1.23312 w_k1_MAT_s014
                                 + 1.20903 w_k1_MSFT_s014 + 1.1426 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.877384 w_k1_ADM_s014
                                 + 0.908588 w_k1_AXP_s014
                                 + 0.943919 w_k1_DIS_s014
                                 + 0.983661 w_k1_FDX_s014
                                 + 0.779594 w_k1_IBM_s014
                                 + 0.98381 w_k1_INTC_s014
                                 + 0.862316 w_k1_KO_s014
                                 + 0.754401 w_k1_MAT_s014
                                 + 0.921673 w_k1_MSFT_s014
                                 + 1.00157 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.740915 w_k1_ADM_s014
                                 + 0.684861 w_k1_AXP_s014
                                 + 0.713318 w_k1_DIS_s014
                                 + 0.80751 w_k1_FDX_s014
                                 + 0.629463 w_k1_IBM_s014
                                 + 0.96802 w_k1_INTC_s014
                                 + 0.675066 w_k1_KO_s014
                                 + 0.455101 w_k1_MAT_s014
                                 + 0.827493 w_k1_MSFT_s014
                                 + 0.867643 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.962775 w_k1_ADM_s014
                                 + 0.742307 w_k1_AXP_s014
                                 + 0.683589 w_k1_DIS_s014
                                 + 0.782062 w_k1_FDX_s014
                                 + 1.0295 w_k1_IBM_s014
                                 + 0.775893 w_k1_INTC_s014
                                 + 0.785254 w_k1_KO_s014
                                 + 0.752583 w_k1_MAT_s014
                                 + 0.702049 w_k1_MSFT_s014
                                 + 0.832238 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.940401 w_k1_ADM_s014
                                 + 1.00818 w_k1_AXP_s014
                                 + 1.04302 w_k1_DIS_s014
                                 + 0.979827 w_k1_FDX_s014
                                 + 1.26547 w_k1_IBM_s014
                                 + 0.965282 w_k1_INTC_s014
                                 + 1.05477 w_k1_KO_s014 + 1.44543 w_k1_MAT_s014
                                 + 0.971008 w_k1_MSFT_s014
                                 + 0.996977 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.14215 w_k1_ADM_s014
                                 + 1.2287 w_k1_AXP_s014 + 1.78512 w_k1_DIS_s014
                                 + 1.38315 w_k1_FDX_s014 + 1.4595 w_k1_IBM_s014
                                 + 1.23119 w_k1_INTC_s014
                                 + 1.49626 w_k1_KO_s014 + 1.21833 w_k1_MAT_s014
                                 + 1.34784 w_k1_MSFT_s014 + 1.3164 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.27737 w_k1_ADM_s014
                                 + 1.09577 w_k1_AXP_s014
                                 + 1.13994 w_k1_DIS_s014
                                 + 1.25722 w_k1_FDX_s014
                                 + 0.760154 w_k1_IBM_s014
                                 + 1.1481 w_k1_INTC_s014
                                 + 0.947799 w_k1_KO_s014
                                 + 0.681843 w_k1_MAT_s014
                                 + 1.02111 w_k1_MSFT_s014
                                 + 1.04349 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.04816 w_k1_ADM_s015
                                 + 1.16163 w_k1_AXP_s015
                                 + 1.04936 w_k1_DIS_s015
                                 + 1.20085 w_k1_FDX_s015
                                 + 0.738653 w_k1_IBM_s015
                                 + 1.20296 w_k1_INTC_s015
                                 + 0.911648 w_k1_KO_s015
                                 + 0.839956 w_k1_MAT_s015
                                 + 1.19497 w_k1_MSFT_s015 + 1.0706 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.985674 w_k1_ADM_s015
                                 + 1.18888 w_k1_AXP_s015
                                 + 0.79436 w_k1_DIS_s015
                                 + 1.00703 w_k1_FDX_s015
                                 + 0.782651 w_k1_IBM_s015
                                 + 0.801824 w_k1_INTC_s015
                                 + 0.814309 w_k1_KO_s015
                                 + 1.17897 w_k1_MAT_s015
                                 + 0.9072 w_k1_MSFT_s015
                                 + 0.978053 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.1859 w_k1_ADM_s015
                                 + 1.11295 w_k1_AXP_s015
                                 + 1.50408 w_k1_DIS_s015
                                 + 1.15217 w_k1_FDX_s015
                                 + 1.40885 w_k1_IBM_s015
                                 + 0.847144 w_k1_INTC_s015
                                 + 1.40119 w_k1_KO_s015 + 3.00355 w_k1_MAT_s015
                                 + 1.19159 w_k1_MSFT_s015
                                 + 1.14758 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.896125 w_k1_ADM_s015
                                 + 0.952826 w_k1_AXP_s015
                                 + 1.21365 w_k1_DIS_s015
                                 + 1.00875 w_k1_FDX_s015
                                 + 0.96326 w_k1_IBM_s015
                                 + 0.997904 w_k1_INTC_s015
                                 + 0.977555 w_k1_KO_s015
                                 + 0.708264 w_k1_MAT_s015
                                 + 1.0276 w_k1_MSFT_s015
                                 + 0.894118 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.26564 w_k1_ADM_s015
                                 + 1.2228 w_k1_AXP_s015 + 1.77774 w_k1_DIS_s015
                                 + 1.28558 w_k1_FDX_s015
                                 + 1.08123 w_k1_IBM_s015
                                 + 1.10247 w_k1_INTC_s015 + 1.1462 w_k1_KO_s015
                                 + 1.29159 w_k1_MAT_s015
                                 + 1.28421 w_k1_MSFT_s015
                                 + 1.17417 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.900481 w_k1_ADM_s015
                                 + 0.728765 w_k1_AXP_s015
                                 + 0.72058 w_k1_DIS_s015
                                 + 0.85336 w_k1_FDX_s015
                                 + 0.875909 w_k1_IBM_s015
                                 + 0.737617 w_k1_INTC_s015
                                 + 0.808437 w_k1_KO_s015
                                 + 0.655086 w_k1_MAT_s015
                                 + 0.825334 w_k1_MSFT_s015
                                 + 0.839785 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.49673 w_k1_ADM_s015
                                 + 1.60631 w_k1_AXP_s015
                                 + 1.24657 w_k1_DIS_s015
                                 + 1.15151 w_k1_FDX_s015
                                 + 1.36117 w_k1_IBM_s015
                                 + 0.857858 w_k1_INTC_s015
                                 + 1.07448 w_k1_KO_s015 + 1.79468 w_k1_MAT_s015
                                 + 0.975376 w_k1_MSFT_s015
                                 + 1.15946 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.06542 w_k1_ADM_s015
                                 + 1.01749 w_k1_AXP_s015
                                 + 1.01378 w_k1_DIS_s015
                                 + 0.997007 w_k1_FDX_s015
                                 + 1.02548 w_k1_IBM_s015
                                 + 0.809293 w_k1_INTC_s015
                                 + 0.929789 w_k1_KO_s015 + 1.0389 w_k1_MAT_s015
                                 + 0.965773 w_k1_MSFT_s015
                                 + 1.0198 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.4707 w_k1_ADM_s016
                                 + 1.09858 w_k1_AXP_s016
                                 + 1.51882 w_k1_DIS_s016
                                 + 1.28979 w_k1_FDX_s016
                                 + 0.899445 w_k1_IBM_s016
                                 + 1.04525 w_k1_INTC_s016
                                 + 1.30476 w_k1_KO_s016 + 1.02156 w_k1_MAT_s016
                                 + 1.36812 w_k1_MSFT_s016
                                 + 1.49467 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.990226 w_k1_ADM_s016
                                 + 0.850537 w_k1_AXP_s016
                                 + 0.869362 w_k1_DIS_s016
                                 + 0.89761 w_k1_FDX_s016
                                 + 0.657566 w_k1_IBM_s016
                                 + 0.728383 w_k1_INTC_s016
                                 + 0.813241 w_k1_KO_s016
                                 + 0.60664 w_k1_MAT_s016
                                 + 0.855665 w_k1_MSFT_s016
                                 + 0.883527 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.860298 w_k1_ADM_s016
                                 + 0.757952 w_k1_AXP_s016
                                 + 1.01339 w_k1_DIS_s016
                                 + 0.964028 w_k1_FDX_s016
                                 + 0.975186 w_k1_IBM_s016
                                 + 0.634163 w_k1_INTC_s016
                                 + 0.927745 w_k1_KO_s016
                                 + 0.971196 w_k1_MAT_s016
                                 + 0.944666 w_k1_MSFT_s016
                                 + 0.991339 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.877339 w_k1_ADM_s016
                                 + 1.04327 w_k1_AXP_s016
                                 + 0.837358 w_k1_DIS_s016
                                 + 0.86012 w_k1_FDX_s016
                                 + 1.15785 w_k1_IBM_s016
                                 + 0.849625 w_k1_INTC_s016
                                 + 1.07731 w_k1_KO_s016 + 1.55933 w_k1_MAT_s016
                                 + 0.965106 w_k1_MSFT_s016
                                 + 1.09577 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.83554 w_k1_ADM_s016
                                 + 0.550438 w_k1_AXP_s016
                                 + 0.654357 w_k1_DIS_s016
                                 + 0.581784 w_k1_FDX_s016
                                 + 1.07266 w_k1_IBM_s016
                                 + 0.597622 w_k1_INTC_s016
                                 + 0.833618 w_k1_KO_s016
                                 + 0.688689 w_k1_MAT_s016
                                 + 0.63571 w_k1_MSFT_s016
                                 + 0.814381 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.625632 w_k1_ADM_s016
                                 + 0.707282 w_k1_AXP_s016
                                 + 0.575398 w_k1_DIS_s016
                                 + 0.705833 w_k1_FDX_s016
                                 + 0.888821 w_k1_IBM_s016
                                 + 0.64609 w_k1_INTC_s016
                                 + 0.646165 w_k1_KO_s016
                                 + 1.01714 w_k1_MAT_s016
                                 + 0.638197 w_k1_MSFT_s016
                                 + 0.916032 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.20224 w_k1_ADM_s016
                                 + 1.52127 w_k1_AXP_s016 + 1.7432 w_k1_DIS_s016
                                 + 1.40048 w_k1_FDX_s016
                                 + 1.60483 w_k1_IBM_s016
                                 + 1.45081 w_k1_INTC_s016
                                 + 1.64532 w_k1_KO_s016 + 1.56126 w_k1_MAT_s016
                                 + 1.52218 w_k1_MSFT_s016
                                 + 1.22356 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.944471 w_k1_ADM_s016
                                 + 1.03381 w_k1_AXP_s016
                                 + 1.12332 w_k1_DIS_s016
                                 + 0.950803 w_k1_FDX_s016
                                 + 1.4059 w_k1_IBM_s016
                                 + 0.78083 w_k1_INTC_s016
                                 + 1.11256 w_k1_KO_s016 + 1.01274 w_k1_MAT_s016
                                 + 0.94311 w_k1_MSFT_s016
                                 + 0.864301 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.800336 w_k1_ADM_s017
                                 + 0.810738 w_k1_AXP_s017
                                 + 0.893188 w_k1_DIS_s017
                                 + 0.84968 w_k1_FDX_s017
                                 + 1.03029 w_k1_IBM_s017
                                 + 0.937164 w_k1_INTC_s017
                                 + 1.06742 w_k1_KO_s017 + 0.89967 w_k1_MAT_s017
                                 + 0.913909 w_k1_MSFT_s017
                                 + 0.943234 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.808831 w_k1_ADM_s017
                                 + 1.14827 w_k1_AXP_s017
                                 + 0.776494 w_k1_DIS_s017
                                 + 0.881381 w_k1_FDX_s017
                                 + 0.897245 w_k1_IBM_s017
                                 + 1.05159 w_k1_INTC_s017
                                 + 0.860493 w_k1_KO_s017
                                 + 0.846937 w_k1_MAT_s017
                                 + 0.85855 w_k1_MSFT_s017
                                 + 1.04351 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.87611 w_k1_ADM_s017
                                 + 1.18415 w_k1_AXP_s017
                                 + 1.00887 w_k1_DIS_s017
                                 + 0.970938 w_k1_FDX_s017
                                 + 1.50447 w_k1_IBM_s017
                                 + 0.932193 w_k1_INTC_s017
                                 + 1.03372 w_k1_KO_s017 + 1.50675 w_k1_MAT_s017
                                 + 0.836428 w_k1_MSFT_s017
                                 + 1.16796 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.913623 w_k1_ADM_s017
                                 + 1.95715 w_k1_AXP_s017
                                 + 1.54042 w_k1_DIS_s017
                                 + 1.25588 w_k1_FDX_s017
                                 + 1.59382 w_k1_IBM_s017
                                 + 1.2955 w_k1_INTC_s017 + 1.35389 w_k1_KO_s017
                                 + 2.21016 w_k1_MAT_s017
                                 + 1.38829 w_k1_MSFT_s017 + 1.3148 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.855697 w_k1_ADM_s017
                                 + 1.06338 w_k1_AXP_s017
                                 + 1.06398 w_k1_DIS_s017
                                 + 1.13322 w_k1_FDX_s017
                                 + 0.801558 w_k1_IBM_s017
                                 + 1.08832 w_k1_INTC_s017
                                 + 0.92614 w_k1_KO_s017 + 1.00551 w_k1_MAT_s017
                                 + 1.07434 w_k1_MSFT_s017
                                 + 1.05484 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.914145 w_k1_ADM_s017
                                 + 1.01405 w_k1_AXP_s017 + 1.1969 w_k1_DIS_s017
                                 + 0.955141 w_k1_FDX_s017
                                 + 1.35362 w_k1_IBM_s017
                                 + 1.16793 w_k1_INTC_s017
                                 + 1.36996 w_k1_KO_s017 + 1.1169 w_k1_MAT_s017
                                 + 1.23052 w_k1_MSFT_s017
                                 + 1.16271 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.641373 w_k1_ADM_s017
                                 + 0.764315 w_k1_AXP_s017
                                 + 0.655773 w_k1_DIS_s017
                                 + 0.805571 w_k1_FDX_s017
                                 + 0.78391 w_k1_IBM_s017
                                 + 0.863555 w_k1_INTC_s017
                                 + 0.731081 w_k1_KO_s017 + 0.6079 w_k1_MAT_s017
                                 + 0.817397 w_k1_MSFT_s017
                                 + 0.795478 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.10901 w_k1_ADM_s017
                                 + 1.40864 w_k1_AXP_s017 + 1.4984 w_k1_DIS_s017
                                 + 1.34766 w_k1_FDX_s017 + 1.1841 w_k1_IBM_s017
                                 + 1.41934 w_k1_INTC_s017
                                 + 1.36129 w_k1_KO_s017 + 1.66019 w_k1_MAT_s017
                                 + 1.53281 w_k1_MSFT_s017
                                 + 1.31119 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.905071 w_k1_ADM_s100
                                 + 0.928649 w_k1_AXP_s100
                                 + 0.973541 w_k1_DIS_s100
                                 + 0.948043 w_k1_FDX_s100
                                 + 1.10697 w_k1_IBM_s100
                                 + 0.933613 w_k1_INTC_s100
                                 + 1.10255 w_k1_KO_s100 + 1.07537 w_k1_MAT_s100
                                 + 1.02227 w_k1_MSFT_s100
                                 + 0.87129 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.14299 w_k1_ADM_s100
                                 + 1.37271 w_k1_AXP_s100
                                 + 1.37249 w_k1_DIS_s100
                                 + 1.12619 w_k1_FDX_s100
                                 + 1.63442 w_k1_IBM_s100
                                 + 1.19781 w_k1_INTC_s100
                                 + 1.42361 w_k1_KO_s100 + 2.01501 w_k1_MAT_s100
                                 + 1.17668 w_k1_MSFT_s100
                                 + 1.07266 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.957406 w_k1_ADM_s100
                                 + 0.72473 w_k1_AXP_s100
                                 + 0.895866 w_k1_DIS_s100
                                 + 0.819282 w_k1_FDX_s100
                                 + 0.973575 w_k1_IBM_s100
                                 + 0.714903 w_k1_INTC_s100
                                 + 0.921339 w_k1_KO_s100
                                 + 0.772149 w_k1_MAT_s100
                                 + 0.843926 w_k1_MSFT_s100
                                 + 0.808253 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.23711 w_k1_ADM_s100
                                 + 0.989227 w_k1_AXP_s100
                                 + 0.70177 w_k1_DIS_s100
                                 + 0.851985 w_k1_FDX_s100
                                 + 0.88683 w_k1_IBM_s100
                                 + 0.822202 w_k1_INTC_s100
                                 + 0.839933 w_k1_KO_s100
                                 + 0.847084 w_k1_MAT_s100
                                 + 0.874461 w_k1_MSFT_s100
                                 + 0.863334 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.26735 w_k1_ADM_s100
                                 + 0.965146 w_k1_AXP_s100
                                 + 1.18991 w_k1_DIS_s100
                                 + 0.98406 w_k1_FDX_s100 + 1.4604 w_k1_IBM_s100
                                 + 0.987128 w_k1_INTC_s100
                                 + 1.4233 w_k1_KO_s100 + 1.46086 w_k1_MAT_s100
                                 + 1.12838 w_k1_MSFT_s100
                                 + 1.08368 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.20309 w_k1_ADM_s100
                                 + 1.26107 w_k1_AXP_s100
                                 + 1.09175 w_k1_DIS_s100
                                 + 1.03021 w_k1_FDX_s100
                                 + 1.16165 w_k1_IBM_s100
                                 + 1.02105 w_k1_INTC_s100
                                 + 1.10634 w_k1_KO_s100 + 1.30439 w_k1_MAT_s100
                                 + 0.931885 w_k1_MSFT_s100
                                 + 1.05338 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.46506 w_k1_ADM_s100
                                 + 1.49482 w_k1_AXP_s100
                                 + 1.49839 w_k1_DIS_s100
                                 + 1.32864 w_k1_FDX_s100
                                 + 1.06395 w_k1_IBM_s100
                                 + 0.974546 w_k1_INTC_s100
                                 + 1.28418 w_k1_KO_s100 + 1.08279 w_k1_MAT_s100
                                 + 1.14291 w_k1_MSFT_s100
                                 + 1.10279 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.35308 w_k1_ADM_s100
                                 + 0.92058 w_k1_AXP_s100
                                 + 0.976167 w_k1_DIS_s100
                                 + 1.03802 w_k1_FDX_s100
                                 + 0.738788 w_k1_IBM_s100
                                 + 0.991003 w_k1_INTC_s100
                                 + 0.87574 w_k1_KO_s100
                                 + 0.692812 w_k1_MAT_s100
                                 + 0.964226 w_k1_MSFT_s100
                                 + 0.900668 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.10375 w_k1_ADM_s101
                                 + 1.77851 w_k1_AXP_s101
                                 + 1.35546 w_k1_DIS_s101
                                 + 1.43306 w_k1_FDX_s101
                                 + 1.25065 w_k1_IBM_s101
                                 + 0.992255 w_k1_INTC_s101
                                 + 1.17408 w_k1_KO_s101 + 1.95883 w_k1_MAT_s101
                                 + 1.1467 w_k1_MSFT_s101 + 1.20413 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.02021 w_k1_ADM_s101
                                 + 0.815845 w_k1_AXP_s101
                                 + 0.951276 w_k1_DIS_s101
                                 + 0.968615 w_k1_FDX_s101
                                 + 0.781072 w_k1_IBM_s101
                                 + 1.08281 w_k1_INTC_s101
                                 + 0.825496 w_k1_KO_s101
                                 + 0.652041 w_k1_MAT_s101
                                 + 1.02887 w_k1_MSFT_s101
                                 + 0.961138 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.09183 w_k1_ADM_s101
                                 + 0.986813 w_k1_AXP_s101
                                 + 1.17831 w_k1_DIS_s101
                                 + 1.07007 w_k1_FDX_s101
                                 + 1.19001 w_k1_IBM_s101
                                 + 1.05398 w_k1_INTC_s101
                                 + 1.05019 w_k1_KO_s101 + 1.49741 w_k1_MAT_s101
                                 + 1.04951 w_k1_MSFT_s101
                                 + 0.966249 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.728401 w_k1_ADM_s101
                                 + 0.710372 w_k1_AXP_s101
                                 + 0.640714 w_k1_DIS_s101
                                 + 0.710726 w_k1_FDX_s101
                                 + 1.08974 w_k1_IBM_s101
                                 + 0.687274 w_k1_INTC_s101
                                 + 0.903037 w_k1_KO_s101 + 1.5692 w_k1_MAT_s101
                                 + 0.805148 w_k1_MSFT_s101
                                 + 0.841471 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.904195 w_k1_ADM_s101
                                 + 0.983869 w_k1_AXP_s101
                                 + 0.835547 w_k1_DIS_s101
                                 + 0.85463 w_k1_FDX_s101
                                 + 0.947367 w_k1_IBM_s101
                                 + 0.837814 w_k1_INTC_s101
                                 + 0.892631 w_k1_KO_s101
                                 + 1.09791 w_k1_MAT_s101
                                 + 0.941914 w_k1_MSFT_s101
                                 + 0.881394 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.02651 w_k1_ADM_s101
                                 + 1.11903 w_k1_AXP_s101
                                 + 0.763484 w_k1_DIS_s101
                                 + 1.02373 w_k1_FDX_s101
                                 + 0.677804 w_k1_IBM_s101
                                 + 0.989763 w_k1_INTC_s101
                                 + 0.689923 w_k1_KO_s101
                                 + 0.717815 w_k1_MAT_s101
                                 + 0.84196 w_k1_MSFT_s101 + 0.8521 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.905451 w_k1_ADM_s101
                                 + 1.38769 w_k1_AXP_s101
                                 + 0.985619 w_k1_DIS_s101
                                 + 1.00176 w_k1_FDX_s101
                                 + 1.43108 w_k1_IBM_s101
                                 + 1.00384 w_k1_INTC_s101
                                 + 1.20107 w_k1_KO_s101 + 2.30588 w_k1_MAT_s101
                                 + 0.937857 w_k1_MSFT_s101
                                 + 1.13273 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.07896 w_k1_ADM_s101
                                 + 1.34754 w_k1_AXP_s101
                                 + 1.44515 w_k1_DIS_s101
                                 + 1.29568 w_k1_FDX_s101
                                 + 0.991311 w_k1_IBM_s101
                                 + 0.946112 w_k1_INTC_s101
                                 + 1.18691 w_k1_KO_s101 + 1.26761 w_k1_MAT_s101
                                 + 1.24232 w_k1_MSFT_s101
                                 + 1.12579 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.02172 w_k1_ADM_s102
                                 + 1.13945 w_k1_AXP_s102
                                 + 1.17916 w_k1_DIS_s102
                                 + 1.09927 w_k1_FDX_s102
                                 + 0.970506 w_k1_IBM_s102
                                 + 0.989906 w_k1_INTC_s102
                                 + 1.02302 w_k1_KO_s102 + 1.19372 w_k1_MAT_s102
                                 + 1.07543 w_k1_MSFT_s102
                                 + 1.05356 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.968298 w_k1_ADM_s102
                                 + 0.985321 w_k1_AXP_s102
                                 + 0.895098 w_k1_DIS_s102
                                 + 0.840809 w_k1_FDX_s102
                                 + 0.842847 w_k1_IBM_s102
                                 + 0.919365 w_k1_INTC_s102
                                 + 1.03654 w_k1_KO_s102 + 1.08397 w_k1_MAT_s102
                                 + 0.929965 w_k1_MSFT_s102
                                 + 0.997416 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.33783 w_k1_ADM_s102
                                 + 1.69601 w_k1_AXP_s102
                                 + 1.31031 w_k1_DIS_s102
                                 + 1.28434 w_k1_FDX_s102
                                 + 1.01994 w_k1_IBM_s102
                                 + 1.20995 w_k1_INTC_s102
                                 + 1.09804 w_k1_KO_s102 + 1.24636 w_k1_MAT_s102
                                 + 1.08273 w_k1_MSFT_s102
                                 + 1.24031 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.5416 w_k1_ADM_s102
                                 + 1.60444 w_k1_AXP_s102
                                 + 1.61825 w_k1_DIS_s102
                                 + 1.33393 w_k1_FDX_s102
                                 + 1.24838 w_k1_IBM_s102
                                 + 1.28993 w_k1_INTC_s102
                                 + 1.61456 w_k1_KO_s102
                                 + 0.998471 w_k1_MAT_s102
                                 + 1.35311 w_k1_MSFT_s102
                                 + 1.37056 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.859763 w_k1_ADM_s102
                                 + 0.832706 w_k1_AXP_s102
                                 + 0.646966 w_k1_DIS_s102
                                 + 0.77187 w_k1_FDX_s102
                                 + 0.791829 w_k1_IBM_s102
                                 + 0.838966 w_k1_INTC_s102
                                 + 0.771213 w_k1_KO_s102
                                 + 0.745047 w_k1_MAT_s102
                                 + 0.773902 w_k1_MSFT_s102
                                 + 0.87994 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.49949 w_k1_ADM_s102
                                 + 0.89253 w_k1_AXP_s102
                                 + 0.966558 w_k1_DIS_s102
                                 + 1.11507 w_k1_FDX_s102
                                 + 0.649281 w_k1_IBM_s102
                                 + 0.884739 w_k1_INTC_s102
                                 + 0.992585 w_k1_KO_s102
                                 + 0.749785 w_k1_MAT_s102
                                 + 1.04848 w_k1_MSFT_s102
                                 + 1.13272 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.42328 w_k1_ADM_s102
                                 + 1.45185 w_k1_AXP_s102
                                 + 0.944954 w_k1_DIS_s102
                                 + 1.14375 w_k1_FDX_s102
                                 + 0.839052 w_k1_IBM_s102
                                 + 0.977095 w_k1_INTC_s102
                                 + 0.990501 w_k1_KO_s102
                                 + 1.53152 w_k1_MAT_s102
                                 + 0.974011 w_k1_MSFT_s102
                                 + 1.14568 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.18884 w_k1_ADM_s102
                                 + 1.20389 w_k1_AXP_s102
                                 + 0.986175 w_k1_DIS_s102
                                 + 0.949591 w_k1_FDX_s102
                                 + 1.26214 w_k1_IBM_s102
                                 + 0.822394 w_k1_INTC_s102
                                 + 1.04666 w_k1_KO_s102 + 1.57488 w_k1_MAT_s102
                                 + 0.873735 w_k1_MSFT_s102
                                 + 0.959167 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.5013 w_k1_ADM_s103
                                 + 1.89537 w_k1_AXP_s103
                                 + 2.29533 w_k1_DIS_s103
                                 + 1.63032 w_k1_FDX_s103
                                 + 1.58119 w_k1_IBM_s103
                                 + 1.88551 w_k1_INTC_s103
                                 + 1.80539 w_k1_KO_s103 + 1.48739 w_k1_MAT_s103
                                 + 1.8879 w_k1_MSFT_s103 + 1.31196 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.86104 w_k1_ADM_s103
                                 + 1.19786 w_k1_AXP_s103
                                 + 1.14022 w_k1_DIS_s103
                                 + 0.997866 w_k1_FDX_s103
                                 + 1.67902 w_k1_IBM_s103
                                 + 0.984108 w_k1_INTC_s103
                                 + 1.32643 w_k1_KO_s103 + 1.15324 w_k1_MAT_s103
                                 + 0.88808 w_k1_MSFT_s103 + 1.0831 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.796707 w_k1_ADM_s103
                                 + 0.820236 w_k1_AXP_s103
                                 + 0.793405 w_k1_DIS_s103
                                 + 0.832625 w_k1_FDX_s103
                                 + 0.852495 w_k1_IBM_s103
                                 + 0.920058 w_k1_INTC_s103
                                 + 0.90969 w_k1_KO_s103
                                 + 0.784213 w_k1_MAT_s103
                                 + 1.08028 w_k1_MSFT_s103
                                 + 0.839298 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.929079 w_k1_ADM_s103
                                 + 0.86233 w_k1_AXP_s103
                                 + 0.901732 w_k1_DIS_s103
                                 + 0.935058 w_k1_FDX_s103
                                 + 0.846602 w_k1_IBM_s103
                                 + 0.917807 w_k1_INTC_s103
                                 + 0.943357 w_k1_KO_s103
                                 + 0.623151 w_k1_MAT_s103
                                 + 1.02178 w_k1_MSFT_s103
                                 + 0.848209 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.98123 w_k1_ADM_s103
                                 + 0.615004 w_k1_AXP_s103
                                 + 0.737126 w_k1_DIS_s103
                                 + 0.795684 w_k1_FDX_s103
                                 + 0.590856 w_k1_IBM_s103
                                 + 0.710675 w_k1_INTC_s103
                                 + 0.754469 w_k1_KO_s103
                                 + 0.498696 w_k1_MAT_s103
                                 + 0.95547 w_k1_MSFT_s103
                                 + 0.818698 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.30493 w_k1_ADM_s103
                                 + 1.28758 w_k1_AXP_s103
                                 + 1.17633 w_k1_DIS_s103
                                 + 1.16083 w_k1_FDX_s103
                                 + 1.19599 w_k1_IBM_s103
                                 + 1.06976 w_k1_INTC_s103
                                 + 1.20457 w_k1_KO_s103 + 1.29731 w_k1_MAT_s103
                                 + 1.32492 w_k1_MSFT_s103
                                 + 0.976836 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.22681 w_k1_ADM_s103
                                 + 1.07434 w_k1_AXP_s103
                                 + 1.32515 w_k1_DIS_s103
                                 + 1.27519 w_k1_FDX_s103
                                 + 0.718326 w_k1_IBM_s103
                                 + 1.15107 w_k1_INTC_s103
                                 + 1.05309 w_k1_KO_s103
                                 + 0.556312 w_k1_MAT_s103
                                 + 1.37271 w_k1_MSFT_s103
                                 + 0.998952 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.02011 w_k1_ADM_s103
                                 + 0.873177 w_k1_AXP_s103
                                 + 0.72854 w_k1_DIS_s103
                                 + 0.78687 w_k1_FDX_s103
                                 + 1.03446 w_k1_IBM_s103
                                 + 0.846037 w_k1_INTC_s103
                                 + 0.977556 w_k1_KO_s103 + 0.7841 w_k1_MAT_s103
                                 + 0.951003 w_k1_MSFT_s103
                                 + 0.901414 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.890407 w_k1_ADM_s104
                                 + 0.712226 w_k1_AXP_s104
                                 + 0.779727 w_k1_DIS_s104
                                 + 0.831729 w_k1_FDX_s104
                                 + 1.2688 w_k1_IBM_s104
                                 + 0.70754 w_k1_INTC_s104
                                 + 1.04539 w_k1_KO_s104 + 1.23471 w_k1_MAT_s104
                                 + 0.8725 w_k1_MSFT_s104
                                 + 0.840522 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.04001 w_k1_ADM_s104
                                 + 0.853626 w_k1_AXP_s104
                                 + 1.40263 w_k1_DIS_s104
                                 + 1.11557 w_k1_FDX_s104
                                 + 0.909825 w_k1_IBM_s104
                                 + 0.953174 w_k1_INTC_s104
                                 + 1.0505 w_k1_KO_s104 + 0.6893 w_k1_MAT_s104
                                 + 1.08149 w_k1_MSFT_s104
                                 + 0.924289 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.98676 w_k1_ADM_s104
                                 + 0.842983 w_k1_AXP_s104
                                 + 0.72956 w_k1_DIS_s104
                                 + 0.855876 w_k1_FDX_s104
                                 + 0.809821 w_k1_IBM_s104
                                 + 0.863121 w_k1_INTC_s104
                                 + 0.755478 w_k1_KO_s104
                                 + 0.688359 w_k1_MAT_s104
                                 + 0.967509 w_k1_MSFT_s104
                                 + 0.699047 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.936502 w_k1_ADM_s104
                                 + 1.14191 w_k1_AXP_s104
                                 + 0.92799 w_k1_DIS_s104
                                 + 0.882625 w_k1_FDX_s104
                                 + 1.16484 w_k1_IBM_s104
                                 + 0.965901 w_k1_INTC_s104
                                 + 1.0313 w_k1_KO_s104 + 1.37634 w_k1_MAT_s104
                                 + 0.939499 w_k1_MSFT_s104
                                 + 0.871175 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.27741 w_k1_ADM_s104
                                 + 1.19093 w_k1_AXP_s104
                                 + 1.12883 w_k1_DIS_s104
                                 + 1.16285 w_k1_FDX_s104
                                 + 1.01081 w_k1_IBM_s104
                                 + 1.01187 w_k1_INTC_s104
                                 + 0.998845 w_k1_KO_s104
                                 + 1.18579 w_k1_MAT_s104
                                 + 0.956132 w_k1_MSFT_s104
                                 + 0.989012 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.23728 w_k1_ADM_s104
                                 + 0.829458 w_k1_AXP_s104
                                 + 0.864885 w_k1_DIS_s104
                                 + 1.01602 w_k1_FDX_s104
                                 + 1.13743 w_k1_IBM_s104
                                 + 1.01858 w_k1_INTC_s104
                                 + 1.17864 w_k1_KO_s104 + 1.13634 w_k1_MAT_s104
                                 + 0.959527 w_k1_MSFT_s104
                                 + 0.942303 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.807932 w_k1_ADM_s104
                                 + 0.917896 w_k1_AXP_s104
                                 + 0.871751 w_k1_DIS_s104
                                 + 0.812398 w_k1_FDX_s104
                                 + 1.08642 w_k1_IBM_s104
                                 + 0.870295 w_k1_INTC_s104
                                 + 0.855028 w_k1_KO_s104
                                 + 0.87248 w_k1_MAT_s104
                                 + 0.808455 w_k1_MSFT_s104
                                 + 0.845395 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.4223 w_k1_ADM_s104
                                 + 1.40356 w_k1_AXP_s104
                                 + 1.41329 w_k1_DIS_s104
                                 + 1.22254 w_k1_FDX_s104
                                 + 1.59806 w_k1_IBM_s104
                                 + 0.966758 w_k1_INTC_s104
                                 + 1.29811 w_k1_KO_s104 + 1.3884 w_k1_MAT_s104
                                 + 1.02972 w_k1_MSFT_s104
                                 + 1.11494 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.76315 w_k1_ADM_s105
                                 + 0.783205 w_k1_AXP_s105
                                 + 0.798835 w_k1_DIS_s105
                                 + 0.850991 w_k1_FDX_s105
                                 + 0.825474 w_k1_IBM_s105
                                 + 0.985539 w_k1_INTC_s105
                                 + 0.893847 w_k1_KO_s105
                                 + 0.612933 w_k1_MAT_s105
                                 + 0.869263 w_k1_MSFT_s105
                                 + 0.925241 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.37173 w_k1_ADM_s105
                                 + 1.32731 w_k1_AXP_s105
                                 + 1.58948 w_k1_DIS_s105
                                 + 1.38718 w_k1_FDX_s105 + 1.2865 w_k1_IBM_s105
                                 + 1.14589 w_k1_INTC_s105
                                 + 1.16661 w_k1_KO_s105 + 1.40031 w_k1_MAT_s105
                                 + 1.26117 w_k1_MSFT_s105
                                 + 1.22517 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.999848 w_k1_ADM_s105
                                 + 1.0585 w_k1_AXP_s105 + 1.10717 w_k1_DIS_s105
                                 + 0.938962 w_k1_FDX_s105
                                 + 1.19625 w_k1_IBM_s105
                                 + 0.806624 w_k1_INTC_s105
                                 + 1.02628 w_k1_KO_s105 + 1.47688 w_k1_MAT_s105
                                 + 0.84343 w_k1_MSFT_s105
                                 + 1.16932 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.29276 w_k1_ADM_s105
                                 + 1.22561 w_k1_AXP_s105 + 1.3768 w_k1_DIS_s105
                                 + 0.995522 w_k1_FDX_s105
                                 + 1.98479 w_k1_IBM_s105
                                 + 1.11673 w_k1_INTC_s105 + 1.4021 w_k1_KO_s105
                                 + 1.92427 w_k1_MAT_s105
                                 + 1.18199 w_k1_MSFT_s105
                                 + 1.21308 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.820352 w_k1_ADM_s105
                                 + 1.17977 w_k1_AXP_s105 + 1.0223 w_k1_DIS_s105
                                 + 0.964776 w_k1_FDX_s105
                                 + 1.3606 w_k1_IBM_s105
                                 + 0.901275 w_k1_INTC_s105
                                 + 1.08052 w_k1_KO_s105 + 1.61882 w_k1_MAT_s105
                                 + 0.845571 w_k1_MSFT_s105
                                 + 1.05577 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.01615 w_k1_ADM_s105
                                 + 0.993085 w_k1_AXP_s105
                                 + 1.02331 w_k1_DIS_s105 + 1.0724 w_k1_FDX_s105
                                 + 0.877199 w_k1_IBM_s105
                                 + 1.1694 w_k1_INTC_s105
                                 + 0.997226 w_k1_KO_s105
                                 + 0.970079 w_k1_MAT_s105
                                 + 0.992303 w_k1_MSFT_s105
                                 + 1.02951 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.19157 w_k1_ADM_s105
                                 + 0.969335 w_k1_AXP_s105
                                 + 1.14849 w_k1_DIS_s105
                                 + 1.28586 w_k1_FDX_s105
                                 + 0.634287 w_k1_IBM_s105
                                 + 0.925945 w_k1_INTC_s105
                                 + 0.888623 w_k1_KO_s105
                                 + 0.913985 w_k1_MAT_s105
                                 + 1.14826 w_k1_MSFT_s105
                                 + 1.14955 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.35465 w_k1_ADM_s105
                                 + 1.81822 w_k1_AXP_s105
                                 + 1.23266 w_k1_DIS_s105
                                 + 1.33326 w_k1_FDX_s105
                                 + 1.02452 w_k1_IBM_s105
                                 + 1.14782 w_k1_INTC_s105
                                 + 0.960083 w_k1_KO_s105 + 1.4103 w_k1_MAT_s105
                                 + 1.113 w_k1_MSFT_s105 + 1.19024 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.08935 w_k1_ADM_s106
                                 + 1.16781 w_k1_AXP_s106
                                 + 1.56433 w_k1_DIS_s106
                                 + 1.40979 w_k1_FDX_s106
                                 + 1.22503 w_k1_IBM_s106
                                 + 1.37682 w_k1_INTC_s106
                                 + 1.54813 w_k1_KO_s106 + 1.28592 w_k1_MAT_s106
                                 + 1.60481 w_k1_MSFT_s106
                                 + 1.17103 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.875919 w_k1_ADM_s106
                                 + 0.847619 w_k1_AXP_s106
                                 + 0.800406 w_k1_DIS_s106
                                 + 0.850024 w_k1_FDX_s106
                                 + 0.988676 w_k1_IBM_s106
                                 + 0.83929 w_k1_INTC_s106
                                 + 0.922993 w_k1_KO_s106
                                 + 1.22767 w_k1_MAT_s106
                                 + 0.943938 w_k1_MSFT_s106
                                 + 0.854512 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.0031 w_k1_ADM_s106
                                 + 1.16802 w_k1_AXP_s106
                                 + 1.15283 w_k1_DIS_s106
                                 + 1.08702 w_k1_FDX_s106
                                 + 0.836966 w_k1_IBM_s106
                                 + 1.00332 w_k1_INTC_s106
                                 + 1.02961 w_k1_KO_s106
                                 + 0.945423 w_k1_MAT_s106
                                 + 1.13052 w_k1_MSFT_s106
                                 + 0.989138 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.00144 w_k1_ADM_s106
                                 + 1.70955 w_k1_AXP_s106
                                 + 1.29867 w_k1_DIS_s106
                                 + 1.20908 w_k1_FDX_s106
                                 + 1.11421 w_k1_IBM_s106
                                 + 1.40846 w_k1_INTC_s106 + 1.0693 w_k1_KO_s106
                                 + 1.33573 w_k1_MAT_s106
                                 + 1.21048 w_k1_MSFT_s106
                                 + 1.06778 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.13542 w_k1_ADM_s106
                                 + 1.23001 w_k1_AXP_s106
                                 + 1.46853 w_k1_DIS_s106
                                 + 1.23293 w_k1_FDX_s106
                                 + 0.870742 w_k1_IBM_s106
                                 + 1.29825 w_k1_INTC_s106
                                 + 1.10408 w_k1_KO_s106 + 1.02577 w_k1_MAT_s106
                                 + 1.33694 w_k1_MSFT_s106
                                 + 1.10836 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.0441 w_k1_ADM_s106
                                 + 0.938704 w_k1_AXP_s106
                                 + 0.893932 w_k1_DIS_s106
                                 + 0.869342 w_k1_FDX_s106
                                 + 1.30068 w_k1_IBM_s106
                                 + 0.866591 w_k1_INTC_s106
                                 + 1.11005 w_k1_KO_s106 + 1.24687 w_k1_MAT_s106
                                 + 0.997451 w_k1_MSFT_s106
                                 + 0.971835 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.07299 w_k1_ADM_s106
                                 + 1.31934 w_k1_AXP_s106
                                 + 1.19568 w_k1_DIS_s106
                                 + 1.09305 w_k1_FDX_s106
                                 + 1.70535 w_k1_IBM_s106
                                 + 1.1088 w_k1_INTC_s106 + 1.31456 w_k1_KO_s106
                                 + 1.83682 w_k1_MAT_s106
                                 + 1.19535 w_k1_MSFT_s106
                                 + 1.17797 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.889917 w_k1_ADM_s106
                                 + 0.859877 w_k1_AXP_s106
                                 + 0.737131 w_k1_DIS_s106
                                 + 0.866662 w_k1_FDX_s106
                                 + 0.680572 w_k1_IBM_s106
                                 + 0.807104 w_k1_INTC_s106
                                 + 0.761456 w_k1_KO_s106
                                 + 0.723932 w_k1_MAT_s106
                                 + 0.90236 w_k1_MSFT_s106
                                 + 0.872271 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.36794 w_k1_ADM_s107
                                 + 1.69456 w_k1_AXP_s107
                                 + 1.54834 w_k1_DIS_s107
                                 + 1.42733 w_k1_FDX_s107
                                 + 0.979175 w_k1_IBM_s107
                                 + 1.38481 w_k1_INTC_s107
                                 + 1.05136 w_k1_KO_s107 + 1.29297 w_k1_MAT_s107
                                 + 1.27689 w_k1_MSFT_s107
                                 + 1.38198 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.03472 w_k1_ADM_s107
                                 + 0.912782 w_k1_AXP_s107
                                 + 1.18863 w_k1_DIS_s107
                                 + 1.00097 w_k1_FDX_s107
                                 + 0.948737 w_k1_IBM_s107
                                 + 0.910392 w_k1_INTC_s107
                                 + 1.12995 w_k1_KO_s107 + 1.11948 w_k1_MAT_s107
                                 + 1.22514 w_k1_MSFT_s107
                                 + 1.11384 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.924258 w_k1_ADM_s107
                                 + 0.937849 w_k1_AXP_s107
                                 + 0.856827 w_k1_DIS_s107
                                 + 0.819509 w_k1_FDX_s107
                                 + 1.10867 w_k1_IBM_s107
                                 + 0.85152 w_k1_INTC_s107
                                 + 0.846343 w_k1_KO_s107
                                 + 1.14036 w_k1_MAT_s107
                                 + 0.885443 w_k1_MSFT_s107
                                 + 1.05852 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.775492 w_k1_ADM_s107
                                 + 0.784607 w_k1_AXP_s107
                                 + 0.689793 w_k1_DIS_s107
                                 + 0.772799 w_k1_FDX_s107
                                 + 0.836454 w_k1_IBM_s107
                                 + 0.85346 w_k1_INTC_s107
                                 + 0.677167 w_k1_KO_s107
                                 + 0.787839 w_k1_MAT_s107
                                 + 0.810148 w_k1_MSFT_s107
                                 + 0.972588 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.879124 w_k1_ADM_s107
                                 + 1.15812 w_k1_AXP_s107
                                 + 0.911125 w_k1_DIS_s107
                                 + 1.03605 w_k1_FDX_s107
                                 + 0.966246 w_k1_IBM_s107
                                 + 0.920327 w_k1_INTC_s107
                                 + 0.943181 w_k1_KO_s107
                                 + 1.30177 w_k1_MAT_s107
                                 + 1.06211 w_k1_MSFT_s107
                                 + 1.12701 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.972158 w_k1_ADM_s107
                                 + 0.96314 w_k1_AXP_s107
                                 + 0.83252 w_k1_DIS_s107
                                 + 0.979975 w_k1_FDX_s107
                                 + 0.729589 w_k1_IBM_s107
                                 + 0.89208 w_k1_INTC_s107
                                 + 0.651286 w_k1_KO_s107
                                 + 0.530659 w_k1_MAT_s107
                                 + 0.984932 w_k1_MSFT_s107
                                 + 0.925124 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.16586 w_k1_ADM_s107
                                 + 1.23503 w_k1_AXP_s107
                                 + 1.29701 w_k1_DIS_s107
                                 + 1.10403 w_k1_FDX_s107
                                 + 1.58285 w_k1_IBM_s107
                                 + 0.971832 w_k1_INTC_s107
                                 + 1.18897 w_k1_KO_s107 + 1.19405 w_k1_MAT_s107
                                 + 1.1154 w_k1_MSFT_s107 + 1.22454 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.36807 w_k1_ADM_s107
                                 + 1.10079 w_k1_AXP_s107
                                 + 1.55307 w_k1_DIS_s107 + 1.2023 w_k1_FDX_s107
                                 + 1.19608 w_k1_IBM_s107
                                 + 1.08862 w_k1_INTC_s107
                                 + 1.22642 w_k1_KO_s107 + 1.03937 w_k1_MAT_s107
                                 + 1.15099 w_k1_MSFT_s107
                                 + 1.35808 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.15805 w_k1_ADM_s110
                                 + 0.943009 w_k1_AXP_s110
                                 + 0.951381 w_k1_DIS_s110
                                 + 0.931068 w_k1_FDX_s110
                                 + 1.27624 w_k1_IBM_s110
                                 + 1.02031 w_k1_INTC_s110 + 1.1746 w_k1_KO_s110
                                 + 1.30684 w_k1_MAT_s110
                                 + 1.05717 w_k1_MSFT_s110
                                 + 1.16884 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.00759 w_k1_ADM_s110
                                 + 0.825804 w_k1_AXP_s110
                                 + 0.876187 w_k1_DIS_s110
                                 + 1.0178 w_k1_FDX_s110
                                 + 0.666875 w_k1_IBM_s110
                                 + 0.946114 w_k1_INTC_s110
                                 + 0.832691 w_k1_KO_s110
                                 + 0.916691 w_k1_MAT_s110
                                 + 0.883335 w_k1_MSFT_s110
                                 + 1.04421 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.17645 w_k1_ADM_s110
                                 + 1.26745 w_k1_AXP_s110
                                 + 1.29083 w_k1_DIS_s110 + 1.1472 w_k1_FDX_s110
                                 + 1.09377 w_k1_IBM_s110
                                 + 1.22797 w_k1_INTC_s110
                                 + 1.32415 w_k1_KO_s110 + 1.09828 w_k1_MAT_s110
                                 + 1.21829 w_k1_MSFT_s110
                                 + 1.14238 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.837811 w_k1_ADM_s110
                                 + 0.70096 w_k1_AXP_s110
                                 + 0.707391 w_k1_DIS_s110
                                 + 0.685772 w_k1_FDX_s110
                                 + 0.827132 w_k1_IBM_s110
                                 + 0.718245 w_k1_INTC_s110
                                 + 0.763777 w_k1_KO_s110
                                 + 0.714443 w_k1_MAT_s110
                                 + 0.835598 w_k1_MSFT_s110
                                 + 0.822754 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.888909 w_k1_ADM_s110
                                 + 0.790925 w_k1_AXP_s110
                                 + 0.912424 w_k1_DIS_s110
                                 + 0.85354 w_k1_FDX_s110
                                 + 1.00679 w_k1_IBM_s110
                                 + 1.00878 w_k1_INTC_s110
                                 + 1.03404 w_k1_KO_s110 + 0.86537 w_k1_MAT_s110
                                 + 0.900971 w_k1_MSFT_s110
                                 + 0.959299 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.07493 w_k1_ADM_s110
                                 + 0.8758 w_k1_AXP_s110 + 1.18328 w_k1_DIS_s110
                                 + 1.08269 w_k1_FDX_s110
                                 + 0.895229 w_k1_IBM_s110
                                 + 1.23488 w_k1_INTC_s110
                                 + 1.13857 w_k1_KO_s110
                                 + 0.882726 w_k1_MAT_s110
                                 + 1.24174 w_k1_MSFT_s110
                                 + 1.16574 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.13401 w_k1_ADM_s110
                                 + 1.0577 w_k1_AXP_s110 + 1.22125 w_k1_DIS_s110
                                 + 0.981316 w_k1_FDX_s110
                                 + 1.80666 w_k1_IBM_s110
                                 + 1.13872 w_k1_INTC_s110
                                 + 1.48166 w_k1_KO_s110 + 2.08772 w_k1_MAT_s110
                                 + 0.990348 w_k1_MSFT_s110
                                 + 1.13058 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.33985 w_k1_ADM_s110
                                 + 1.43788 w_k1_AXP_s110
                                 + 1.74601 w_k1_DIS_s110
                                 + 1.34077 w_k1_FDX_s110
                                 + 1.23554 w_k1_IBM_s110
                                 + 1.37989 w_k1_INTC_s110
                                 + 1.50748 w_k1_KO_s110 + 1.16975 w_k1_MAT_s110
                                 + 1.41911 w_k1_MSFT_s110
                                 + 1.45242 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.29735 w_k1_ADM_s111
                                 + 1.51782 w_k1_AXP_s111
                                 + 1.53459 w_k1_DIS_s111
                                 + 1.22465 w_k1_FDX_s111
                                 + 1.49884 w_k1_IBM_s111
                                 + 1.48842 w_k1_INTC_s111 + 1.3741 w_k1_KO_s111
                                 + 1.28772 w_k1_MAT_s111
                                 + 1.27488 w_k1_MSFT_s111 + 1.1523 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.838171 w_k1_ADM_s111
                                 + 0.850321 w_k1_AXP_s111
                                 + 0.668275 w_k1_DIS_s111
                                 + 0.807508 w_k1_FDX_s111
                                 + 0.941589 w_k1_IBM_s111
                                 + 0.873421 w_k1_INTC_s111
                                 + 0.826629 w_k1_KO_s111
                                 + 0.873952 w_k1_MAT_s111
                                 + 0.779593 w_k1_MSFT_s111
                                 + 0.783369 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.906641 w_k1_ADM_s111
                                 + 0.967307 w_k1_AXP_s111
                                 + 1.01135 w_k1_DIS_s111
                                 + 0.997754 w_k1_FDX_s111
                                 + 0.670853 w_k1_IBM_s111
                                 + 1.07031 w_k1_INTC_s111
                                 + 0.835691 w_k1_KO_s111
                                 + 0.59721 w_k1_MAT_s111
                                 + 0.951768 w_k1_MSFT_s111
                                 + 0.886579 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.22769 w_k1_ADM_s111
                                 + 1.46213 w_k1_AXP_s111
                                 + 1.01494 w_k1_DIS_s111
                                 + 1.18179 w_k1_FDX_s111
                                 + 1.08534 w_k1_IBM_s111
                                 + 1.17779 w_k1_INTC_s111
                                 + 0.910351 w_k1_KO_s111
                                 + 0.98777 w_k1_MAT_s111
                                 + 1.00218 w_k1_MSFT_s111
                                 + 1.03908 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.17157 w_k1_ADM_s111
                                 + 1.43982 w_k1_AXP_s111
                                 + 1.57052 w_k1_DIS_s111
                                 + 1.40837 w_k1_FDX_s111 + 1.1276 w_k1_IBM_s111
                                 + 1.47139 w_k1_INTC_s111
                                 + 1.26832 w_k1_KO_s111 + 1.15856 w_k1_MAT_s111
                                 + 1.38634 w_k1_MSFT_s111
                                 + 1.17778 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.9394 w_k1_ADM_s111
                                 + 0.884467 w_k1_AXP_s111
                                 + 0.982422 w_k1_DIS_s111
                                 + 0.930755 w_k1_FDX_s111
                                 + 0.993576 w_k1_IBM_s111
                                 + 1.13737 w_k1_INTC_s111
                                 + 0.94122 w_k1_KO_s111
                                 + 0.743014 w_k1_MAT_s111
                                 + 0.932327 w_k1_MSFT_s111
                                 + 0.896888 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.660216 w_k1_ADM_s111
                                 + 0.802071 w_k1_AXP_s111
                                 + 0.936347 w_k1_DIS_s111
                                 + 0.811737 w_k1_FDX_s111
                                 + 1.13144 w_k1_IBM_s111
                                 + 1.12108 w_k1_INTC_s111
                                 + 1.00913 w_k1_KO_s111 + 1.03729 w_k1_MAT_s111
                                 + 1.05253 w_k1_MSFT_s111
                                 + 0.832518 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.18963 w_k1_ADM_s111
                                 + 1.00366 w_k1_AXP_s111
                                 + 1.27191 w_k1_DIS_s111
                                 + 1.08984 w_k1_FDX_s111 + 1.1243 w_k1_IBM_s111
                                 + 1.19577 w_k1_INTC_s111
                                 + 1.35321 w_k1_KO_s111 + 1.0685 w_k1_MAT_s111
                                 + 1.10985 w_k1_MSFT_s111
                                 + 1.05093 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.917388 w_k1_ADM_s112
                                 + 0.665974 w_k1_AXP_s112
                                 + 0.889518 w_k1_DIS_s112
                                 + 0.91505 w_k1_FDX_s112
                                 + 0.734226 w_k1_IBM_s112
                                 + 0.840804 w_k1_INTC_s112
                                 + 0.8735 w_k1_KO_s112 + 0.628683 w_k1_MAT_s112
                                 + 1.07798 w_k1_MSFT_s112
                                 + 0.831136 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.1317 w_k1_ADM_s112
                                 + 1.30205 w_k1_AXP_s112
                                 + 0.942376 w_k1_DIS_s112
                                 + 1.07487 w_k1_FDX_s112
                                 + 1.07536 w_k1_IBM_s112
                                 + 1.15551 w_k1_INTC_s112
                                 + 1.11623 w_k1_KO_s112 + 1.21434 w_k1_MAT_s112
                                 + 1.15042 w_k1_MSFT_s112
                                 + 1.02031 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.07399 w_k1_ADM_s112
                                 + 0.853388 w_k1_AXP_s112
                                 + 0.819398 w_k1_DIS_s112
                                 + 0.976049 w_k1_FDX_s112
                                 + 0.848797 w_k1_IBM_s112
                                 + 0.968195 w_k1_INTC_s112
                                 + 0.850914 w_k1_KO_s112
                                 + 0.929735 w_k1_MAT_s112
                                 + 1.01364 w_k1_MSFT_s112
                                 + 0.944823 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.844868 w_k1_ADM_s112
                                 + 0.929582 w_k1_AXP_s112
                                 + 1.0347 w_k1_DIS_s112 + 0.94114 w_k1_FDX_s112
                                 + 1.23176 w_k1_IBM_s112
                                 + 0.956772 w_k1_INTC_s112
                                 + 1.32354 w_k1_KO_s112 + 1.56864 w_k1_MAT_s112
                                 + 1.08471 w_k1_MSFT_s112
                                 + 1.12829 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.702474 w_k1_ADM_s112
                                 + 0.804469 w_k1_AXP_s112
                                 + 0.693611 w_k1_DIS_s112
                                 + 0.746289 w_k1_FDX_s112
                                 + 0.854499 w_k1_IBM_s112
                                 + 0.799857 w_k1_INTC_s112
                                 + 0.729308 w_k1_KO_s112
                                 + 0.946861 w_k1_MAT_s112
                                 + 0.822836 w_k1_MSFT_s112
                                 + 0.817686 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.939732 w_k1_ADM_s112
                                 + 1.20312 w_k1_AXP_s112
                                 + 0.834322 w_k1_DIS_s112
                                 + 1.01537 w_k1_FDX_s112
                                 + 0.686026 w_k1_IBM_s112
                                 + 0.886189 w_k1_INTC_s112
                                 + 0.763839 w_k1_KO_s112
                                 + 0.955308 w_k1_MAT_s112
                                 + 1.00168 w_k1_MSFT_s112
                                 + 0.885535 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.943701 w_k1_ADM_s112
                                 + 1.18745 w_k1_AXP_s112
                                 + 1.03438 w_k1_DIS_s112
                                 + 0.892018 w_k1_FDX_s112
                                 + 1.22105 w_k1_IBM_s112
                                 + 1.0875 w_k1_INTC_s112 + 1.07267 w_k1_KO_s112
                                 + 1.23962 w_k1_MAT_s112
                                 + 1.04358 w_k1_MSFT_s112
                                 + 1.01437 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.24057 w_k1_ADM_s112
                                 + 1.28987 w_k1_AXP_s112
                                 + 1.45523 w_k1_DIS_s112
                                 + 1.25841 w_k1_FDX_s112
                                 + 1.17795 w_k1_IBM_s112
                                 + 1.10881 w_k1_INTC_s112
                                 + 1.13664 w_k1_KO_s112 + 1.22291 w_k1_MAT_s112
                                 + 1.40773 w_k1_MSFT_s112
                                 + 1.16603 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.98415 w_k1_ADM_s113
                                 + 0.825044 w_k1_AXP_s113
                                 + 0.873834 w_k1_DIS_s113
                                 + 0.853332 w_k1_FDX_s113
                                 + 1.00552 w_k1_IBM_s113
                                 + 0.828176 w_k1_INTC_s113
                                 + 0.870331 w_k1_KO_s113
                                 + 0.983165 w_k1_MAT_s113
                                 + 0.843668 w_k1_MSFT_s113
                                 + 0.943165 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.04959 w_k1_ADM_s113
                                 + 0.973174 w_k1_AXP_s113
                                 + 1.02648 w_k1_DIS_s113
                                 + 0.949475 w_k1_FDX_s113
                                 + 1.30522 w_k1_IBM_s113
                                 + 1.03436 w_k1_INTC_s113 + 1.0377 w_k1_KO_s113
                                 + 1.31848 w_k1_MAT_s113
                                 + 0.88951 w_k1_MSFT_s113
                                 + 1.01346 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.778892 w_k1_ADM_s113
                                 + 0.809608 w_k1_AXP_s113
                                 + 0.840604 w_k1_DIS_s113
                                 + 0.867126 w_k1_FDX_s113
                                 + 0.811333 w_k1_IBM_s113
                                 + 0.914461 w_k1_INTC_s113
                                 + 0.757094 w_k1_KO_s113
                                 + 0.859104 w_k1_MAT_s113
                                 + 0.916402 w_k1_MSFT_s113
                                 + 0.919971 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.682004 w_k1_ADM_s113
                                 + 0.669349 w_k1_AXP_s113
                                 + 0.699917 w_k1_DIS_s113
                                 + 0.794894 w_k1_FDX_s113
                                 + 0.635329 w_k1_IBM_s113
                                 + 0.857446 w_k1_INTC_s113
                                 + 0.579107 w_k1_KO_s113
                                 + 0.560429 w_k1_MAT_s113
                                 + 0.798028 w_k1_MSFT_s113
                                 + 0.751869 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.15814 w_k1_ADM_s113
                                 + 1.26557 w_k1_AXP_s113 + 1.265 w_k1_DIS_s113
                                 + 1.05018 w_k1_FDX_s113
                                 + 1.94227 w_k1_IBM_s113
                                 + 0.937089 w_k1_INTC_s113
                                 + 1.17317 w_k1_KO_s113 + 1.92281 w_k1_MAT_s113
                                 + 1.04441 w_k1_MSFT_s113
                                 + 1.13043 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.956025 w_k1_ADM_s113
                                 + 1.11643 w_k1_AXP_s113
                                 + 1.17308 w_k1_DIS_s113
                                 + 1.05717 w_k1_FDX_s113
                                 + 1.10155 w_k1_IBM_s113
                                 + 1.08924 w_k1_INTC_s113
                                 + 1.12409 w_k1_KO_s113 + 1.29814 w_k1_MAT_s113
                                 + 1.0802 w_k1_MSFT_s113 + 1.08694 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.09544 w_k1_ADM_s113
                                 + 1.39116 w_k1_AXP_s113
                                 + 1.48385 w_k1_DIS_s113 + 1.2881 w_k1_FDX_s113
                                 + 1.32035 w_k1_IBM_s113
                                 + 0.905099 w_k1_INTC_s113
                                 + 1.1607 w_k1_KO_s113 + 1.31626 w_k1_MAT_s113
                                 + 1.13842 w_k1_MSFT_s113
                                 + 1.26634 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.26019 w_k1_ADM_s113
                                 + 0.963043 w_k1_AXP_s113
                                 + 1.12759 w_k1_DIS_s113
                                 + 1.15649 w_k1_FDX_s113
                                 + 0.726923 w_k1_IBM_s113
                                 + 0.990656 w_k1_INTC_s113
                                 + 0.78202 w_k1_KO_s113
                                 + 0.863461 w_k1_MAT_s113
                                 + 1.11582 w_k1_MSFT_s113
                                 + 1.07057 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.32999 w_k1_ADM_s114
                                 + 1.38802 w_k1_AXP_s114
                                 + 1.81414 w_k1_DIS_s114
                                 + 1.46806 w_k1_FDX_s114
                                 + 1.05187 w_k1_IBM_s114
                                 + 1.26632 w_k1_INTC_s114
                                 + 1.24302 w_k1_KO_s114 + 1.23176 w_k1_MAT_s114
                                 + 1.43842 w_k1_MSFT_s114
                                 + 1.14637 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.971886 w_k1_ADM_s114
                                 + 0.663005 w_k1_AXP_s114
                                 + 1.18598 w_k1_DIS_s114
                                 + 0.844028 w_k1_FDX_s114
                                 + 1.29557 w_k1_IBM_s114
                                 + 0.880247 w_k1_INTC_s114
                                 + 1.37621 w_k1_KO_s114
                                 + 0.911586 w_k1_MAT_s114
                                 + 0.992926 w_k1_MSFT_s114
                                 + 1.0874 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.02211 w_k1_ADM_s114
                                 + 0.981378 w_k1_AXP_s114
                                 + 0.771153 w_k1_DIS_s114
                                 + 0.794082 w_k1_FDX_s114
                                 + 1.35259 w_k1_IBM_s114
                                 + 0.872454 w_k1_INTC_s114
                                 + 1.05615 w_k1_KO_s114 + 1.1115 w_k1_MAT_s114
                                 + 0.81968 w_k1_MSFT_s114
                                 + 0.971298 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.84918 w_k1_ADM_s114
                                 + 0.616642 w_k1_AXP_s114
                                 + 0.767802 w_k1_DIS_s114
                                 + 0.75221 w_k1_FDX_s114
                                 + 0.736054 w_k1_IBM_s114
                                 + 0.888292 w_k1_INTC_s114
                                 + 0.741245 w_k1_KO_s114
                                 + 0.457369 w_k1_MAT_s114
                                 + 0.786418 w_k1_MSFT_s114
                                 + 0.811177 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.14199 w_k1_ADM_s114
                                 + 1.01446 w_k1_AXP_s114
                                 + 1.16875 w_k1_DIS_s114
                                 + 0.991507 w_k1_FDX_s114
                                 + 1.14021 w_k1_IBM_s114
                                 + 0.941563 w_k1_INTC_s114
                                 + 1.13326 w_k1_KO_s114
                                 + 0.999924 w_k1_MAT_s114
                                 + 1.05692 w_k1_MSFT_s114
                                 + 1.00226 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.22958 w_k1_ADM_s114
                                 + 1.31581 w_k1_AXP_s114
                                 + 1.60115 w_k1_DIS_s114
                                 + 1.19536 w_k1_FDX_s114
                                 + 1.31029 w_k1_IBM_s114
                                 + 1.13651 w_k1_INTC_s114
                                 + 1.26147 w_k1_KO_s114 + 1.27417 w_k1_MAT_s114
                                 + 1.16831 w_k1_MSFT_s114 + 1.1141 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.30305 w_k1_ADM_s114
                                 + 1.18832 w_k1_AXP_s114
                                 + 1.10608 w_k1_DIS_s114 + 1.1513 w_k1_FDX_s114
                                 + 0.776493 w_k1_IBM_s114
                                 + 1.0222 w_k1_INTC_s114
                                 + 0.901228 w_k1_KO_s114
                                 + 0.754264 w_k1_MAT_s114
                                 + 1.01679 w_k1_MSFT_s114
                                 + 1.01439 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.05861 w_k1_ADM_s114
                                 + 0.826367 w_k1_AXP_s114
                                 + 1.32662 w_k1_DIS_s114
                                 + 1.08446 w_k1_FDX_s114
                                 + 0.706797 w_k1_IBM_s114
                                 + 0.883147 w_k1_INTC_s114
                                 + 0.992948 w_k1_KO_s114
                                 + 0.730216 w_k1_MAT_s114
                                 + 1.15378 w_k1_MSFT_s114
                                 + 1.01549 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.20766 w_k1_ADM_s115
                                 + 1.14601 w_k1_AXP_s115 + 1.1703 w_k1_DIS_s115
                                 + 1.09623 w_k1_FDX_s115
                                 + 1.38312 w_k1_IBM_s115
                                 + 1.10167 w_k1_INTC_s115
                                 + 1.36533 w_k1_KO_s115 + 1.32088 w_k1_MAT_s115
                                 + 1.2291 w_k1_MSFT_s115 + 1.04039 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.998332 w_k1_ADM_s115
                                 + 0.797569 w_k1_AXP_s115
                                 + 1.11157 w_k1_DIS_s115
                                 + 1.09899 w_k1_FDX_s115
                                 + 0.654483 w_k1_IBM_s115
                                 + 1.14508 w_k1_INTC_s115
                                 + 0.894464 w_k1_KO_s115
                                 + 0.44937 w_k1_MAT_s115
                                 + 1.19841 w_k1_MSFT_s115
                                 + 0.99911 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.31797 w_k1_ADM_s115
                                 + 1.34717 w_k1_AXP_s115
                                 + 1.40794 w_k1_DIS_s115
                                 + 1.33877 w_k1_FDX_s115
                                 + 0.892732 w_k1_IBM_s115
                                 + 1.11108 w_k1_INTC_s115 + 1.0445 w_k1_KO_s115
                                 + 1.24227 w_k1_MAT_s115
                                 + 1.27506 w_k1_MSFT_s115
                                 + 1.12956 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.08905 w_k1_ADM_s115
                                 + 0.9033 w_k1_AXP_s115
                                 + 0.812706 w_k1_DIS_s115
                                 + 0.940106 w_k1_FDX_s115
                                 + 0.674039 w_k1_IBM_s115
                                 + 0.842182 w_k1_INTC_s115
                                 + 0.902213 w_k1_KO_s115
                                 + 0.898966 w_k1_MAT_s115
                                 + 1.03394 w_k1_MSFT_s115
                                 + 0.946976 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.755366 w_k1_ADM_s115
                                 + 0.771466 w_k1_AXP_s115
                                 + 0.709763 w_k1_DIS_s115
                                 + 0.69341 w_k1_FDX_s115
                                 + 0.981603 w_k1_IBM_s115
                                 + 0.770517 w_k1_INTC_s115
                                 + 1.01296 w_k1_KO_s115 + 1.09569 w_k1_MAT_s115
                                 + 0.826958 w_k1_MSFT_s115
                                 + 0.825243 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.01235 w_k1_ADM_s115
                                 + 0.977568 w_k1_AXP_s115
                                 + 0.870972 w_k1_DIS_s115
                                 + 0.936381 w_k1_FDX_s115
                                 + 1.12412 w_k1_IBM_s115
                                 + 0.878799 w_k1_INTC_s115
                                 + 1.15093 w_k1_KO_s115 + 1.48304 w_k1_MAT_s115
                                 + 0.945928 w_k1_MSFT_s115
                                 + 0.944739 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.802477 w_k1_ADM_s115
                                 + 0.679008 w_k1_AXP_s115
                                 + 0.575714 w_k1_DIS_s115
                                 + 0.721825 w_k1_FDX_s115
                                 + 0.70611 w_k1_IBM_s115
                                 + 0.721958 w_k1_INTC_s115
                                 + 0.824069 w_k1_KO_s115
                                 + 0.700854 w_k1_MAT_s115
                                 + 0.760098 w_k1_MSFT_s115
                                 + 0.752276 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.19557 w_k1_ADM_s115
                                 + 1.00803 w_k1_AXP_s115
                                 + 0.835676 w_k1_DIS_s115
                                 + 0.916046 w_k1_FDX_s115
                                 + 0.937253 w_k1_IBM_s115
                                 + 0.827542 w_k1_INTC_s115
                                 + 0.934778 w_k1_KO_s115
                                 + 1.00262 w_k1_MAT_s115
                                 + 0.898269 w_k1_MSFT_s115
                                 + 0.916954 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.09642 w_k1_ADM_s116
                                 + 1.02045 w_k1_AXP_s116
                                 + 1.05539 w_k1_DIS_s116
                                 + 1.14974 w_k1_FDX_s116
                                 + 0.64066 w_k1_IBM_s116
                                 + 0.935721 w_k1_INTC_s116
                                 + 0.940003 w_k1_KO_s116
                                 + 0.596383 w_k1_MAT_s116
                                 + 1.10413 w_k1_MSFT_s116
                                 + 0.937283 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.676543 w_k1_ADM_s116
                                 + 0.678651 w_k1_AXP_s116
                                 + 0.741396 w_k1_DIS_s116
                                 + 0.786926 w_k1_FDX_s116
                                 + 0.803338 w_k1_IBM_s116
                                 + 0.882584 w_k1_INTC_s116
                                 + 0.903569 w_k1_KO_s116
                                 + 0.686552 w_k1_MAT_s116
                                 + 0.928962 w_k1_MSFT_s116
                                 + 0.855494 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.26062 w_k1_ADM_s116
                                 + 1.62961 w_k1_AXP_s116
                                 + 1.44215 w_k1_DIS_s116
                                 + 1.33251 w_k1_FDX_s116
                                 + 1.27924 w_k1_IBM_s116
                                 + 1.23221 w_k1_INTC_s116
                                 + 1.23843 w_k1_KO_s116 + 1.52149 w_k1_MAT_s116
                                 + 1.36732 w_k1_MSFT_s116
                                 + 1.23531 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.35732 w_k1_ADM_s116
                                 + 1.32968 w_k1_AXP_s116
                                 + 1.28736 w_k1_DIS_s116
                                 + 1.24368 w_k1_FDX_s116
                                 + 0.849383 w_k1_IBM_s116
                                 + 1.19052 w_k1_INTC_s116
                                 + 1.08624 w_k1_KO_s116
                                 + 0.932738 w_k1_MAT_s116
                                 + 1.1511 w_k1_MSFT_s116 + 1.03935 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.967996 w_k1_ADM_s116
                                 + 1.09422 w_k1_AXP_s116
                                 + 0.753011 w_k1_DIS_s116
                                 + 0.88029 w_k1_FDX_s116
                                 + 1.19202 w_k1_IBM_s116
                                 + 0.888698 w_k1_INTC_s116
                                 + 1.00523 w_k1_KO_s116 + 1.2775 w_k1_MAT_s116
                                 + 0.940528 w_k1_MSFT_s116
                                 + 0.998558 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.31951 w_k1_ADM_s116
                                 + 1.2279 w_k1_AXP_s116 + 1.06816 w_k1_DIS_s116
                                 + 1.01214 w_k1_FDX_s116
                                 + 1.73541 w_k1_IBM_s116
                                 + 1.01278 w_k1_INTC_s116
                                 + 1.20842 w_k1_KO_s116 + 1.81224 w_k1_MAT_s116
                                 + 1.118 w_k1_MSFT_s116 + 1.12905 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.05283 w_k1_ADM_s116
                                 + 1.13823 w_k1_AXP_s116
                                 + 0.777754 w_k1_DIS_s116
                                 + 0.962125 w_k1_FDX_s116
                                 + 0.840416 w_k1_IBM_s116
                                 + 0.982545 w_k1_INTC_s116
                                 + 0.921015 w_k1_KO_s116
                                 + 0.906349 w_k1_MAT_s116
                                 + 0.977271 w_k1_MSFT_s116
                                 + 0.958628 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.985223 w_k1_ADM_s116
                                 + 0.758524 w_k1_AXP_s116
                                 + 0.58814 w_k1_DIS_s116
                                 + 0.891438 w_k1_FDX_s116
                                 + 0.494415 w_k1_IBM_s116
                                 + 0.79858 w_k1_INTC_s116
                                 + 0.651956 w_k1_KO_s116
                                 + 0.337112 w_k1_MAT_s116
                                 + 0.819322 w_k1_MSFT_s116
                                 + 0.856647 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.02494 w_k1_ADM_s117
                                 + 0.855402 w_k1_AXP_s117
                                 + 1.15182 w_k1_DIS_s117
                                 + 1.02066 w_k1_FDX_s117
                                 + 1.13053 w_k1_IBM_s117
                                 + 1.08249 w_k1_INTC_s117 + 1.0805 w_k1_KO_s117
                                 + 0.744582 w_k1_MAT_s117
                                 + 1.14813 w_k1_MSFT_s117
                                 + 1.00497 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.894652 w_k1_ADM_s117
                                 + 0.601526 w_k1_AXP_s117
                                 + 0.655402 w_k1_DIS_s117
                                 + 0.758463 w_k1_FDX_s117
                                 + 0.590091 w_k1_IBM_s117
                                 + 0.759383 w_k1_INTC_s117
                                 + 0.701665 w_k1_KO_s117
                                 + 0.39058 w_k1_MAT_s117
                                 + 0.780992 w_k1_MSFT_s117
                                 + 0.794948 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.890796 w_k1_ADM_s117
                                 + 0.778091 w_k1_AXP_s117
                                 + 0.741441 w_k1_DIS_s117
                                 + 0.77474 w_k1_FDX_s117
                                 + 1.13767 w_k1_IBM_s117
                                 + 0.681619 w_k1_INTC_s117
                                 + 0.819473 w_k1_KO_s117
                                 + 0.874613 w_k1_MAT_s117
                                 + 0.78357 w_k1_MSFT_s117
                                 + 0.993685 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.875275 w_k1_ADM_s117
                                 + 1.29651 w_k1_AXP_s117
                                 + 1.07379 w_k1_DIS_s117
                                 + 0.940976 w_k1_FDX_s117
                                 + 1.37887 w_k1_IBM_s117
                                 + 0.979885 w_k1_INTC_s117
                                 + 1.15295 w_k1_KO_s117 + 1.56808 w_k1_MAT_s117
                                 + 1.06467 w_k1_MSFT_s117
                                 + 1.06953 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.15571 w_k1_ADM_s117
                                 + 1.1201 w_k1_AXP_s117
                                 + 0.786007 w_k1_DIS_s117
                                 + 0.916603 w_k1_FDX_s117
                                 + 1.15111 w_k1_IBM_s117
                                 + 0.974173 w_k1_INTC_s117
                                 + 0.85475 w_k1_KO_s117 + 1.27576 w_k1_MAT_s117
                                 + 0.905444 w_k1_MSFT_s117
                                 + 1.04883 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.886734 w_k1_ADM_s117
                                 + 0.800431 w_k1_AXP_s117
                                 + 0.766478 w_k1_DIS_s117
                                 + 0.884278 w_k1_FDX_s117
                                 + 0.730566 w_k1_IBM_s117
                                 + 0.929997 w_k1_INTC_s117
                                 + 0.711119 w_k1_KO_s117
                                 + 0.690393 w_k1_MAT_s117
                                 + 0.943431 w_k1_MSFT_s117
                                 + 0.860001 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.21142 w_k1_ADM_s117
                                 + 1.43398 w_k1_AXP_s117
                                 + 1.34983 w_k1_DIS_s117 + 1.2753 w_k1_FDX_s117
                                 + 1.06711 w_k1_IBM_s117
                                 + 1.26307 w_k1_INTC_s117 + 1.1313 w_k1_KO_s117
                                 + 1.16281 w_k1_MAT_s117
                                 + 1.24383 w_k1_MSFT_s117
                                 + 1.15323 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.972447 w_k1_ADM_s117
                                 + 0.773162 w_k1_AXP_s117
                                 + 1.12625 w_k1_DIS_s117
                                 + 0.989832 w_k1_FDX_s117
                                 + 0.861617 w_k1_IBM_s117
                                 + 1.06751 w_k1_INTC_s117
                                 + 1.02324 w_k1_KO_s117
                                 + 0.837595 w_k1_MAT_s117
                                 + 1.07783 w_k1_MSFT_s117
                                 + 0.96501 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00903807465273667 w_k1_ADM_s000
                                 + 0.0144331301598551 w_k1_ADM_s001
                                 + 0.00991417424698736 w_k1_ADM_s002
                                 + 0.011736894688442 w_k1_ADM_s003
                                 + 0.017859527007093 w_k1_ADM_s004
                                 + 0.0120839757330791 w_k1_ADM_s005
                                 + 0.0105006660293848 w_k1_ADM_s006
                                 + 0.0229789366009577 w_k1_ADM_s007
                                 + 0.0102523263508154 w_k1_ADM_s010
                                 + 0.0142087449651023 w_k1_ADM_s011
                                 + 0.013606191166668 w_k1_ADM_s012
                                 + 0.0117551339524091 w_k1_ADM_s013
                                 + 0.0393762319455449 w_k1_ADM_s014
                                 + 0.00996963490653087 w_k1_ADM_s015
                                 + 0.0131024678103622 w_k1_ADM_s016
                                 + 0.0274290856938289 w_k1_ADM_s017
                                 + 0.0330932655313616 w_k1_ADM_s100
                                 + 0.0405710943891627 w_k1_ADM_s101
                                 + 0.0151219966031419 w_k1_ADM_s102
                                 + 0.0445462397204622 w_k1_ADM_s103
                                 + 0.0141189402813895 w_k1_ADM_s104
                                 + 0.0197164054493268 w_k1_ADM_s105
                                 + 0.0603012124137985 w_k1_ADM_s106
                                 + 0.0194179912986259 w_k1_ADM_s107
                                 + 0.0709287543900752 w_k1_ADM_s110
                                 + 0.0381762999701011 w_k1_ADM_s111
                                 + 0.0311197168325243 w_k1_ADM_s112
                                 + 0.121666248168499 w_k1_ADM_s113
                                 + 0.0589884985367261 w_k1_ADM_s114
                                 + 0.0788237278271063 w_k1_ADM_s115
                                 + 0.0497063767942215 w_k1_ADM_s116
                                 + 0.0792710835732672 w_k1_ADM_s117
                                 + 0.0109889134748912 w_k1_AXP_s000
                                 + 0.0149578619311428 w_k1_AXP_s001
                                 + 0.0107278535428492 w_k1_AXP_s002
                                 + 0.0121855028165068 w_k1_AXP_s003
                                 + 0.0196404443888025 w_k1_AXP_s004
                                 + 0.0125746926652553 w_k1_AXP_s005
                                 + 0.00976639719592741 w_k1_AXP_s006
                                 + 0.0268384296314669 w_k1_AXP_s007
                                 + 0.0100419347418355 w_k1_AXP_s010
                                 + 0.0165596573949971 w_k1_AXP_s011
                                 + 0.0145035176926228 w_k1_AXP_s012
                                 + 0.013963422294575 w_k1_AXP_s013
                                 + 0.037930516756454 w_k1_AXP_s014
                                 + 0.00975107063277093 w_k1_AXP_s015
                                 + 0.0120863989092122 w_k1_AXP_s016
                                 + 0.0340358410577659 w_k1_AXP_s017
                                 + 0.0291277991425353 w_k1_AXP_s100
                                 + 0.0458517863266121 w_k1_AXP_s101
                                 + 0.0151125541765118 w_k1_AXP_s102
                                 + 0.0422954958658692 w_k1_AXP_s103
                                 + 0.0128817565889706 w_k1_AXP_s104
                                 + 0.0209943543011887 w_k1_AXP_s105
                                 + 0.0671292527271077 w_k1_AXP_s106
                                 + 0.0207411949369083 w_k1_AXP_s107
                                 + 0.0641258126783245 w_k1_AXP_s110
                                 + 0.0400016938349808 w_k1_AXP_s111
                                 + 0.0332833745631218 w_k1_AXP_s112
                                 + 0.116493807505542 w_k1_AXP_s113
                                 + 0.0516820380376761 w_k1_AXP_s114
                                 + 0.0733627666514911 w_k1_AXP_s115
                                 + 0.0510673657837089 w_k1_AXP_s116
                                 + 0.0798539871403157 w_k1_AXP_s117
                                 + 0.0104262435075086 w_k1_DIS_s000
                                 + 0.0138631948916806 w_k1_DIS_s001
                                 + 0.0110631076893869 w_k1_DIS_s002
                                 + 0.0113167045100982 w_k1_DIS_s003
                                 + 0.0198938792421648 w_k1_DIS_s004
                                 + 0.0121002043250919 w_k1_DIS_s005
                                 + 0.0103779008443299 w_k1_DIS_s006
                                 + 0.0259568910733265 w_k1_DIS_s007
                                 + 0.0102758876917566 w_k1_DIS_s010
                                 + 0.0161470759715418 w_k1_DIS_s011
                                 + 0.0141289173858056 w_k1_DIS_s012
                                 + 0.0123200723906585 w_k1_DIS_s013
                                 + 0.0435656322019202 w_k1_DIS_s014
                                 + 0.0101733819667 w_k1_DIS_s015
                                 + 0.0139795979402494 w_k1_DIS_s016
                                 + 0.0335437091945485 w_k1_DIS_s017
                                 + 0.0292127064832689 w_k1_DIS_s100
                                 + 0.0411567624820235 w_k1_DIS_s101
                                 + 0.0135307555369814 w_k1_DIS_s102
                                 + 0.0438929003361131 w_k1_DIS_s103
                                 + 0.0129244282800816 w_k1_DIS_s104
                                 + 0.0205840285050843 w_k1_DIS_s105
                                 + 0.0640925461010306 w_k1_DIS_s106
                                 + 0.0200628174848096 w_k1_DIS_s107
                                 + 0.0724732473858475 w_k1_DIS_s110
                                 + 0.0393242580160276 w_k1_DIS_s111
                                 + 0.0325345510374682 w_k1_DIS_s112
                                 + 0.124944203155556 w_k1_DIS_s113
                                 + 0.0624532986622481 w_k1_DIS_s114
                                 + 0.0714836641606895 w_k1_DIS_s115
                                 + 0.0434799714536001 w_k1_DIS_s116
                                 + 0.0804301446258118 w_k1_DIS_s117
                                 + 0.0103535922528851 w_k1_FDX_s000
                                 + 0.014496752229486 w_k1_FDX_s001
                                 + 0.0110949430563953 w_k1_FDX_s002
                                 + 0.0118173410437025 w_k1_FDX_s003
                                 + 0.0188251432089427 w_k1_FDX_s004
                                 + 0.0123451046986838 w_k1_FDX_s005
                                 + 0.0105202332110494 w_k1_FDX_s006
                                 + 0.0248891898020497 w_k1_FDX_s007
                                 + 0.0103240179647686 w_k1_FDX_s010
                                 + 0.0156999169774913 w_k1_FDX_s011
                                 + 0.0155937450946667 w_k1_FDX_s012
                                 + 0.0125412355894337 w_k1_FDX_s013
                                 + 0.0415423448925443 w_k1_FDX_s014
                                 + 0.00973247459566651 w_k1_FDX_s015
                                 + 0.0127244933664961 w_k1_FDX_s016
                                 + 0.0324315419467196 w_k1_FDX_s017
                                 + 0.0278320163063205 w_k1_FDX_s100
                                 + 0.0424686157243454 w_k1_FDX_s101
                                 + 0.0132054908397434 w_k1_FDX_s102
                                 + 0.0423729199473103 w_k1_FDX_s103
                                 + 0.0127906845706269 w_k1_FDX_s104
                                 + 0.0204488633689101 w_k1_FDX_s105
                                 + 0.0629477948493471 w_k1_FDX_s106
                                 + 0.0195593074815999 w_k1_FDX_s107
                                 + 0.0663688062345168 w_k1_FDX_s110
                                 + 0.0381157713448838 w_k1_FDX_s111
                                 + 0.0316327630617658 w_k1_FDX_s112
                                 + 0.120320598837417 w_k1_FDX_s113
                                 + 0.0539618268812834 w_k1_FDX_s114
                                 + 0.0724629879171676 w_k1_FDX_s115
                                 + 0.0463853357892021 w_k1_FDX_s116
                                 + 0.0786478271541334 w_k1_FDX_s117
                                 + 0.00967395309172352 w_k1_IBM_s000
                                 + 0.0146725244534862 w_k1_IBM_s001
                                 + 0.0107489173994067 w_k1_IBM_s002
                                 + 0.0119882829473672 w_k1_IBM_s003
                                 + 0.0199781123357913 w_k1_IBM_s004
                                 + 0.0121015491423829 w_k1_IBM_s005
                                 + 0.0113494236734759 w_k1_IBM_s006
                                 + 0.0259899780880213 w_k1_IBM_s007
                                 + 0.0105199779796726 w_k1_IBM_s010
                                 + 0.0174644217469599 w_k1_IBM_s011
                                 + 0.0162203971677662 w_k1_IBM_s012
                                 + 0.0127636133813474 w_k1_IBM_s013
                                 + 0.0432834102128338 w_k1_IBM_s014
                                 + 0.00942553173334424 w_k1_IBM_s015
                                 + 0.0127165712924328 w_k1_IBM_s016
                                 + 0.0338540386235073 w_k1_IBM_s017
                                 + 0.0295463780652097 w_k1_IBM_s100
                                 + 0.0412784318188143 w_k1_IBM_s101
                                 + 0.0123962155055886 w_k1_IBM_s102
                                 + 0.040673173384776 w_k1_IBM_s103
                                 + 0.0137213613605529 w_k1_IBM_s104
                                 + 0.0196895002508714 w_k1_IBM_s105
                                 + 0.0626325245931336 w_k1_IBM_s106
                                 + 0.0192516950230411 w_k1_IBM_s107
                                 + 0.0708683658137228 w_k1_IBM_s110
                                 + 0.0390979961314445 w_k1_IBM_s111
                                 + 0.0322479145145291 w_k1_IBM_s112
                                 + 0.124281723773559 w_k1_IBM_s113
                                 + 0.0557667939969123 w_k1_IBM_s114
                                 + 0.0704472189249808 w_k1_IBM_s115
                                 + 0.0465334997275615 w_k1_IBM_s116
                                 + 0.0785999565612732 w_k1_IBM_s117
                                 + 0.010646594767439 w_k1_INTC_s000
                                 + 0.0126025062878059 w_k1_INTC_s001
                                 + 0.0112931194465266 w_k1_INTC_s002
                                 + 0.0112989286329015 w_k1_INTC_s003
                                 + 0.020248577655815 w_k1_INTC_s004
                                 + 0.0112578040814413 w_k1_INTC_s005
                                 + 0.0104870072781548 w_k1_INTC_s006
                                 + 0.0246136361753922 w_k1_INTC_s007
                                 + 0.00956588605913891 w_k1_INTC_s010
                                 + 0.0148726143242479 w_k1_INTC_s011
                                 + 0.0150104983768342 w_k1_INTC_s012
                                 + 0.0118908449202202 w_k1_INTC_s013
                                 + 0.0415115220921843 w_k1_INTC_s014
                                 + 0.00835479037909887 w_k1_INTC_s015
                                 + 0.0106552833351814 w_k1_INTC_s016
                                 + 0.0344073514950959 w_k1_INTC_s017
                                 + 0.0259750780663303 w_k1_INTC_s100
                                 + 0.0394987571005864 w_k1_INTC_s101
                                 + 0.0125206791341636 w_k1_INTC_s102
                                 + 0.0410822869249573 w_k1_INTC_s103
                                 + 0.0115906475030595 w_k1_INTC_s104
                                 + 0.0196659849416965 w_k1_INTC_s105
                                 + 0.0622353300916145 w_k1_INTC_s106
                                 + 0.0184167550919094 w_k1_INTC_s107
                                 + 0.0720624090755379 w_k1_INTC_s110
                                 + 0.0426709085171811 w_k1_INTC_s111
                                 + 0.0309806188819445 w_k1_INTC_s112
                                 + 0.11368096181533 w_k1_INTC_s113
                                 + 0.0533780004718955 w_k1_INTC_s114
                                 + 0.0675667776572216 w_k1_INTC_s115
                                 + 0.0448241936492238 w_k1_INTC_s116
                                 + 0.0805558372230914 w_k1_INTC_s117
                                 + 0.00915963993033729 w_k1_KO_s000
                                 + 0.0150764006802166 w_k1_KO_s001
                                 + 0.0105067277617678 w_k1_KO_s002
                                 + 0.0108779621831604 w_k1_KO_s003
                                 + 0.0214471832571822 w_k1_KO_s004
                                 + 0.0115890548458538 w_k1_KO_s005
                                 + 0.0100628691009436 w_k1_KO_s006
                                 + 0.0231805288629503 w_k1_KO_s007
                                 + 0.0116464866213558 w_k1_KO_s010
                                 + 0.0159815024295978 w_k1_KO_s011
                                 + 0.0155977790216524 w_k1_KO_s012
                                 + 0.013146737536385 w_k1_KO_s013
                                 + 0.0404229670551901 w_k1_KO_s014
                                 + 0.00911298962482333 w_k1_KO_s015
                                 + 0.0133019790826932 w_k1_KO_s016
                                 + 0.03416969601246 w_k1_KO_s017
                                 + 0.0297239507578873 w_k1_KO_s100
                                 + 0.0393972918786062 w_k1_KO_s101
                                 + 0.0138068656171725 w_k1_KO_s102
                                 + 0.043649546290311 w_k1_KO_s103
                                 + 0.0125879986378465 w_k1_KO_s104
                                 + 0.0192364899416521 w_k1_KO_s105
                                 + 0.063748104307542 w_k1_KO_s106
                                 + 0.0178593027931525 w_k1_KO_s107
                                 + 0.0756303207034042 w_k1_KO_s110
                                 + 0.0384894651620356 w_k1_KO_s111
                                 + 0.0321861296738392 w_k1_KO_s112
                                 + 0.108402860705961 w_k1_KO_s113
                                 + 0.0570211167595819 w_k1_KO_s114
                                 + 0.075871565584987 w_k1_KO_s115
                                 + 0.0459340136156335 w_k1_KO_s116
                                 + 0.0753543347554839 w_k1_KO_s117
                                 + 0.0118753536411151 w_k1_MAT_s000
                                 + 0.0177536877499736 w_k1_MAT_s001
                                 + 0.0095216893921362 w_k1_MAT_s002
                                 + 0.0109452302502793 w_k1_MAT_s003
                                 + 0.0256391601236611 w_k1_MAT_s004
                                 + 0.0113209402810159 w_k1_MAT_s005
                                 + 0.0106124896185649 w_k1_MAT_s006
                                 + 0.0286745642172264 w_k1_MAT_s007
                                 + 0.00988416195710618 w_k1_MAT_s010
                                 + 0.0144269218936353 w_k1_MAT_s011
                                 + 0.015136955767733 w_k1_MAT_s012
                                 + 0.0136628396301182 w_k1_MAT_s013
                                 + 0.0406914584799163 w_k1_MAT_s014
                                 + 0.0107657026118322 w_k1_MAT_s015
                                 + 0.0123504725846578 w_k1_MAT_s016
                                 + 0.0357323394487805 w_k1_MAT_s017
                                 + 0.0296594660770481 w_k1_MAT_s100
                                 + 0.0530388632408066 w_k1_MAT_s101
                                 + 0.0143867879245393 w_k1_MAT_s102
                                 + 0.0345006362360629 w_k1_MAT_s103
                                 + 0.0132191343005638 w_k1_MAT_s104
                                 + 0.0218035694340435 w_k1_MAT_s105
                                 + 0.0694802467262152 w_k1_MAT_s106
                                 + 0.0201311137095668 w_k1_MAT_s107
                                 + 0.0717450831814446 w_k1_MAT_s110
                                 + 0.0349458332582539 w_k1_MAT_s111
                                 + 0.0359345118470962 w_k1_MAT_s112
                                 + 0.12886315614889 w_k1_MAT_s113
                                 + 0.0475275660689265 w_k1_MAT_s114
                                 + 0.0806647877527116 w_k1_MAT_s115
                                 + 0.0485653151784719 w_k1_MAT_s116
                                 + 0.0747624566411079 w_k1_MAT_s117
                                 + 0.0102853650397776 w_k1_MSFT_s000
                                 + 0.0143740636467747 w_k1_MSFT_s001
                                 + 0.0103851107528597 w_k1_MSFT_s002
                                 + 0.0112735202828882 w_k1_MSFT_s003
                                 + 0.0199981691268092 w_k1_MSFT_s004
                                 + 0.0111552295486233 w_k1_MSFT_s005
                                 + 0.0100725431640618 w_k1_MSFT_s006
                                 + 0.0263660274757854 w_k1_MSFT_s007
                                 + 0.0103165911787656 w_k1_MSFT_s010
                                 + 0.0142549910603433 w_k1_MSFT_s011
                                 + 0.0159297919849479 w_k1_MSFT_s012
                                 + 0.0113119170607614 w_k1_MSFT_s013
                                 + 0.0400949543375745 w_k1_MSFT_s014
                                 + 0.00943872929498507 w_k1_MSFT_s015
                                 + 0.0129741144045287 w_k1_MSFT_s016
                                 + 0.0342856719013941 w_k1_MSFT_s017
                                 + 0.0274183378244523 w_k1_MSFT_s100
                                 + 0.0405584064226439 w_k1_MSFT_s101
                                 + 0.0128693475113826 w_k1_MSFT_s102
                                 + 0.0472870541109472 w_k1_MSFT_s103
                                 + 0.0120479770134445 w_k1_MSFT_s104
                                 + 0.0189936828119143 w_k1_MSFT_s105
                                 + 0.0675658627697669 w_k1_MSFT_s106
                                 + 0.0200850207465212 w_k1_MSFT_s107
                                 + 0.0711210191992816 w_k1_MSFT_s110
                                 + 0.0376514839637611 w_k1_MSFT_s111
                                 + 0.0350701831021856 w_k1_MSFT_s112
                                 + 0.117260155585028 w_k1_MSFT_s113
                                 + 0.0553623545928056 w_k1_MSFT_s114
                                 + 0.0753461998503314 w_k1_MSFT_s115
                                 + 0.0474587901588986 w_k1_MSFT_s116
                                 + 0.082332150255293 w_k1_MSFT_s117
                                 + 0.00993003464643026 w_k1_PG_s000
                                 + 0.0141765480809818 w_k1_PG_s001
                                 + 0.0101640556524704 w_k1_PG_s002
                                 + 0.0121950797345731 w_k1_PG_s003
                                 + 0.020213217765991 w_k1_PG_s004
                                 + 0.0114787352720283 w_k1_PG_s005
                                 + 0.00908038020997166 w_k1_PG_s006
                                 + 0.0235583531839654 w_k1_PG_s007
                                 + 0.011022911736056 w_k1_PG_s010
                                 + 0.0148105974130584 w_k1_PG_s011
                                 + 0.0161401613515159 w_k1_PG_s012
                                 + 0.0132661532251177 w_k1_PG_s013
                                 + 0.0415227651413021 w_k1_PG_s014
                                 + 0.00934221845087689 w_k1_PG_s015
                                 + 0.0133753679503349 w_k1_PG_s016
                                 + 0.033927459059103 w_k1_PG_s017
                                 + 0.0263047689137401 w_k1_PG_s100
                                 + 0.0401674856748793 w_k1_PG_s101
                                 + 0.0137781206124175 w_k1_PG_s102
                                 + 0.0386741720225611 w_k1_PG_s103
                                 + 0.0113441030899716 w_k1_PG_s104
                                 + 0.0206803068041199 w_k1_PG_s105
                                 + 0.0607420169223298 w_k1_PG_s106
                                 + 0.021417798742422 w_k1_PG_s107
                                 + 0.0733883497783582 w_k1_PG_s110
                                 + 0.0355863302646245 w_k1_PG_s111
                                 + 0.0316728959194911 w_k1_PG_s112
                                 + 0.121847312609517 w_k1_PG_s113
                                 + 0.0541434147831058 w_k1_PG_s114
                                 + 0.0691113553669732 w_k1_PG_s115
                                 + 0.0454962260996701 w_k1_PG_s116
                                 + 0.0787156199789937 w_k1_PG_s117 >= 1.14
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
