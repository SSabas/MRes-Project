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
 w_k1_ADM_s0_return_constraint:  1.13265 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.900926 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.905699 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.21657 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.07018 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.13907 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.05267 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.20716 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.29108 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.27895 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.764513 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.817683 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.968315 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.01409 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.819105 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.861211 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.804552 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.0502 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.5039 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.728766 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.985351 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.826532 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.34329 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.929074 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.13387 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.784434 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.07237 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.781664 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.807121 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.815063 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.986631 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.761372 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.891257 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.726664 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.29286 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.963772 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.885761 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.2129 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.21309 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.833912 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.86234 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.25719 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.961023 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.16399 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.24456 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.929009 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.55269 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.24544 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.749151 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.763684 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.08132 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.00323 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.881824 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.41739 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.834233 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.993383 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.57791 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.672968 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.5368 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.870375 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.77171 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.12179 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.41788 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.857374 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.13729 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.740009 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.840294 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.886479 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.10181 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.817079 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.05184 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.684238 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.42621 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.06334 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.746855 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.98354 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.22862 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.809553 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.813437 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.32841 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.973196 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.21368 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.43941 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.11594 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.76176 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.28072 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.677336 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.845113 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.08676 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.00065 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.22291 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.1907 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.891398 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.11214 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.93058 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.725889 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.39738 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.831768 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.9232 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.980478 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.48791 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.733009 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.28779 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.626814 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.656132 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.88442 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.19894 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.60762 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.03734 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.698041 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.62681 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.00755 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.658652 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.80539 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.14107 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.867713 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.87908 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.19798 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.01356 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.07721 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.16347 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.15072 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.42412 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.22323 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.80287 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.79435 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.912608 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.996453 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.01157 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.14781 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.781255 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.16062 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.61467 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.800188 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.11563 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.880071 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.65995 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.990869 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.17966 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.732479 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.16014 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.747973 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.728682 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.98499 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.00317 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.647018 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.09974 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.70323 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.40437 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.02038 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.819714 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.35827 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.1044 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.895453 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.920767 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.21081 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.878578 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.52362 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.4973 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.679883 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.18435 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.902077 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.719336 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.23539 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.86996 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.08881 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.07602 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.35197 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.478 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.722996 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.18217 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.708588 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.95237 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.02782 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.30021 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.16786 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.67227 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.08354 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.10837 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.854487 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.35357 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.705719 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.40573 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.04529 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.704328 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.793486 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.02774 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.9889 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.73271 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.65646 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.08191 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.891512 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.858947 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.10955 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.968017 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.1843 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.23956 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.1896 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.53456 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.2212 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.01721 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.873291 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.10896 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.00961 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.07078 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.29989 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.952179 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.12414 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.57058 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.734327 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.37812 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.907507 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.3022 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.03977 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.10385 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.905268 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.09984 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.927789 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.878131 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.03583 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.881452 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.672595 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.933353 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.756105 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.1003 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.928853 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.806305 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.22251 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.12189 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.852632 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.887426 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.17723 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.970378 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.3577 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.22531 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.936437 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.35004 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.01187 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.838924 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.08492 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.37033 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.00209 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.14092 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.19385 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.11163 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.966229 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.46346 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.745354 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.75894 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.987624 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.42593 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.04157 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.29614 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.860662 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.1461 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.845609 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.946285 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.945322 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.08829 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.775908 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.72931 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.736518 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.32818 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.851859 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.745904 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.51627 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.24083 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.806084 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.818348 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.06923 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.832869 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.65493 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.36424 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.655858 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.18365 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.993457 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.755351 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.30812 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.87808 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.966351 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.01178 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.59303 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.67695 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.734648 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.44305 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.700802 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.27444 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.11712 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.73341 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.43183 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 2.08556 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.21356 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.47222 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.976211 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.33743 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.866361 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.36418 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.958188 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.615878 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.732461 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.04868 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.00028 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.560786 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 2.37913 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1371 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.866997 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.864957 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.15707 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.02576 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.14818 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.2457 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.15524 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.33361 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.17557 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.848662 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.88367 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.04055 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.03143 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.00038 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.08887 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.955106 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.973365 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.48297 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.802152 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.1142 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.856559 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.39939 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.934044 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.13303 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.807053 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.16005 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.75806 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.760586 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.954247 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.983589 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.677086 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.954036 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.777267 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.33864 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.0014 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.886095 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.40868 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12249 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.896762 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.882757 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.1372 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.01485 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.16029 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.16347 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.99728 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.32205 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.0085 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.872611 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.991093 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.16131 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.01701 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.0155 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.08832 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.966892 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.960901 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.247 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.811862 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.20384 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.910774 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.36261 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.95172 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.12223 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.921463 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.05052 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.82591 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.911653 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.900664 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.14161 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.744631 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.922732 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.76218 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.0917 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.948381 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.865749 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.2857 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.231000003347791 z_k1_s0000
                                 - 0.196000048967835 z_k1_s0001
                                 - 0.376001264065154 z_k1_s0002
                                 - 0.260999021930346 z_k1_s0003
                                 - 0.188999956611816 z_k1_s0004
                                 - 0.312000347181527 z_k1_s0005
                                 - 0.347000808961624 z_k1_s0006
                                 - 0.625000272642364 z_k1_s0007
                                 - 0.83000112354051 z_k1_s0010
                                 - 0.970000709183764 z_k1_s0011
                                 - 0.50100181882788 z_k1_s0012
                                 - 0.476001932598868 z_k1_s0013
                                 - 0.676001022430964 z_k1_s0014
                                 - 1.05800235810913 z_k1_s0015
                                 - 0.809000350341071 z_k1_s0016
                                 - 0.249000893964121 z_k1_s0017
                                 - 0.172000022084184 z_k1_s0020
                                 - 0.130000043356664 z_k1_s0021
                                 - 0.443000594626391 z_k1_s0022
                                 - 0.234000154482033 z_k1_s0023
                                 - 0.234000154482033 z_k1_s0024
                                 - 0.116000050447491 z_k1_s0025
                                 - 0.230999883001361 z_k1_s0026
                                 - 0.350999276222578 z_k1_s0027
                                 - 0.294001289813363 z_k1_s0030
                                 - 0.160000113469017 z_k1_s0031
                                 - 0.617999712460543 z_k1_s0032
                                 - 0.236000433992181 z_k1_s0033
                                 - 0.16700015546459 z_k1_s0034
                                 - 0.310001182660096 z_k1_s0035
                                 - 0.272000548397837 z_k1_s0036
                                 - 0.313001606944493 z_k1_s0037
                                 - 0.912001847912959 z_k1_s0040
                                 - 0.559001007155598 z_k1_s0041
                                 - 1.3460028909903 z_k1_s0042
                                 - 0.738001659048328 z_k1_s0043
                                 - 1.29399940254084 z_k1_s0044
                                 - 1.13200065670434 z_k1_s0045
                                 - 1.57200656830439 z_k1_s0046
                                 - 0.741001605104496 z_k1_s0047
                                 - 0.384999108384765 z_k1_s0050
                                 - 0.604999085890704 z_k1_s0051
                                 - 0.498999561592731 z_k1_s0052
                                 - 0.285000110319161 z_k1_s0053
                                 - 0.248000160588651 z_k1_s0054
                                 - 0.309000169847128 z_k1_s0055
                                 - 0.645000321438154 z_k1_s0056
                                 - 0.433001272842208 z_k1_s0057
                                 - 0.425999228039421 z_k1_s0060
                                 - 0.511999773906761 z_k1_s0061
                                 - 0.54100113730427 z_k1_s0062
                                 - 0.229000007304426 z_k1_s0063
                                 - 0.337999600524304 z_k1_s0064
                                 - 0.120000147184462 z_k1_s0065
                                 - 0.198000096059315 z_k1_s0066
                                 - 0.305000073831244 z_k1_s0067
                                 - 1.13699944496158 z_k1_s0070
                                 - 0.367999530570539 z_k1_s0071
                                 - 0.312999900152077 z_k1_s0072
                                 - 0.428999804761215 z_k1_s0073
                                 - 0.740996730954232 z_k1_s0074
                                 - 0.912996552443681 z_k1_s0075
                                 - 0.685000039070573 z_k1_s0076
                                 - 0.789000806133053 z_k1_s0077
                                 - 0.297000207488967 z_k1_s0100
                                 - 0.249999064614627 z_k1_s0101
                                 - 0.555000025762544 z_k1_s0102
                                 - 0.5170006266287 z_k1_s0103
                                 - 0.111999800735151 z_k1_s0104
                                 - 0.25699882819203 z_k1_s0105
                                 - 0.23199987732969 z_k1_s0106
                                 - 0.168999855035041 z_k1_s0107
                                 - 0.199999835661765 z_k1_s0110
                                 - 0.230000254750505 z_k1_s0111
                                 - 0.107999935461325 z_k1_s0112
                                 - 0.296998864444777 z_k1_s0113
                                 - 0.219999980587751 z_k1_s0114
                                 - 0.38000033319659 z_k1_s0115
                                 - 0.270999765219706 z_k1_s0116
                                 - 0.31099884487311 z_k1_s0117
                                 - 0.300999083035235 z_k1_s0120
                                 - 0.189999449025521 z_k1_s0121
                                 - 0.257999692886611 z_k1_s0122
                                 - 0.242999515858618 z_k1_s0123
                                 - 0.203000161115593 z_k1_s0124
                                 - 0.126999878706154 z_k1_s0125
                                 - 0.163999700842811 z_k1_s0126
                                 - 0.189999449025521 z_k1_s0127
                                 - 0.764999978356282 z_k1_s0130
                                 - 1.02399849173864 z_k1_s0131
                                 - 0.482998654973842 z_k1_s0132
                                 - 0.32699914015668 z_k1_s0133
                                 - 0.990998122849817 z_k1_s0134
                                 - 1.19599774722739 z_k1_s0135
                                 - 0.547998505205843 z_k1_s0136
                                 - 0.952999603214829 z_k1_s0137
                                 - 0.2189997603112 z_k1_s0140
                                 - 0.114999812994573 z_k1_s0141
                                 - 0.0739998389139708 z_k1_s0142
                                 - 0.15999950694938 z_k1_s0143
                                 - 0.239999260424071 z_k1_s0144
                                 - 0.186000163277361 z_k1_s0145
                                 - 0.114999812994573 z_k1_s0146
                                 - 0.229999625989146 z_k1_s0147
                                 - 0.63799657312636 z_k1_s0150
                                 - 0.446998850165278 z_k1_s0151
                                 - 0.768996339080575 z_k1_s0152
                                 - 0.435998743215247 z_k1_s0153
                                 - 0.320999181460953 z_k1_s0154
                                 - 0.775999471222782 z_k1_s0155
                                 - 0.596999627667439 z_k1_s0156
                                 - 1.36599596674801 z_k1_s0157
                                 - 0.137999802312573 z_k1_s0160
                                 - 0.175999944659359 z_k1_s0161
                                 - 0.0909998709887243 z_k1_s0162
                                 - 0.115999925480062 z_k1_s0163
                                 - 0.448999929625242 z_k1_s0164
                                 - 0.329000263266357 z_k1_s0165
                                 - 0.400000127861982 z_k1_s0166
                                 - 0.160999800364644 z_k1_s0167
                                 - 0.473998795333843 z_k1_s0170
                                 - 0.174999744413196 z_k1_s0171
                                 - 0.296999377457034 z_k1_s0172
                                 - 0.320999504212827 z_k1_s0173
                                 - 0.22899969251436 z_k1_s0174
                                 - 0.869999538406948 z_k1_s0175
                                 - 0.630999455978304 z_k1_s0176
                                 - 0.373998267184705 z_k1_s0177
                                 - 0.473000145231795 z_k1_s1000
                                 - 0.250999444404327 z_k1_s1001
                                 - 0.16899997506698 z_k1_s1002
                                 - 0.131999858262402 z_k1_s1003
                                 - 0.41300053352154 z_k1_s1004
                                 - 0.361999794818061 z_k1_s1005
                                 - 0.161000026838946 z_k1_s1006
                                 - 0.414000408300071 z_k1_s1007
                                 - 0.231999854658201 z_k1_s1010
                                 - 0.41100018780183 z_k1_s1011
                                 - 0.55099813561464 z_k1_s1012
                                 - 0.404000480311013 z_k1_s1013
                                 - 0.462998557447396 z_k1_s1014
                                 - 0.274999631271757 z_k1_s1015
                                 - 0.362999589238649 z_k1_s1016
                                 - 1.0989996520159 z_k1_s1017
                                 - 0.647998296552438 z_k1_s1020
                                 - 0.530000754883132 z_k1_s1021
                                 - 0.577998196850893 z_k1_s1022
                                 - 0.252000030625688 z_k1_s1023
                                 - 0.570998843480491 z_k1_s1024
                                 - 0.172999918105372 z_k1_s1025
                                 - 0.249000026352764 z_k1_s1026
                                 - 0.281999745055045 z_k1_s1027
                                 - 0.317999837113198 z_k1_s1030
                                 - 0.128999875862939 z_k1_s1031
                                 - 0.261999229898713 z_k1_s1032
                                 - 0.171999981017156 z_k1_s1033
                                 - 0.172999850560365 z_k1_s1034
                                 - 0.241999641035984 z_k1_s1035
                                 - 0.472000407953754 z_k1_s1036
                                 - 0.429998853543612 z_k1_s1037
                                 - 0.227999091500734 z_k1_s1040
                                 - 0.88599448836561 z_k1_s1041
                                 - 0.732996767141698 z_k1_s1042
                                 - 0.721995428758733 z_k1_s1043
                                 - 0.624998895687558 z_k1_s1044
                                 - 0.459996499876208 z_k1_s1045
                                 - 0.372998410981852 z_k1_s1046
                                 - 0.392998393323787 z_k1_s1047
                                 - 0.715999071866725 z_k1_s1050
                                 - 0.226999358429001 z_k1_s1051
                                 - 0.873997317047872 z_k1_s1052
                                 - 0.95799827020278 z_k1_s1053
                                 - 0.425998741672181 z_k1_s1054
                                 - 0.629999441680973 z_k1_s1055
                                 - 0.378999003524006 z_k1_s1056
                                 - 0.927997562791226 z_k1_s1057
                                 - 0.451999771166934 z_k1_s1060
                                 - 0.327999939563578 z_k1_s1061
                                 - 0.37099934216816 z_k1_s1062
                                 - 0.333000702773115 z_k1_s1063
                                 - 0.258998955268548 z_k1_s1064
                                 - 0.144999933984523 z_k1_s1065
                                 - 0.175999891885361 z_k1_s1066
                                 - 0.322999176354041 z_k1_s1067
                                 - 1.10900124296244 z_k1_s1070
                                 - 1.43799270966709 z_k1_s1071
                                 - 0.440999312666546 z_k1_s1072
                                 - 0.954994307506709 z_k1_s1073
                                 - 2.06699368415108 z_k1_s1074
                                 - 1.26799572790774 z_k1_s1075
                                 - 0.879998001549077 z_k1_s1076
                                 - 1.18199318914393 z_k1_s1077
                                 - 0.234000045429981 z_k1_s1100
                                 - 0.361000496502525 z_k1_s1101
                                 - 0.191000644951498 z_k1_s1102
                                 - 0.165000592725771 z_k1_s1103
                                 - 0.372001923049355 z_k1_s1104
                                 - 0.361000496502525 z_k1_s1105
                                 - 0.197000689665248 z_k1_s1106
                                 - 0.242001661920719 z_k1_s1107
                                 - 0.0990001308418183 z_k1_s1110
                                 - 0.247000106710002 z_k1_s1111
                                 - 0.114000397866778 z_k1_s1112
                                 - 0.0670000748161555 z_k1_s1113
                                 - 0.243000609557706 z_k1_s1114
                                 - 0.127999770901841 z_k1_s1115
                                 - 0.10900023332499 z_k1_s1116
                                 - 0.126000022325693 z_k1_s1117
                                 - 0.188000172669259 z_k1_s1120
                                 - 0.143000601419382 z_k1_s1121
                                 - 0.0959999020010288 z_k1_s1122
                                 - 0.0879997643340455 z_k1_s1123
                                 - 0.101000534918759 z_k1_s1124
                                 - 0.0930003972517759 z_k1_s1125
                                 - 0.101000534918759 z_k1_s1126
                                 - 0.0650000904176114 z_k1_s1127
                                 - 0.0719999835370329 z_k1_s1130
                                 - 0.0679998651181472 z_k1_s1131
                                 - 0.082999906438423 z_k1_s1132
                                 - 0.0569999422167571 z_k1_s1133
                                 - 0.0420000619966995 z_k1_s1134
                                 - 0.0879999202118483 z_k1_s1135
                                 - 0.0870000248573087 z_k1_s1136
                                 - 0.0400000564873294 z_k1_s1137
                                 - 0.166999414915719 z_k1_s1140
                                 - 0.188999538964351 z_k1_s1141
                                 - 0.0689999027710606 z_k1_s1142
                                 - 0.113999952943409 z_k1_s1143
                                 - 0.238999594711404 z_k1_s1144
                                 - 0.279999316984223 z_k1_s1145
                                 - 0.0989999362192929 z_k1_s1146
                                 - 0.0869999477199819 z_k1_s1147
                                 - 0.187000528511284 z_k1_s1150
                                 - 0.192000556482207 z_k1_s1151
                                 - 0.168000422221776 z_k1_s1152
                                 - 0.397000150486336 z_k1_s1153
                                 - 0.296999591067874 z_k1_s1154
                                 - 0.771000689907664 z_k1_s1155
                                 - 0.295001650284462 z_k1_s1156
                                 - 0.281000536763398 z_k1_s1157
                                 - 0.214000865840844 z_k1_s1160
                                 - 0.107000121920115 z_k1_s1161
                                 - 0.133000058536127 z_k1_s1162
                                 - 0.155000220211425 z_k1_s1163
                                 - 0.232999874559305 z_k1_s1164
                                 - 0.28700041276245 z_k1_s1165
                                 - 0.301999957540506 z_k1_s1166
                                 - 0.12400017616914 z_k1_s1167
                                 - 0.113000086994477 z_k1_s1170
                                 - 0.0900001311992021 z_k1_s1171
                                 - 0.0310000063241311 z_k1_s1172
                                 - 0.12400014189664 z_k1_s1173
                                 - 0.0689998673858238 z_k1_s1174
                                 - 0.0440000447084795 z_k1_s1175
                                 - 0.0680000213948757 z_k1_s1176
                                 - 0.0440000447084795 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.990749 w_k1_ADM_s000
                                 + 1.08909 w_k1_AXP_s000
                                 + 0.865806 w_k1_DIS_s000
                                 + 1.03835 w_k1_FDX_s000
                                 + 0.706711 w_k1_IBM_s000
                                 + 0.957403 w_k1_INTC_s000
                                 + 0.776406 w_k1_KO_s000
                                 + 0.710112 w_k1_MAT_s000
                                 + 0.866374 w_k1_MSFT_s000
                                 + 0.878021 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.57089 w_k1_ADM_s000
                                 + 1.48721 w_k1_AXP_s000
                                 + 1.49404 w_k1_DIS_s000
                                 + 1.22363 w_k1_FDX_s000
                                 + 1.74146 w_k1_IBM_s000
                                 + 1.23317 w_k1_INTC_s000
                                 + 1.69226 w_k1_KO_s000 + 1.87005 w_k1_MAT_s000
                                 + 1.1446 w_k1_MSFT_s000 + 1.25206 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.38072 w_k1_ADM_s000
                                 + 1.41499 w_k1_AXP_s000
                                 + 1.25222 w_k1_DIS_s000
                                 + 1.27062 w_k1_FDX_s000
                                 + 1.22441 w_k1_IBM_s000
                                 + 1.23062 w_k1_INTC_s000
                                 + 1.43111 w_k1_KO_s000 + 1.39419 w_k1_MAT_s000
                                 + 1.06066 w_k1_MSFT_s000
                                 + 1.22082 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.16447 w_k1_ADM_s000
                                 + 1.13464 w_k1_AXP_s000
                                 + 0.951978 w_k1_DIS_s000
                                 + 1.1396 w_k1_FDX_s000
                                 + 0.829844 w_k1_IBM_s000
                                 + 0.913551 w_k1_INTC_s000
                                 + 0.99896 w_k1_KO_s000 + 1.00438 w_k1_MAT_s000
                                 + 0.893844 w_k1_MSFT_s000
                                 + 1.08396 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.940112 w_k1_ADM_s000
                                 + 0.816041 w_k1_AXP_s000
                                 + 0.555989 w_k1_DIS_s000
                                 + 0.824981 w_k1_FDX_s000
                                 + 0.614048 w_k1_IBM_s000
                                 + 0.925159 w_k1_INTC_s000
                                 + 0.707599 w_k1_KO_s000
                                 + 0.604226 w_k1_MAT_s000
                                 + 0.674 w_k1_MSFT_s000 + 0.807145 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.10074 w_k1_ADM_s000
                                 + 1.20077 w_k1_AXP_s000
                                 + 0.925779 w_k1_DIS_s000
                                 + 0.960015 w_k1_FDX_s000
                                 + 1.26164 w_k1_IBM_s000
                                 + 0.743144 w_k1_INTC_s000
                                 + 1.05684 w_k1_KO_s000 + 1.59318 w_k1_MAT_s000
                                 + 0.84847 w_k1_MSFT_s000
                                 + 1.00217 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.1846 w_k1_ADM_s000
                                 + 1.06147 w_k1_AXP_s000
                                 + 1.27162 w_k1_DIS_s000
                                 + 1.08892 w_k1_FDX_s000
                                 + 1.01694 w_k1_IBM_s000
                                 + 0.998788 w_k1_INTC_s000
                                 + 1.37565 w_k1_KO_s000 + 1.41122 w_k1_MAT_s000
                                 + 1.08958 w_k1_MSFT_s000
                                 + 1.16212 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.07555 w_k1_ADM_s000
                                 + 0.886211 w_k1_AXP_s000
                                 + 0.784423 w_k1_DIS_s000
                                 + 0.889618 w_k1_FDX_s000
                                 + 0.885083 w_k1_IBM_s000
                                 + 0.933829 w_k1_INTC_s000
                                 + 0.917988 w_k1_KO_s000
                                 + 0.953096 w_k1_MAT_s000
                                 + 0.826985 w_k1_MSFT_s000
                                 + 0.890852 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.681865 w_k1_ADM_s001
                                 + 0.843609 w_k1_AXP_s001
                                 + 0.946146 w_k1_DIS_s001
                                 + 0.819576 w_k1_FDX_s001
                                 + 1.03819 w_k1_IBM_s001
                                 + 0.797212 w_k1_INTC_s001
                                 + 0.984893 w_k1_KO_s001
                                 + 0.892145 w_k1_MAT_s001
                                 + 0.877364 w_k1_MSFT_s001
                                 + 0.976394 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.752982 w_k1_ADM_s001
                                 + 0.919249 w_k1_AXP_s001
                                 + 0.616989 w_k1_DIS_s001
                                 + 0.826513 w_k1_FDX_s001
                                 + 0.749486 w_k1_IBM_s001
                                 + 0.79177 w_k1_INTC_s001
                                 + 0.668806 w_k1_KO_s001
                                 + 0.864537 w_k1_MAT_s001
                                 + 0.734953 w_k1_MSFT_s001
                                 + 0.894238 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.821176 w_k1_ADM_s001
                                 + 0.935611 w_k1_AXP_s001
                                 + 0.784792 w_k1_DIS_s001
                                 + 0.821215 w_k1_FDX_s001
                                 + 1.47155 w_k1_IBM_s001
                                 + 0.808031 w_k1_INTC_s001
                                 + 0.906462 w_k1_KO_s001 + 1.7053 w_k1_MAT_s001
                                 + 0.822516 w_k1_MSFT_s001
                                 + 0.938311 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.11716 w_k1_ADM_s001
                                 + 1.54413 w_k1_AXP_s001
                                 + 1.74664 w_k1_DIS_s001
                                 + 1.25485 w_k1_FDX_s001
                                 + 1.53293 w_k1_IBM_s001
                                 + 1.18009 w_k1_INTC_s001
                                 + 1.65589 w_k1_KO_s001 + 1.94624 w_k1_MAT_s001
                                 + 1.22151 w_k1_MSFT_s001
                                 + 1.26659 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.987905 w_k1_ADM_s001
                                 + 1.26577 w_k1_AXP_s001
                                 + 1.25142 w_k1_DIS_s001
                                 + 1.28369 w_k1_FDX_s001
                                 + 1.03936 w_k1_IBM_s001 + 1.28 w_k1_INTC_s001
                                 + 1.01203 w_k1_KO_s001 + 1.05577 w_k1_MAT_s001
                                 + 1.27317 w_k1_MSFT_s001
                                 + 1.13648 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.858283 w_k1_ADM_s001
                                 + 1.20279 w_k1_AXP_s001
                                 + 1.11723 w_k1_DIS_s001
                                 + 1.10157 w_k1_FDX_s001
                                 + 1.11456 w_k1_IBM_s001
                                 + 1.11207 w_k1_INTC_s001
                                 + 0.972888 w_k1_KO_s001
                                 + 1.04358 w_k1_MAT_s001
                                 + 0.951583 w_k1_MSFT_s001
                                 + 1.04869 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.843528 w_k1_ADM_s001
                                 + 1.04953 w_k1_AXP_s001
                                 + 0.933996 w_k1_DIS_s001
                                 + 1.03848 w_k1_FDX_s001
                                 + 0.798243 w_k1_IBM_s001
                                 + 1.06954 w_k1_INTC_s001
                                 + 0.904143 w_k1_KO_s001
                                 + 1.07374 w_k1_MAT_s001
                                 + 0.979964 w_k1_MSFT_s001
                                 + 1.04392 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.929193 w_k1_ADM_s001
                                 + 0.934483 w_k1_AXP_s001
                                 + 0.76097 w_k1_DIS_s001
                                 + 0.885741 w_k1_FDX_s001
                                 + 1.05406 w_k1_IBM_s001
                                 + 0.900296 w_k1_INTC_s001
                                 + 1.04288 w_k1_KO_s001
                                 + 0.813022 w_k1_MAT_s001
                                 + 0.792323 w_k1_MSFT_s001
                                 + 0.966305 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.08291 w_k1_ADM_s002
                                 + 0.663258 w_k1_AXP_s002
                                 + 0.959584 w_k1_DIS_s002
                                 + 0.999524 w_k1_FDX_s002
                                 + 0.653105 w_k1_IBM_s002
                                 + 0.866171 w_k1_INTC_s002
                                 + 0.825249 w_k1_KO_s002
                                 + 0.53416 w_k1_MAT_s002
                                 + 0.998701 w_k1_MSFT_s002
                                 + 0.940054 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.2625 w_k1_ADM_s002
                                 + 1.34423 w_k1_AXP_s002
                                 + 1.43454 w_k1_DIS_s002
                                 + 1.44062 w_k1_FDX_s002
                                 + 0.718207 w_k1_IBM_s002
                                 + 0.942076 w_k1_INTC_s002
                                 + 1.0493 w_k1_KO_s002 + 1.11064 w_k1_MAT_s002
                                 + 1.33715 w_k1_MSFT_s002
                                 + 1.01123 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.16131 w_k1_ADM_s002
                                 + 1.33108 w_k1_AXP_s002
                                 + 1.36097 w_k1_DIS_s002
                                 + 1.27282 w_k1_FDX_s002
                                 + 1.16099 w_k1_IBM_s002
                                 + 0.976825 w_k1_INTC_s002
                                 + 1.18992 w_k1_KO_s002 + 1.11525 w_k1_MAT_s002
                                 + 1.24906 w_k1_MSFT_s002
                                 + 1.14432 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.827024 w_k1_ADM_s002
                                 + 0.901437 w_k1_AXP_s002
                                 + 0.72003 w_k1_DIS_s002
                                 + 0.951784 w_k1_FDX_s002
                                 + 0.807997 w_k1_IBM_s002
                                 + 0.796759 w_k1_INTC_s002
                                 + 0.76346 w_k1_KO_s002
                                 + 0.932142 w_k1_MAT_s002
                                 + 0.907488 w_k1_MSFT_s002
                                 + 0.879387 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.924306 w_k1_ADM_s002
                                 + 0.75727 w_k1_AXP_s002
                                 + 0.830721 w_k1_DIS_s002
                                 + 0.859852 w_k1_FDX_s002
                                 + 0.781987 w_k1_IBM_s002
                                 + 0.735311 w_k1_INTC_s002
                                 + 0.790473 w_k1_KO_s002
                                 + 0.690451 w_k1_MAT_s002
                                 + 0.8055 w_k1_MSFT_s002
                                 + 0.865104 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.871975 w_k1_ADM_s002
                                 + 0.752938 w_k1_AXP_s002
                                 + 0.633364 w_k1_DIS_s002
                                 + 0.731861 w_k1_FDX_s002
                                 + 0.924311 w_k1_IBM_s002
                                 + 0.705927 w_k1_INTC_s002
                                 + 0.760025 w_k1_KO_s002 + 0.9273 w_k1_MAT_s002
                                 + 0.688527 w_k1_MSFT_s002
                                 + 0.743161 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.00296 w_k1_ADM_s002
                                 + 1.05793 w_k1_AXP_s002
                                 + 1.18416 w_k1_DIS_s002
                                 + 1.00199 w_k1_FDX_s002
                                 + 1.68703 w_k1_IBM_s002
                                 + 0.97115 w_k1_INTC_s002
                                 + 1.28079 w_k1_KO_s002 + 1.43632 w_k1_MAT_s002
                                 + 0.949806 w_k1_MSFT_s002
                                 + 0.97519 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.848183 w_k1_ADM_s002
                                 + 1.00054 w_k1_AXP_s002
                                 + 1.01464 w_k1_DIS_s002
                                 + 1.04397 w_k1_FDX_s002
                                 + 1.11302 w_k1_IBM_s002
                                 + 1.04028 w_k1_INTC_s002
                                 + 1.15952 w_k1_KO_s002 + 1.54055 w_k1_MAT_s002
                                 + 1.06153 w_k1_MSFT_s002
                                 + 0.984623 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.37276 w_k1_ADM_s003
                                 + 1.26418 w_k1_AXP_s003
                                 + 1.65717 w_k1_DIS_s003
                                 + 1.31892 w_k1_FDX_s003
                                 + 1.23248 w_k1_IBM_s003
                                 + 1.35372 w_k1_INTC_s003
                                 + 1.58448 w_k1_KO_s003 + 1.23309 w_k1_MAT_s003
                                 + 1.45037 w_k1_MSFT_s003
                                 + 1.33682 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.29683 w_k1_ADM_s003
                                 + 1.67684 w_k1_AXP_s003 + 1.2045 w_k1_DIS_s003
                                 + 1.23205 w_k1_FDX_s003
                                 + 1.03923 w_k1_IBM_s003
                                 + 1.25459 w_k1_INTC_s003
                                 + 1.15273 w_k1_KO_s003 + 1.53962 w_k1_MAT_s003
                                 + 1.12651 w_k1_MSFT_s003
                                 + 1.30709 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.799524 w_k1_ADM_s003
                                 + 0.938301 w_k1_AXP_s003
                                 + 0.69306 w_k1_DIS_s003
                                 + 0.770978 w_k1_FDX_s003
                                 + 0.859489 w_k1_IBM_s003
                                 + 0.91848 w_k1_INTC_s003
                                 + 0.909074 w_k1_KO_s003
                                 + 0.78772 w_k1_MAT_s003
                                 + 0.845104 w_k1_MSFT_s003
                                 + 0.916515 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.975367 w_k1_ADM_s003
                                 + 0.905205 w_k1_AXP_s003
                                 + 0.883642 w_k1_DIS_s003
                                 + 0.89019 w_k1_FDX_s003
                                 + 1.38123 w_k1_IBM_s003
                                 + 0.822965 w_k1_INTC_s003
                                 + 1.25345 w_k1_KO_s003 + 1.39096 w_k1_MAT_s003
                                 + 0.865532 w_k1_MSFT_s003
                                 + 1.19502 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.07044 w_k1_ADM_s003
                                 + 1.26321 w_k1_AXP_s003
                                 + 1.04494 w_k1_DIS_s003
                                 + 1.18247 w_k1_FDX_s003
                                 + 0.828887 w_k1_IBM_s003
                                 + 0.984782 w_k1_INTC_s003
                                 + 0.895441 w_k1_KO_s003
                                 + 0.903386 w_k1_MAT_s003
                                 + 1.18311 w_k1_MSFT_s003 + 1.1209 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.0262 w_k1_ADM_s003
                                 + 0.998667 w_k1_AXP_s003
                                 + 1.30711 w_k1_DIS_s003
                                 + 1.01857 w_k1_FDX_s003
                                 + 1.37926 w_k1_IBM_s003
                                 + 1.21105 w_k1_INTC_s003
                                 + 1.43175 w_k1_KO_s003 + 1.23564 w_k1_MAT_s003
                                 + 1.06007 w_k1_MSFT_s003 + 1.1916 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.862198 w_k1_ADM_s003
                                 + 1.14775 w_k1_AXP_s003
                                 + 0.997688 w_k1_DIS_s003
                                 + 1.0607 w_k1_FDX_s003 + 1.10498 w_k1_IBM_s003
                                 + 1.03187 w_k1_INTC_s003
                                 + 1.11154 w_k1_KO_s003 + 1.27483 w_k1_MAT_s003
                                 + 1.09293 w_k1_MSFT_s003
                                 + 1.07906 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.907252 w_k1_ADM_s003
                                 + 0.90164 w_k1_AXP_s003
                                 + 0.911375 w_k1_DIS_s003
                                 + 1.0122 w_k1_FDX_s003
                                 + 0.775175 w_k1_IBM_s003
                                 + 1.14167 w_k1_INTC_s003
                                 + 1.00229 w_k1_KO_s003
                                 + 0.895371 w_k1_MAT_s003
                                 + 1.0449 w_k1_MSFT_s003 + 1.13327 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.11717 w_k1_ADM_s004
                                 + 0.923871 w_k1_AXP_s004
                                 + 1.07994 w_k1_DIS_s004
                                 + 1.07425 w_k1_FDX_s004
                                 + 0.831543 w_k1_IBM_s004
                                 + 0.89089 w_k1_INTC_s004
                                 + 0.727089 w_k1_KO_s004
                                 + 0.953564 w_k1_MAT_s004
                                 + 0.98521 w_k1_MSFT_s004
                                 + 1.00035 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.56857 w_k1_ADM_s004
                                 + 1.32346 w_k1_AXP_s004
                                 + 1.59752 w_k1_DIS_s004
                                 + 1.45553 w_k1_FDX_s004
                                 + 0.839974 w_k1_IBM_s004
                                 + 1.34691 w_k1_INTC_s004
                                 + 0.931996 w_k1_KO_s004
                                 + 0.815279 w_k1_MAT_s004
                                 + 1.4372 w_k1_MSFT_s004 + 1.01799 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.677893 w_k1_ADM_s004
                                 + 0.6528 w_k1_AXP_s004
                                 + 0.656164 w_k1_DIS_s004
                                 + 0.703474 w_k1_FDX_s004
                                 + 1.00856 w_k1_IBM_s004
                                 + 0.701782 w_k1_INTC_s004
                                 + 0.879723 w_k1_KO_s004
                                 + 0.868302 w_k1_MAT_s004
                                 + 0.784989 w_k1_MSFT_s004
                                 + 0.801332 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.13971 w_k1_ADM_s004
                                 + 1.00451 w_k1_AXP_s004
                                 + 0.792643 w_k1_DIS_s004
                                 + 1.00267 w_k1_FDX_s004
                                 + 0.601171 w_k1_IBM_s004
                                 + 0.766708 w_k1_INTC_s004
                                 + 0.647574 w_k1_KO_s004
                                 + 0.753231 w_k1_MAT_s004
                                 + 0.84463 w_k1_MSFT_s004
                                 + 0.912577 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.29605 w_k1_ADM_s004
                                 + 1.18202 w_k1_AXP_s004
                                 + 1.21442 w_k1_DIS_s004
                                 + 1.10036 w_k1_FDX_s004
                                 + 1.10777 w_k1_IBM_s004
                                 + 0.942307 w_k1_INTC_s004
                                 + 1.08718 w_k1_KO_s004
                                 + 0.996658 w_k1_MAT_s004
                                 + 1.10695 w_k1_MSFT_s004
                                 + 1.11856 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.812814 w_k1_ADM_s004
                                 + 0.643968 w_k1_AXP_s004
                                 + 0.935748 w_k1_DIS_s004
                                 + 0.91761 w_k1_FDX_s004
                                 + 0.674038 w_k1_IBM_s004
                                 + 0.806843 w_k1_INTC_s004
                                 + 0.732606 w_k1_KO_s004
                                 + 0.544941 w_k1_MAT_s004
                                 + 0.922234 w_k1_MSFT_s004
                                 + 0.947809 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.85919 w_k1_ADM_s004
                                 + 0.854102 w_k1_AXP_s004
                                 + 1.04273 w_k1_DIS_s004
                                 + 0.944945 w_k1_FDX_s004
                                 + 1.40685 w_k1_IBM_s004
                                 + 0.797506 w_k1_INTC_s004
                                 + 0.947361 w_k1_KO_s004
                                 + 1.43947 w_k1_MAT_s004
                                 + 0.996178 w_k1_MSFT_s004
                                 + 1.07904 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.43802 w_k1_ADM_s004
                                 + 1.41711 w_k1_AXP_s004
                                 + 1.68173 w_k1_DIS_s004
                                 + 1.33353 w_k1_FDX_s004
                                 + 1.52814 w_k1_IBM_s004
                                 + 1.25253 w_k1_INTC_s004
                                 + 1.28592 w_k1_KO_s004 + 1.28552 w_k1_MAT_s004
                                 + 1.40748 w_k1_MSFT_s004
                                 + 1.23438 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.0265 w_k1_ADM_s005
                                 + 1.11611 w_k1_AXP_s005
                                 + 0.947832 w_k1_DIS_s005
                                 + 0.978339 w_k1_FDX_s005
                                 + 1.24051 w_k1_IBM_s005
                                 + 1.02157 w_k1_INTC_s005
                                 + 1.05753 w_k1_KO_s005 + 1.0829 w_k1_MAT_s005
                                 + 1.01462 w_k1_MSFT_s005
                                 + 1.04688 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.96675 w_k1_ADM_s005
                                 + 1.08896 w_k1_AXP_s005
                                 + 1.00603 w_k1_DIS_s005
                                 + 1.14753 w_k1_FDX_s005
                                 + 0.909664 w_k1_IBM_s005
                                 + 1.2057 w_k1_INTC_s005
                                 + 0.965718 w_k1_KO_s005
                                 + 0.799122 w_k1_MAT_s005
                                 + 1.08987 w_k1_MSFT_s005
                                 + 1.01044 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.905468 w_k1_ADM_s005
                                 + 0.807168 w_k1_AXP_s005
                                 + 0.6179 w_k1_DIS_s005
                                 + 0.844861 w_k1_FDX_s005
                                 + 0.702961 w_k1_IBM_s005
                                 + 1.11171 w_k1_INTC_s005
                                 + 0.799091 w_k1_KO_s005
                                 + 0.46467 w_k1_MAT_s005
                                 + 0.899302 w_k1_MSFT_s005
                                 + 0.787002 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.779097 w_k1_ADM_s005
                                 + 0.799341 w_k1_AXP_s005
                                 + 0.715965 w_k1_DIS_s005
                                 + 0.734772 w_k1_FDX_s005
                                 + 1.24673 w_k1_IBM_s005
                                 + 1.0099 w_k1_INTC_s005
                                 + 0.859777 w_k1_KO_s005
                                 + 0.79624 w_k1_MAT_s005
                                 + 0.728188 w_k1_MSFT_s005
                                 + 0.863402 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.02739 w_k1_ADM_s005
                                 + 1.36493 w_k1_AXP_s005
                                 + 1.30145 w_k1_DIS_s005
                                 + 1.08519 w_k1_FDX_s005
                                 + 1.67507 w_k1_IBM_s005
                                 + 1.23167 w_k1_INTC_s005
                                 + 1.24967 w_k1_KO_s005 + 1.04831 w_k1_MAT_s005
                                 + 1.03372 w_k1_MSFT_s005
                                 + 1.09832 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.39507 w_k1_ADM_s005
                                 + 1.71683 w_k1_AXP_s005
                                 + 1.59539 w_k1_DIS_s005
                                 + 1.29225 w_k1_FDX_s005
                                 + 1.38269 w_k1_IBM_s005
                                 + 1.39842 w_k1_INTC_s005 + 1.4064 w_k1_KO_s005
                                 + 1.48898 w_k1_MAT_s005
                                 + 1.36958 w_k1_MSFT_s005 + 1.1784 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.758939 w_k1_ADM_s005
                                 + 1.0087 w_k1_AXP_s005
                                 + 0.792673 w_k1_DIS_s005
                                 + 0.874811 w_k1_FDX_s005
                                 + 0.972905 w_k1_IBM_s005
                                 + 0.94264 w_k1_INTC_s005
                                 + 0.795651 w_k1_KO_s005
                                 + 0.889484 w_k1_MAT_s005
                                 + 0.787751 w_k1_MSFT_s005
                                 + 0.81542 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.24872 w_k1_ADM_s005
                                 + 1.40722 w_k1_AXP_s005 + 1.5105 w_k1_DIS_s005
                                 + 1.30032 w_k1_FDX_s005
                                 + 1.00766 w_k1_IBM_s005
                                 + 1.34381 w_k1_INTC_s005
                                 + 0.988697 w_k1_KO_s005
                                 + 1.07564 w_k1_MAT_s005
                                 + 1.24292 w_k1_MSFT_s005
                                 + 1.11143 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.10642 w_k1_ADM_s006
                                 + 1.3295 w_k1_AXP_s006 + 1.16265 w_k1_DIS_s006
                                 + 1.07136 w_k1_FDX_s006
                                 + 1.39102 w_k1_IBM_s006
                                 + 0.972902 w_k1_INTC_s006
                                 + 1.29812 w_k1_KO_s006 + 1.65937 w_k1_MAT_s006
                                 + 1.08624 w_k1_MSFT_s006
                                 + 1.10543 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.803385 w_k1_ADM_s006
                                 + 0.824492 w_k1_AXP_s006
                                 + 0.799757 w_k1_DIS_s006
                                 + 0.988021 w_k1_FDX_s006
                                 + 0.678561 w_k1_IBM_s006
                                 + 0.783624 w_k1_INTC_s006
                                 + 0.713473 w_k1_KO_s006
                                 + 0.634301 w_k1_MAT_s006
                                 + 0.809219 w_k1_MSFT_s006 + 0.835 w_k1_PG_s006
                                 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.08821 w_k1_ADM_s006
                                 + 1.12781 w_k1_AXP_s006
                                 + 0.997416 w_k1_DIS_s006
                                 + 1.10933 w_k1_FDX_s006
                                 + 0.995616 w_k1_IBM_s006
                                 + 0.889576 w_k1_INTC_s006
                                 + 0.880206 w_k1_KO_s006
                                 + 0.880447 w_k1_MAT_s006
                                 + 0.876175 w_k1_MSFT_s006
                                 + 1.02707 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.941306 w_k1_ADM_s006
                                 + 0.983034 w_k1_AXP_s006
                                 + 0.731888 w_k1_DIS_s006
                                 + 0.915702 w_k1_FDX_s006
                                 + 1.04375 w_k1_IBM_s006
                                 + 0.967155 w_k1_INTC_s006
                                 + 0.84049 w_k1_KO_s006
                                 + 0.871996 w_k1_MAT_s006
                                 + 0.758501 w_k1_MSFT_s006
                                 + 0.869278 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.31974 w_k1_ADM_s006
                                 + 1.43268 w_k1_AXP_s006
                                 + 1.49427 w_k1_DIS_s006
                                 + 1.35356 w_k1_FDX_s006
                                 + 1.03601 w_k1_IBM_s006
                                 + 1.3307 w_k1_INTC_s006 + 1.34083 w_k1_KO_s006
                                 + 1.3794 w_k1_MAT_s006 + 1.1856 w_k1_MSFT_s006
                                 + 1.14141 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.896845 w_k1_ADM_s006
                                 + 1.56899 w_k1_AXP_s006
                                 + 1.55749 w_k1_DIS_s006 + 1.1246 w_k1_FDX_s006
                                 + 1.97945 w_k1_IBM_s006
                                 + 1.18407 w_k1_INTC_s006
                                 + 1.31009 w_k1_KO_s006 + 2.10338 w_k1_MAT_s006
                                 + 1.17022 w_k1_MSFT_s006
                                 + 0.99904 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.812376 w_k1_ADM_s006
                                 + 0.756842 w_k1_AXP_s006
                                 + 0.974548 w_k1_DIS_s006
                                 + 0.980332 w_k1_FDX_s006
                                 + 1.18152 w_k1_IBM_s006
                                 + 1.03824 w_k1_INTC_s006
                                 + 1.04787 w_k1_KO_s006
                                 + 0.942263 w_k1_MAT_s006
                                 + 0.941623 w_k1_MSFT_s006
                                 + 0.977589 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.941635 w_k1_ADM_s006
                                 + 0.921542 w_k1_AXP_s006
                                 + 0.639531 w_k1_DIS_s006
                                 + 0.786122 w_k1_FDX_s006
                                 + 0.826089 w_k1_IBM_s006
                                 + 0.735238 w_k1_INTC_s006
                                 + 0.820387 w_k1_KO_s006
                                 + 0.98322 w_k1_MAT_s006
                                 + 0.724399 w_k1_MSFT_s006
                                 + 0.843525 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.12517 w_k1_ADM_s007
                                 + 1.14823 w_k1_AXP_s007
                                 + 1.08773 w_k1_DIS_s007
                                 + 1.17893 w_k1_FDX_s007
                                 + 0.983633 w_k1_IBM_s007
                                 + 1.22514 w_k1_INTC_s007
                                 + 1.08062 w_k1_KO_s007 + 1.3006 w_k1_MAT_s007
                                 + 1.07344 w_k1_MSFT_s007
                                 + 1.06531 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.27916 w_k1_ADM_s007
                                 + 1.49302 w_k1_AXP_s007
                                 + 1.79406 w_k1_DIS_s007 + 1.499 w_k1_FDX_s007
                                 + 1.40846 w_k1_IBM_s007
                                 + 1.43055 w_k1_INTC_s007 + 1.6522 w_k1_KO_s007
                                 + 2.25421 w_k1_MAT_s007
                                 + 1.3395 w_k1_MSFT_s007 + 1.28498 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.00307 w_k1_ADM_s007
                                 + 1.0191 w_k1_AXP_s007 + 1.07045 w_k1_DIS_s007
                                 + 1.06739 w_k1_FDX_s007
                                 + 0.783019 w_k1_IBM_s007
                                 + 0.975114 w_k1_INTC_s007
                                 + 0.970236 w_k1_KO_s007
                                 + 0.890341 w_k1_MAT_s007
                                 + 0.974734 w_k1_MSFT_s007
                                 + 0.938569 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.845804 w_k1_ADM_s007
                                 + 1.06104 w_k1_AXP_s007
                                 + 0.696285 w_k1_DIS_s007
                                 + 0.860311 w_k1_FDX_s007
                                 + 0.832067 w_k1_IBM_s007
                                 + 0.994937 w_k1_INTC_s007
                                 + 0.83566 w_k1_KO_s007
                                 + 0.969866 w_k1_MAT_s007
                                 + 0.79166 w_k1_MSFT_s007
                                 + 0.918722 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.997559 w_k1_ADM_s007
                                 + 1.10753 w_k1_AXP_s007
                                 + 1.09227 w_k1_DIS_s007
                                 + 1.03928 w_k1_FDX_s007
                                 + 1.40112 w_k1_IBM_s007
                                 + 1.18753 w_k1_INTC_s007
                                 + 1.40041 w_k1_KO_s007 + 1.74715 w_k1_MAT_s007
                                 + 0.975463 w_k1_MSFT_s007
                                 + 1.14192 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.705636 w_k1_ADM_s007
                                 + 0.782751 w_k1_AXP_s007
                                 + 0.724136 w_k1_DIS_s007
                                 + 0.789815 w_k1_FDX_s007
                                 + 0.792132 w_k1_IBM_s007
                                 + 0.970725 w_k1_INTC_s007
                                 + 1.00884 w_k1_KO_s007
                                 + 0.993267 w_k1_MAT_s007
                                 + 0.906976 w_k1_MSFT_s007
                                 + 0.90936 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.13306 w_k1_ADM_s007
                                 + 0.954741 w_k1_AXP_s007
                                 + 1.38704 w_k1_DIS_s007
                                 + 1.16009 w_k1_FDX_s007
                                 + 1.14252 w_k1_IBM_s007
                                 + 1.19013 w_k1_INTC_s007
                                 + 1.25762 w_k1_KO_s007 + 1.21998 w_k1_MAT_s007
                                 + 1.24375 w_k1_MSFT_s007
                                 + 1.14866 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.970708 w_k1_ADM_s007
                                 + 0.880946 w_k1_AXP_s007
                                 + 1.05325 w_k1_DIS_s007
                                 + 0.951261 w_k1_FDX_s007
                                 + 1.1046 w_k1_IBM_s007
                                 + 1.02051 w_k1_INTC_s007
                                 + 1.09769 w_k1_KO_s007 + 1.29554 w_k1_MAT_s007
                                 + 1.02193 w_k1_MSFT_s007
                                 + 0.969526 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.843015 w_k1_ADM_s010
                                 + 0.937866 w_k1_AXP_s010
                                 + 0.697114 w_k1_DIS_s010
                                 + 0.835509 w_k1_FDX_s010
                                 + 0.908159 w_k1_IBM_s010
                                 + 1.12667 w_k1_INTC_s010
                                 + 0.847269 w_k1_KO_s010
                                 + 1.04779 w_k1_MAT_s010
                                 + 0.973387 w_k1_MSFT_s010
                                 + 0.807158 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.952516 w_k1_ADM_s010
                                 + 0.815953 w_k1_AXP_s010
                                 + 0.703722 w_k1_DIS_s010
                                 + 0.824359 w_k1_FDX_s010
                                 + 0.641173 w_k1_IBM_s010
                                 + 0.994232 w_k1_INTC_s010
                                 + 0.745095 w_k1_KO_s010
                                 + 0.550559 w_k1_MAT_s010
                                 + 1.04988 w_k1_MSFT_s010
                                 + 0.876292 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.32126 w_k1_ADM_s010
                                 + 1.13576 w_k1_AXP_s010
                                 + 1.15458 w_k1_DIS_s010
                                 + 1.07304 w_k1_FDX_s010
                                 + 1.20782 w_k1_IBM_s010
                                 + 0.92552 w_k1_INTC_s010
                                 + 1.20382 w_k1_KO_s010 + 1.80097 w_k1_MAT_s010
                                 + 1.08452 w_k1_MSFT_s010
                                 + 1.09874 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.14043 w_k1_ADM_s010
                                 + 0.736525 w_k1_AXP_s010
                                 + 0.871666 w_k1_DIS_s010
                                 + 0.860548 w_k1_FDX_s010
                                 + 0.965255 w_k1_IBM_s010
                                 + 0.891067 w_k1_INTC_s010
                                 + 1.05278 w_k1_KO_s010
                                 + 0.870576 w_k1_MAT_s010
                                 + 0.93666 w_k1_MSFT_s010
                                 + 0.914331 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.3648 w_k1_ADM_s010
                                 + 1.72716 w_k1_AXP_s010
                                 + 1.71467 w_k1_DIS_s010
                                 + 1.26415 w_k1_FDX_s010
                                 + 1.46519 w_k1_IBM_s010
                                 + 1.26195 w_k1_INTC_s010
                                 + 1.49551 w_k1_KO_s010 + 2.47001 w_k1_MAT_s010
                                 + 1.2706 w_k1_MSFT_s010 + 1.20304 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.16168 w_k1_ADM_s010
                                 + 1.12153 w_k1_AXP_s010
                                 + 1.09344 w_k1_DIS_s010
                                 + 1.03931 w_k1_FDX_s010
                                 + 1.60377 w_k1_IBM_s010
                                 + 0.980966 w_k1_INTC_s010
                                 + 1.32156 w_k1_KO_s010 + 1.89583 w_k1_MAT_s010
                                 + 1.04058 w_k1_MSFT_s010
                                 + 1.15071 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.26255 w_k1_ADM_s010
                                 + 0.95579 w_k1_AXP_s010
                                 + 1.34855 w_k1_DIS_s010 + 1.2624 w_k1_FDX_s010
                                 + 0.808861 w_k1_IBM_s010
                                 + 1.48463 w_k1_INTC_s010
                                 + 1.18186 w_k1_KO_s010
                                 + 0.784873 w_k1_MAT_s010
                                 + 1.45264 w_k1_MSFT_s010
                                 + 1.10507 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.33109 w_k1_ADM_s010
                                 + 1.24274 w_k1_AXP_s010
                                 + 1.10949 w_k1_DIS_s010
                                 + 1.20005 w_k1_FDX_s010
                                 + 0.88774 w_k1_IBM_s010
                                 + 1.36573 w_k1_INTC_s010 + 1.0659 w_k1_KO_s010
                                 + 0.955438 w_k1_MAT_s010
                                 + 1.30285 w_k1_MSFT_s010
                                 + 1.00182 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.906356 w_k1_ADM_s011
                                 + 0.837657 w_k1_AXP_s011
                                 + 0.834804 w_k1_DIS_s011
                                 + 0.793738 w_k1_FDX_s011
                                 + 1.36675 w_k1_IBM_s011
                                 + 0.889211 w_k1_INTC_s011
                                 + 1.0738 w_k1_KO_s011 + 1.31278 w_k1_MAT_s011
                                 + 0.819137 w_k1_MSFT_s011
                                 + 0.945037 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.3087 w_k1_ADM_s011
                                 + 1.16276 w_k1_AXP_s011
                                 + 1.30616 w_k1_DIS_s011
                                 + 1.12154 w_k1_FDX_s011
                                 + 1.26799 w_k1_IBM_s011
                                 + 1.10489 w_k1_INTC_s011
                                 + 1.27615 w_k1_KO_s011 + 1.58137 w_k1_MAT_s011
                                 + 1.16593 w_k1_MSFT_s011
                                 + 1.06497 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.27229 w_k1_ADM_s011
                                 + 1.21993 w_k1_AXP_s011 + 1.6192 w_k1_DIS_s011
                                 + 1.20202 w_k1_FDX_s011
                                 + 1.72907 w_k1_IBM_s011
                                 + 1.21341 w_k1_INTC_s011
                                 + 1.51818 w_k1_KO_s011 + 1.60547 w_k1_MAT_s011
                                 + 1.32608 w_k1_MSFT_s011
                                 + 1.27012 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.37923 w_k1_ADM_s011
                                 + 0.964049 w_k1_AXP_s011
                                 + 1.1194 w_k1_DIS_s011 + 1.11267 w_k1_FDX_s011
                                 + 0.842749 w_k1_IBM_s011
                                 + 1.13829 w_k1_INTC_s011
                                 + 1.13255 w_k1_KO_s011
                                 + 0.691784 w_k1_MAT_s011
                                 + 1.19158 w_k1_MSFT_s011
                                 + 0.969442 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.52054 w_k1_ADM_s011
                                 + 1.38007 w_k1_AXP_s011
                                 + 1.39886 w_k1_DIS_s011
                                 + 1.29226 w_k1_FDX_s011 + 1.007 w_k1_IBM_s011
                                 + 1.01256 w_k1_INTC_s011
                                 + 1.23398 w_k1_KO_s011 + 1.36461 w_k1_MAT_s011
                                 + 1.26668 w_k1_MSFT_s011
                                 + 1.07973 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.31285 w_k1_ADM_s011
                                 + 0.976705 w_k1_AXP_s011
                                 + 0.909576 w_k1_DIS_s011
                                 + 0.931784 w_k1_FDX_s011
                                 + 1.04848 w_k1_IBM_s011
                                 + 0.700836 w_k1_INTC_s011
                                 + 1.05965 w_k1_KO_s011 + 1.15195 w_k1_MAT_s011
                                 + 0.861452 w_k1_MSFT_s011
                                 + 1.02697 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.956894 w_k1_ADM_s011
                                 + 0.64626 w_k1_AXP_s011 + 0.7499 w_k1_DIS_s011
                                 + 0.687545 w_k1_FDX_s011
                                 + 0.977186 w_k1_IBM_s011
                                 + 0.783384 w_k1_INTC_s011
                                 + 0.954182 w_k1_KO_s011
                                 + 0.828225 w_k1_MAT_s011
                                 + 0.805151 w_k1_MSFT_s011
                                 + 0.809889 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.09323 w_k1_ADM_s011
                                 + 0.725333 w_k1_AXP_s011
                                 + 0.73926 w_k1_DIS_s011
                                 + 0.865283 w_k1_FDX_s011
                                 + 0.661681 w_k1_IBM_s011
                                 + 0.877503 w_k1_INTC_s011
                                 + 0.703841 w_k1_KO_s011
                                 + 0.640733 w_k1_MAT_s011
                                 + 0.808293 w_k1_MSFT_s011
                                 + 0.98843 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.12054 w_k1_ADM_s012
                                 + 1.19864 w_k1_AXP_s012
                                 + 1.34804 w_k1_DIS_s012
                                 + 1.31815 w_k1_FDX_s012 + 1.1017 w_k1_IBM_s012
                                 + 1.22679 w_k1_INTC_s012
                                 + 1.14585 w_k1_KO_s012
                                 + 0.966133 w_k1_MAT_s012
                                 + 1.18459 w_k1_MSFT_s012
                                 + 1.06974 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.893106 w_k1_ADM_s012
                                 + 0.914806 w_k1_AXP_s012
                                 + 0.959611 w_k1_DIS_s012
                                 + 1.00473 w_k1_FDX_s012
                                 + 0.904845 w_k1_IBM_s012
                                 + 0.888221 w_k1_INTC_s012
                                 + 0.940424 w_k1_KO_s012
                                 + 0.950167 w_k1_MAT_s012
                                 + 0.885024 w_k1_MSFT_s012
                                 + 0.881816 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.8775 w_k1_ADM_s012
                                 + 0.769689 w_k1_AXP_s012
                                 + 0.77784 w_k1_DIS_s012
                                 + 0.834673 w_k1_FDX_s012
                                 + 0.67527 w_k1_IBM_s012
                                 + 1.07907 w_k1_INTC_s012
                                 + 0.828383 w_k1_KO_s012
                                 + 0.368087 w_k1_MAT_s012
                                 + 0.817313 w_k1_MSFT_s012
                                 + 0.859204 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.958921 w_k1_ADM_s012
                                 + 0.976821 w_k1_AXP_s012
                                 + 0.826422 w_k1_DIS_s012
                                 + 0.835167 w_k1_FDX_s012
                                 + 1.14446 w_k1_IBM_s012
                                 + 1.10888 w_k1_INTC_s012
                                 + 1.17755 w_k1_KO_s012 + 1.10893 w_k1_MAT_s012
                                 + 0.916419 w_k1_MSFT_s012
                                 + 0.912086 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.00072 w_k1_ADM_s012
                                 + 1.40781 w_k1_AXP_s012
                                 + 1.15249 w_k1_DIS_s012 + 1.117 w_k1_FDX_s012
                                 + 1.50562 w_k1_IBM_s012
                                 + 1.31709 w_k1_INTC_s012
                                 + 1.32589 w_k1_KO_s012 + 1.30691 w_k1_MAT_s012
                                 + 0.944576 w_k1_MSFT_s012
                                 + 0.960409 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.04661 w_k1_ADM_s012
                                 + 1.09328 w_k1_AXP_s012
                                 + 0.818091 w_k1_DIS_s012
                                 + 0.967608 w_k1_FDX_s012
                                 + 1.42329 w_k1_IBM_s012
                                 + 0.965199 w_k1_INTC_s012
                                 + 1.27316 w_k1_KO_s012 + 1.12527 w_k1_MAT_s012
                                 + 0.840942 w_k1_MSFT_s012
                                 + 0.997094 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.2674 w_k1_ADM_s012
                                 + 1.03988 w_k1_AXP_s012
                                 + 1.26486 w_k1_DIS_s012
                                 + 1.22187 w_k1_FDX_s012
                                 + 0.76241 w_k1_IBM_s012
                                 + 0.987346 w_k1_INTC_s012
                                 + 0.984651 w_k1_KO_s012
                                 + 0.674256 w_k1_MAT_s012
                                 + 1.0079 w_k1_MSFT_s012 + 1.00882 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.959965 w_k1_ADM_s012
                                 + 0.933216 w_k1_AXP_s012
                                 + 0.778838 w_k1_DIS_s012
                                 + 0.913428 w_k1_FDX_s012
                                 + 0.908045 w_k1_IBM_s012
                                 + 1.14913 w_k1_INTC_s012
                                 + 1.01287 w_k1_KO_s012 + 0.87503 w_k1_MAT_s012
                                 + 0.896263 w_k1_MSFT_s012
                                 + 0.911449 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.03629 w_k1_ADM_s013
                                 + 1.04322 w_k1_AXP_s013
                                 + 0.949158 w_k1_DIS_s013
                                 + 0.907958 w_k1_FDX_s013
                                 + 1.47918 w_k1_IBM_s013
                                 + 1.01336 w_k1_INTC_s013 + 1.2268 w_k1_KO_s013
                                 + 2.05522 w_k1_MAT_s013
                                 + 1.09563 w_k1_MSFT_s013
                                 + 1.02379 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.876619 w_k1_ADM_s013
                                 + 0.758714 w_k1_AXP_s013
                                 + 0.903307 w_k1_DIS_s013
                                 + 0.930092 w_k1_FDX_s013
                                 + 0.648075 w_k1_IBM_s013
                                 + 0.897247 w_k1_INTC_s013
                                 + 0.826518 w_k1_KO_s013
                                 + 0.718934 w_k1_MAT_s013
                                 + 1.15458 w_k1_MSFT_s013
                                 + 0.895539 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.22367 w_k1_ADM_s013
                                 + 1.52718 w_k1_AXP_s013
                                 + 1.46666 w_k1_DIS_s013 + 1.2504 w_k1_FDX_s013
                                 + 1.87419 w_k1_IBM_s013
                                 + 1.39486 w_k1_INTC_s013
                                 + 1.36154 w_k1_KO_s013 + 1.99516 w_k1_MAT_s013
                                 + 1.37104 w_k1_MSFT_s013
                                 + 1.29044 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.47738 w_k1_ADM_s013
                                 + 1.65004 w_k1_AXP_s013
                                 + 1.62823 w_k1_DIS_s013
                                 + 1.45529 w_k1_FDX_s013
                                 + 1.13611 w_k1_IBM_s013
                                 + 1.17565 w_k1_INTC_s013
                                 + 1.27017 w_k1_KO_s013 + 1.3846 w_k1_MAT_s013
                                 + 1.62141 w_k1_MSFT_s013
                                 + 1.30354 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.13463 w_k1_ADM_s013
                                 + 0.982546 w_k1_AXP_s013
                                 + 0.817497 w_k1_DIS_s013
                                 + 0.888778 w_k1_FDX_s013
                                 + 1.0584 w_k1_IBM_s013
                                 + 0.96035 w_k1_INTC_s013
                                 + 0.91433 w_k1_KO_s013 + 1.27977 w_k1_MAT_s013
                                 + 1.03375 w_k1_MSFT_s013
                                 + 0.99577 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.00613 w_k1_ADM_s013
                                 + 1.26602 w_k1_AXP_s013
                                 + 0.733734 w_k1_DIS_s013
                                 + 0.963679 w_k1_FDX_s013
                                 + 0.882471 w_k1_IBM_s013
                                 + 0.940271 w_k1_INTC_s013
                                 + 0.773414 w_k1_KO_s013
                                 + 1.21663 w_k1_MAT_s013
                                 + 0.959061 w_k1_MSFT_s013
                                 + 0.937071 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.09401 w_k1_ADM_s013
                                 + 1.34379 w_k1_AXP_s013
                                 + 1.33968 w_k1_DIS_s013
                                 + 1.27286 w_k1_FDX_s013
                                 + 1.09249 w_k1_IBM_s013
                                 + 1.12353 w_k1_INTC_s013
                                 + 1.23891 w_k1_KO_s013 + 1.75259 w_k1_MAT_s013
                                 + 1.32065 w_k1_MSFT_s013
                                 + 1.20239 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.11055 w_k1_ADM_s013
                                 + 1.04393 w_k1_AXP_s013
                                 + 1.08782 w_k1_DIS_s013
                                 + 1.04558 w_k1_FDX_s013
                                 + 1.00865 w_k1_IBM_s013
                                 + 1.02265 w_k1_INTC_s013 + 1.1254 w_k1_KO_s013
                                 + 0.917094 w_k1_MAT_s013
                                 + 1.2101 w_k1_MSFT_s013
                                 + 0.987084 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.807007 w_k1_ADM_s014
                                 + 0.950848 w_k1_AXP_s014
                                 + 1.11285 w_k1_DIS_s014
                                 + 1.07134 w_k1_FDX_s014
                                 + 0.746108 w_k1_IBM_s014
                                 + 0.871908 w_k1_INTC_s014
                                 + 0.885608 w_k1_KO_s014
                                 + 0.823376 w_k1_MAT_s014
                                 + 1.07299 w_k1_MSFT_s014
                                 + 0.955163 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.730036 w_k1_ADM_s014
                                 + 0.709478 w_k1_AXP_s014
                                 + 0.782242 w_k1_DIS_s014
                                 + 0.718087 w_k1_FDX_s014
                                 + 0.882829 w_k1_IBM_s014
                                 + 0.762239 w_k1_INTC_s014
                                 + 1.0096 w_k1_KO_s014 + 0.947101 w_k1_MAT_s014
                                 + 0.8865 w_k1_MSFT_s014
                                 + 0.910734 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.473345 w_k1_ADM_s014
                                 + 0.622659 w_k1_AXP_s014
                                 + 0.804284 w_k1_DIS_s014
                                 + 0.817869 w_k1_FDX_s014
                                 + 0.569093 w_k1_IBM_s014
                                 + 0.717071 w_k1_INTC_s014
                                 + 0.790277 w_k1_KO_s014
                                 + 0.716579 w_k1_MAT_s014
                                 + 0.940101 w_k1_MSFT_s014
                                 + 0.933444 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.82065 w_k1_ADM_s014
                                 + 0.773575 w_k1_AXP_s014
                                 + 0.618936 w_k1_DIS_s014
                                 + 0.795407 w_k1_FDX_s014
                                 + 0.703912 w_k1_IBM_s014
                                 + 0.906945 w_k1_INTC_s014
                                 + 0.730448 w_k1_KO_s014
                                 + 0.645009 w_k1_MAT_s014
                                 + 0.798682 w_k1_MSFT_s014
                                 + 0.788721 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.883062 w_k1_ADM_s014
                                 + 1.26407 w_k1_AXP_s014
                                 + 1.39843 w_k1_DIS_s014
                                 + 1.11873 w_k1_FDX_s014
                                 + 1.32102 w_k1_IBM_s014
                                 + 1.09594 w_k1_INTC_s014
                                 + 1.32254 w_k1_KO_s014 + 1.49034 w_k1_MAT_s014
                                 + 1.17357 w_k1_MSFT_s014
                                 + 1.12539 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.958401 w_k1_ADM_s014
                                 + 0.974176 w_k1_AXP_s014
                                 + 0.792905 w_k1_DIS_s014
                                 + 0.917597 w_k1_FDX_s014
                                 + 0.737619 w_k1_IBM_s014
                                 + 0.804921 w_k1_INTC_s014
                                 + 0.826839 w_k1_KO_s014
                                 + 0.540645 w_k1_MAT_s014
                                 + 0.872591 w_k1_MSFT_s014
                                 + 0.939898 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.865674 w_k1_ADM_s014
                                 + 1.47072 w_k1_AXP_s014
                                 + 1.47676 w_k1_DIS_s014
                                 + 1.38786 w_k1_FDX_s014
                                 + 1.01566 w_k1_IBM_s014
                                 + 1.46494 w_k1_INTC_s014
                                 + 1.20739 w_k1_KO_s014 + 1.41086 w_k1_MAT_s014
                                 + 1.40496 w_k1_MSFT_s014
                                 + 1.25049 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.683166 w_k1_ADM_s014
                                 + 1.16288 w_k1_AXP_s014
                                 + 0.955659 w_k1_DIS_s014
                                 + 0.866882 w_k1_FDX_s014
                                 + 1.18574 w_k1_IBM_s014
                                 + 0.942428 w_k1_INTC_s014
                                 + 1.04671 w_k1_KO_s014 + 1.27022 w_k1_MAT_s014
                                 + 0.930348 w_k1_MSFT_s014
                                 + 0.931619 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.985914 w_k1_ADM_s015
                                 + 1.0044 w_k1_AXP_s015 + 0.77145 w_k1_DIS_s015
                                 + 0.829609 w_k1_FDX_s015
                                 + 1.25332 w_k1_IBM_s015
                                 + 0.976857 w_k1_INTC_s015
                                 + 0.9811 w_k1_KO_s015 + 1.15217 w_k1_MAT_s015
                                 + 0.815922 w_k1_MSFT_s015
                                 + 0.976562 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.42388 w_k1_ADM_s015
                                 + 1.33438 w_k1_AXP_s015
                                 + 1.52872 w_k1_DIS_s015
                                 + 1.43071 w_k1_FDX_s015
                                 + 1.14395 w_k1_IBM_s015
                                 + 1.53964 w_k1_INTC_s015
                                 + 1.35003 w_k1_KO_s015 + 1.22669 w_k1_MAT_s015
                                 + 1.28699 w_k1_MSFT_s015
                                 + 1.21595 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.32973 w_k1_ADM_s015
                                 + 0.93788 w_k1_AXP_s015
                                 + 1.02029 w_k1_DIS_s015
                                 + 1.01494 w_k1_FDX_s015
                                 + 1.09999 w_k1_IBM_s015
                                 + 1.16139 w_k1_INTC_s015
                                 + 1.20179 w_k1_KO_s015 + 1.09535 w_k1_MAT_s015
                                 + 1.13732 w_k1_MSFT_s015
                                 + 1.04108 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.12582 w_k1_ADM_s015
                                 + 1.1481 w_k1_AXP_s015
                                 + 0.959787 w_k1_DIS_s015
                                 + 1.09955 w_k1_FDX_s015
                                 + 0.890146 w_k1_IBM_s015
                                 + 1.31029 w_k1_INTC_s015
                                 + 0.982125 w_k1_KO_s015
                                 + 0.927647 w_k1_MAT_s015
                                 + 1.13101 w_k1_MSFT_s015 + 1.1148 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.119 w_k1_ADM_s015
                                 + 1.08046 w_k1_AXP_s015
                                 + 1.25339 w_k1_DIS_s015
                                 + 0.988038 w_k1_FDX_s015
                                 + 1.86492 w_k1_IBM_s015
                                 + 1.23593 w_k1_INTC_s015
                                 + 1.58023 w_k1_KO_s015 + 1.79456 w_k1_MAT_s015
                                 + 1.16432 w_k1_MSFT_s015
                                 + 1.03355 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.2302 w_k1_ADM_s015
                                 + 1.17137 w_k1_AXP_s015
                                 + 1.39776 w_k1_DIS_s015
                                 + 1.22035 w_k1_FDX_s015 + 1.083 w_k1_IBM_s015
                                 + 1.35437 w_k1_INTC_s015
                                 + 1.31233 w_k1_KO_s015 + 1.54203 w_k1_MAT_s015
                                 + 1.25889 w_k1_MSFT_s015 + 1.0542 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.81365 w_k1_ADM_s015
                                 + 0.662709 w_k1_AXP_s015
                                 + 0.539427 w_k1_DIS_s015
                                 + 0.672562 w_k1_FDX_s015
                                 + 0.980148 w_k1_IBM_s015
                                 + 0.82942 w_k1_INTC_s015
                                 + 0.897841 w_k1_KO_s015
                                 + 0.795746 w_k1_MAT_s015
                                 + 0.69994 w_k1_MSFT_s015
                                 + 0.796062 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.01315 w_k1_ADM_s015
                                 + 0.933034 w_k1_AXP_s015
                                 + 0.859451 w_k1_DIS_s015
                                 + 0.960759 w_k1_FDX_s015
                                 + 0.708895 w_k1_IBM_s015
                                 + 1.12582 w_k1_INTC_s015
                                 + 0.80456 w_k1_KO_s015
                                 + 0.762127 w_k1_MAT_s015
                                 + 0.97042 w_k1_MSFT_s015
                                 + 0.822948 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.932085 w_k1_ADM_s016
                                 + 0.897261 w_k1_AXP_s016
                                 + 1.00718 w_k1_DIS_s016
                                 + 1.01396 w_k1_FDX_s016
                                 + 1.14895 w_k1_IBM_s016
                                 + 1.01292 w_k1_INTC_s016
                                 + 0.906354 w_k1_KO_s016
                                 + 1.05944 w_k1_MAT_s016
                                 + 0.893548 w_k1_MSFT_s016
                                 + 1.12796 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.22727 w_k1_ADM_s016
                                 + 1.18625 w_k1_AXP_s016
                                 + 1.24897 w_k1_DIS_s016
                                 + 1.27523 w_k1_FDX_s016
                                 + 1.19976 w_k1_IBM_s016
                                 + 0.886103 w_k1_INTC_s016
                                 + 1.0335 w_k1_KO_s016 + 1.26372 w_k1_MAT_s016
                                 + 1.08781 w_k1_MSFT_s016
                                 + 1.02801 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.27532 w_k1_ADM_s016
                                 + 1.43275 w_k1_AXP_s016
                                 + 1.93312 w_k1_DIS_s016
                                 + 1.55869 w_k1_FDX_s016
                                 + 1.04759 w_k1_IBM_s016
                                 + 1.01477 w_k1_INTC_s016
                                 + 1.22885 w_k1_KO_s016 + 1.33666 w_k1_MAT_s016
                                 + 1.5596 w_k1_MSFT_s016 + 1.23244 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.36394 w_k1_ADM_s016
                                 + 0.777302 w_k1_AXP_s016
                                 + 0.965114 w_k1_DIS_s016
                                 + 0.798027 w_k1_FDX_s016
                                 + 1.67154 w_k1_IBM_s016
                                 + 0.661207 w_k1_INTC_s016
                                 + 0.988242 w_k1_KO_s016 + 1.2012 w_k1_MAT_s016
                                 + 0.957051 w_k1_MSFT_s016
                                 + 1.01102 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.21604 w_k1_ADM_s016
                                 + 0.740269 w_k1_AXP_s016
                                 + 0.87167 w_k1_DIS_s016
                                 + 0.921675 w_k1_FDX_s016
                                 + 0.720304 w_k1_IBM_s016
                                 + 0.75244 w_k1_INTC_s016
                                 + 0.755806 w_k1_KO_s016
                                 + 0.684695 w_k1_MAT_s016
                                 + 0.998964 w_k1_MSFT_s016
                                 + 0.943112 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.836389 w_k1_ADM_s016
                                 + 0.745717 w_k1_AXP_s016
                                 + 0.748995 w_k1_DIS_s016
                                 + 0.820592 w_k1_FDX_s016
                                 + 1.10101 w_k1_IBM_s016
                                 + 0.685019 w_k1_INTC_s016
                                 + 0.789532 w_k1_KO_s016
                                 + 0.866948 w_k1_MAT_s016
                                 + 0.755283 w_k1_MSFT_s016
                                 + 0.807408 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.52119 w_k1_ADM_s016
                                 + 0.863876 w_k1_AXP_s016
                                 + 1.36468 w_k1_DIS_s016
                                 + 1.12932 w_k1_FDX_s016
                                 + 0.956747 w_k1_IBM_s016
                                 + 1.19559 w_k1_INTC_s016
                                 + 1.12547 w_k1_KO_s016
                                 + 0.907087 w_k1_MAT_s016
                                 + 1.36971 w_k1_MSFT_s016
                                 + 1.14038 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.32261 w_k1_ADM_s016
                                 + 1.32777 w_k1_AXP_s016
                                 + 1.63522 w_k1_DIS_s016 + 1.3358 w_k1_FDX_s016
                                 + 1.59457 w_k1_IBM_s016
                                 + 1.17297 w_k1_INTC_s016
                                 + 1.12925 w_k1_KO_s016 + 1.93703 w_k1_MAT_s016
                                 + 1.3803 w_k1_MSFT_s016 + 1.35125 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.974912 w_k1_ADM_s017
                                 + 1.1993 w_k1_AXP_s017
                                 + 0.903033 w_k1_DIS_s017
                                 + 0.896357 w_k1_FDX_s017
                                 + 1.38388 w_k1_IBM_s017
                                 + 0.93881 w_k1_INTC_s017 + 1.0379 w_k1_KO_s017
                                 + 1.33767 w_k1_MAT_s017
                                 + 0.851086 w_k1_MSFT_s017
                                 + 1.02739 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.41345 w_k1_ADM_s017
                                 + 1.75126 w_k1_AXP_s017
                                 + 1.41558 w_k1_DIS_s017 + 1.2878 w_k1_FDX_s017
                                 + 1.541 w_k1_IBM_s017 + 1.05866 w_k1_INTC_s017
                                 + 1.33074 w_k1_KO_s017 + 2.07733 w_k1_MAT_s017
                                 + 1.08375 w_k1_MSFT_s017
                                 + 1.21951 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.656102 w_k1_ADM_s017
                                 + 0.872265 w_k1_AXP_s017
                                 + 0.696569 w_k1_DIS_s017
                                 + 0.874718 w_k1_FDX_s017
                                 + 0.694757 w_k1_IBM_s017
                                 + 1.0183 w_k1_INTC_s017
                                 + 0.645381 w_k1_KO_s017
                                 + 0.613003 w_k1_MAT_s017
                                 + 0.863952 w_k1_MSFT_s017
                                 + 0.862685 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.897109 w_k1_ADM_s017
                                 + 1.07436 w_k1_AXP_s017
                                 + 1.26407 w_k1_DIS_s017
                                 + 1.09082 w_k1_FDX_s017
                                 + 1.23649 w_k1_IBM_s017
                                 + 1.19856 w_k1_INTC_s017
                                 + 1.24451 w_k1_KO_s017 + 1.31652 w_k1_MAT_s017
                                 + 1.22957 w_k1_MSFT_s017 + 1.1141 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.875933 w_k1_ADM_s017
                                 + 0.742463 w_k1_AXP_s017
                                 + 0.613216 w_k1_DIS_s017
                                 + 0.661458 w_k1_FDX_s017
                                 + 0.734341 w_k1_IBM_s017
                                 + 0.764426 w_k1_INTC_s017
                                 + 0.724266 w_k1_KO_s017
                                 + 0.604078 w_k1_MAT_s017
                                 + 0.725073 w_k1_MSFT_s017
                                 + 0.890608 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.06659 w_k1_ADM_s017
                                 + 0.827652 w_k1_AXP_s017
                                 + 0.820692 w_k1_DIS_s017
                                 + 0.869104 w_k1_FDX_s017
                                 + 0.98614 w_k1_IBM_s017
                                 + 1.00177 w_k1_INTC_s017
                                 + 0.963919 w_k1_KO_s017
                                 + 0.960863 w_k1_MAT_s017
                                 + 0.878947 w_k1_MSFT_s017
                                 + 0.981937 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.12601 w_k1_ADM_s017
                                 + 1.17986 w_k1_AXP_s017
                                 + 1.03281 w_k1_DIS_s017
                                 + 1.11413 w_k1_FDX_s017
                                 + 0.813878 w_k1_IBM_s017
                                 + 1.17216 w_k1_INTC_s017
                                 + 0.926953 w_k1_KO_s017
                                 + 1.11309 w_k1_MAT_s017
                                 + 1.19955 w_k1_MSFT_s017
                                 + 1.05837 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.35784 w_k1_ADM_s017
                                 + 1.56557 w_k1_AXP_s017
                                 + 1.49493 w_k1_DIS_s017
                                 + 1.32099 w_k1_FDX_s017
                                 + 0.95839 w_k1_IBM_s017
                                 + 1.4421 w_k1_INTC_s017 + 1.17875 w_k1_KO_s017
                                 + 1.15167 w_k1_MAT_s017
                                 + 1.34773 w_k1_MSFT_s017
                                 + 1.00013 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.998851 w_k1_ADM_s100
                                 + 0.820069 w_k1_AXP_s100
                                 + 0.810612 w_k1_DIS_s100
                                 + 0.810813 w_k1_FDX_s100
                                 + 0.807622 w_k1_IBM_s100
                                 + 0.924683 w_k1_INTC_s100
                                 + 0.870929 w_k1_KO_s100
                                 + 0.819812 w_k1_MAT_s100
                                 + 0.891177 w_k1_MSFT_s100
                                 + 0.864456 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.970974 w_k1_ADM_s100
                                 + 0.797787 w_k1_AXP_s100
                                 + 0.715755 w_k1_DIS_s100
                                 + 0.792292 w_k1_FDX_s100
                                 + 0.599841 w_k1_IBM_s100
                                 + 0.837123 w_k1_INTC_s100
                                 + 0.686502 w_k1_KO_s100
                                 + 0.586842 w_k1_MAT_s100
                                 + 0.846504 w_k1_MSFT_s100
                                 + 0.759797 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.07427 w_k1_ADM_s100
                                 + 1.28034 w_k1_AXP_s100
                                 + 0.894748 w_k1_DIS_s100
                                 + 0.9272 w_k1_FDX_s100 + 1.16894 w_k1_IBM_s100
                                 + 1.2564 w_k1_INTC_s100 + 1.05102 w_k1_KO_s100
                                 + 1.20991 w_k1_MAT_s100
                                 + 1.02237 w_k1_MSFT_s100
                                 + 1.00414 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.786107 w_k1_ADM_s100
                                 + 0.83889 w_k1_AXP_s100
                                 + 0.679633 w_k1_DIS_s100
                                 + 0.776659 w_k1_FDX_s100
                                 + 1.02132 w_k1_IBM_s100
                                 + 0.929306 w_k1_INTC_s100
                                 + 0.816179 w_k1_KO_s100
                                 + 0.85111 w_k1_MAT_s100
                                 + 0.790949 w_k1_MSFT_s100
                                 + 0.873791 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.906355 w_k1_ADM_s100
                                 + 0.97136 w_k1_AXP_s100
                                 + 1.05739 w_k1_DIS_s100
                                 + 1.02571 w_k1_FDX_s100
                                 + 0.759819 w_k1_IBM_s100
                                 + 1.26049 w_k1_INTC_s100
                                 + 0.913123 w_k1_KO_s100
                                 + 0.720865 w_k1_MAT_s100
                                 + 0.984927 w_k1_MSFT_s100
                                 + 0.890806 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.48678 w_k1_ADM_s100
                                 + 1.40372 w_k1_AXP_s100
                                 + 1.54862 w_k1_DIS_s100
                                 + 1.29008 w_k1_FDX_s100
                                 + 0.967955 w_k1_IBM_s100
                                 + 1.54524 w_k1_INTC_s100 + 1.245 w_k1_KO_s100
                                 + 0.645218 w_k1_MAT_s100
                                 + 1.43391 w_k1_MSFT_s100
                                 + 1.06216 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.939806 w_k1_ADM_s100
                                 + 0.795478 w_k1_AXP_s100
                                 + 1.09891 w_k1_DIS_s100
                                 + 0.854886 w_k1_FDX_s100
                                 + 1.1411 w_k1_IBM_s100
                                 + 1.11529 w_k1_INTC_s100
                                 + 1.21819 w_k1_KO_s100 + 1.22379 w_k1_MAT_s100
                                 + 1.04701 w_k1_MSFT_s100 + 0.9074 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.98703 w_k1_ADM_s100
                                 + 1.33377 w_k1_AXP_s100
                                 + 1.24439 w_k1_DIS_s100
                                 + 1.05699 w_k1_FDX_s100 + 1.4655 w_k1_IBM_s100
                                 + 1.53116 w_k1_INTC_s100
                                 + 1.34982 w_k1_KO_s100 + 1.5193 w_k1_MAT_s100
                                 + 1.17655 w_k1_MSFT_s100
                                 + 1.15354 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.927436 w_k1_ADM_s101
                                 + 0.828799 w_k1_AXP_s101
                                 + 0.904788 w_k1_DIS_s101
                                 + 0.787624 w_k1_FDX_s101
                                 + 1.31356 w_k1_IBM_s101
                                 + 0.732611 w_k1_INTC_s101
                                 + 1.04421 w_k1_KO_s101 + 1.03584 w_k1_MAT_s101
                                 + 0.828325 w_k1_MSFT_s101
                                 + 1.00051 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.04054 w_k1_ADM_s101
                                 + 1.41641 w_k1_AXP_s101
                                 + 1.50071 w_k1_DIS_s101
                                 + 1.21214 w_k1_FDX_s101
                                 + 1.23685 w_k1_IBM_s101
                                 + 1.33126 w_k1_INTC_s101
                                 + 1.22023 w_k1_KO_s101 + 1.19642 w_k1_MAT_s101
                                 + 1.3787 w_k1_MSFT_s101 + 1.13162 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.12669 w_k1_ADM_s101
                                 + 1.08837 w_k1_AXP_s101 + 1.0045 w_k1_DIS_s101
                                 + 1.01669 w_k1_FDX_s101
                                 + 1.20898 w_k1_IBM_s101
                                 + 1.10152 w_k1_INTC_s101
                                 + 1.05804 w_k1_KO_s101 + 1.08166 w_k1_MAT_s101
                                 + 1.02746 w_k1_MSFT_s101
                                 + 1.03007 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.03023 w_k1_ADM_s101
                                 + 0.841615 w_k1_AXP_s101
                                 + 1.05706 w_k1_DIS_s101
                                 + 0.884776 w_k1_FDX_s101
                                 + 0.956846 w_k1_IBM_s101
                                 + 0.797704 w_k1_INTC_s101
                                 + 0.943227 w_k1_KO_s101
                                 + 0.579316 w_k1_MAT_s101
                                 + 1.07585 w_k1_MSFT_s101
                                 + 1.01032 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.06534 w_k1_ADM_s101
                                 + 1.25955 w_k1_AXP_s101
                                 + 1.31423 w_k1_DIS_s101
                                 + 1.24437 w_k1_FDX_s101
                                 + 0.868488 w_k1_IBM_s101
                                 + 1.15538 w_k1_INTC_s101
                                 + 1.08148 w_k1_KO_s101
                                 + 0.897827 w_k1_MAT_s101
                                 + 1.33947 w_k1_MSFT_s101 + 1.1095 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.37993 w_k1_ADM_s101
                                 + 1.61603 w_k1_AXP_s101
                                 + 1.35779 w_k1_DIS_s101
                                 + 1.29519 w_k1_FDX_s101
                                 + 0.994854 w_k1_IBM_s101
                                 + 1.14246 w_k1_INTC_s101
                                 + 1.03365 w_k1_KO_s101 + 1.07692 w_k1_MAT_s101
                                 + 1.24475 w_k1_MSFT_s101
                                 + 1.17451 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.14095 w_k1_ADM_s101
                                 + 1.3608 w_k1_AXP_s101 + 1.19269 w_k1_DIS_s101
                                 + 1.02645 w_k1_FDX_s101
                                 + 1.69229 w_k1_IBM_s101
                                 + 1.12867 w_k1_INTC_s101
                                 + 1.26065 w_k1_KO_s101 + 1.96222 w_k1_MAT_s101
                                 + 1.08292 w_k1_MSFT_s101
                                 + 1.11456 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.963755 w_k1_ADM_s101
                                 + 0.9281 w_k1_AXP_s101
                                 + 0.821185 w_k1_DIS_s101
                                 + 0.848639 w_k1_FDX_s101
                                 + 0.802516 w_k1_IBM_s101
                                 + 0.760982 w_k1_INTC_s101
                                 + 0.862108 w_k1_KO_s101
                                 + 0.593461 w_k1_MAT_s101
                                 + 0.872052 w_k1_MSFT_s101
                                 + 0.884528 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.04028 w_k1_ADM_s102
                                 + 1.06385 w_k1_AXP_s102
                                 + 1.20807 w_k1_DIS_s102 + 1.2214 w_k1_FDX_s102
                                 + 0.728591 w_k1_IBM_s102
                                 + 1.2729 w_k1_INTC_s102 + 1.05034 w_k1_KO_s102
                                 + 0.779783 w_k1_MAT_s102
                                 + 1.1173 w_k1_MSFT_s102 + 1.12896 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.04223 w_k1_ADM_s102
                                 + 1.08345 w_k1_AXP_s102
                                 + 0.974941 w_k1_DIS_s102
                                 + 1.0982 w_k1_FDX_s102 + 1.02995 w_k1_IBM_s102
                                 + 1.25934 w_k1_INTC_s102
                                 + 1.06592 w_k1_KO_s102
                                 + 0.896624 w_k1_MAT_s102
                                 + 1.15011 w_k1_MSFT_s102
                                 + 1.02112 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.95564 w_k1_ADM_s102
                                 + 0.723272 w_k1_AXP_s102
                                 + 0.68957 w_k1_DIS_s102
                                 + 0.764889 w_k1_FDX_s102
                                 + 1.04869 w_k1_IBM_s102
                                 + 0.9977 w_k1_INTC_s102
                                 + 0.918757 w_k1_KO_s102
                                 + 0.676614 w_k1_MAT_s102
                                 + 0.857296 w_k1_MSFT_s102
                                 + 0.913669 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.0352 w_k1_ADM_s102
                                 + 1.36556 w_k1_AXP_s102
                                 + 1.06815 w_k1_DIS_s102
                                 + 1.10975 w_k1_FDX_s102
                                 + 1.25531 w_k1_IBM_s102
                                 + 1.08406 w_k1_INTC_s102
                                 + 1.21571 w_k1_KO_s102 + 1.51473 w_k1_MAT_s102
                                 + 1.12516 w_k1_MSFT_s102
                                 + 1.16615 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.833986 w_k1_ADM_s102
                                 + 0.73994 w_k1_AXP_s102
                                 + 0.634498 w_k1_DIS_s102
                                 + 0.799904 w_k1_FDX_s102
                                 + 0.723092 w_k1_IBM_s102
                                 + 0.972765 w_k1_INTC_s102
                                 + 0.811806 w_k1_KO_s102
                                 + 0.640027 w_k1_MAT_s102
                                 + 0.870811 w_k1_MSFT_s102
                                 + 0.877189 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.0139 w_k1_ADM_s102
                                 + 1.19296 w_k1_AXP_s102
                                 + 1.35125 w_k1_DIS_s102
                                 + 0.964155 w_k1_FDX_s102
                                 + 1.7053 w_k1_IBM_s102
                                 + 1.29823 w_k1_INTC_s102
                                 + 1.68714 w_k1_KO_s102 + 1.31485 w_k1_MAT_s102
                                 + 1.21334 w_k1_MSFT_s102
                                 + 1.21115 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.00153 w_k1_ADM_s102
                                 + 1.36198 w_k1_AXP_s102
                                 + 1.85591 w_k1_DIS_s102 + 1.407 w_k1_FDX_s102
                                 + 1.17857 w_k1_IBM_s102
                                 + 1.58306 w_k1_INTC_s102 + 1.568 w_k1_KO_s102
                                 + 1.13556 w_k1_MAT_s102
                                 + 1.42315 w_k1_MSFT_s102
                                 + 1.24735 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.857692 w_k1_ADM_s102
                                 + 0.927986 w_k1_AXP_s102
                                 + 1.02112 w_k1_DIS_s102
                                 + 0.888487 w_k1_FDX_s102
                                 + 1.40993 w_k1_IBM_s102
                                 + 1.29607 w_k1_INTC_s102
                                 + 1.22184 w_k1_KO_s102 + 1.02796 w_k1_MAT_s102
                                 + 1.0105 w_k1_MSFT_s102 + 1.1086 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.884514 w_k1_ADM_s103
                                 + 0.752054 w_k1_AXP_s103
                                 + 0.956618 w_k1_DIS_s103
                                 + 0.99163 w_k1_FDX_s103 + 1.0003 w_k1_IBM_s103
                                 + 0.925613 w_k1_INTC_s103
                                 + 1.06922 w_k1_KO_s103
                                 + 0.628903 w_k1_MAT_s103
                                 + 1.00812 w_k1_MSFT_s103
                                 + 0.960447 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.942959 w_k1_ADM_s103
                                 + 0.960137 w_k1_AXP_s103
                                 + 0.722786 w_k1_DIS_s103
                                 + 0.89166 w_k1_FDX_s103
                                 + 0.958869 w_k1_IBM_s103
                                 + 0.848542 w_k1_INTC_s103
                                 + 1.01134 w_k1_KO_s103 + 1.19997 w_k1_MAT_s103
                                 + 0.872109 w_k1_MSFT_s103
                                 + 0.922133 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.24446 w_k1_ADM_s103
                                 + 0.991995 w_k1_AXP_s103
                                 + 1.03558 w_k1_DIS_s103
                                 + 1.18317 w_k1_FDX_s103
                                 + 0.815201 w_k1_IBM_s103
                                 + 0.956953 w_k1_INTC_s103
                                 + 0.90284 w_k1_KO_s103 + 1.01165 w_k1_MAT_s103
                                 + 1.24992 w_k1_MSFT_s103
                                 + 0.979422 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.40023 w_k1_ADM_s103
                                 + 1.40575 w_k1_AXP_s103
                                 + 1.82654 w_k1_DIS_s103
                                 + 1.61416 w_k1_FDX_s103
                                 + 1.49968 w_k1_IBM_s103
                                 + 1.2798 w_k1_INTC_s103 + 1.64481 w_k1_KO_s103
                                 + 1.7604 w_k1_MAT_s103
                                 + 1.34715 w_k1_MSFT_s103
                                 + 1.25125 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.917576 w_k1_ADM_s103
                                 + 0.817955 w_k1_AXP_s103
                                 + 0.994787 w_k1_DIS_s103
                                 + 1.00483 w_k1_FDX_s103
                                 + 1.13182 w_k1_IBM_s103
                                 + 0.969717 w_k1_INTC_s103
                                 + 1.03274 w_k1_KO_s103 + 1.25285 w_k1_MAT_s103
                                 + 1.10771 w_k1_MSFT_s103
                                 + 0.930587 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.03445 w_k1_ADM_s103
                                 + 1.04818 w_k1_AXP_s103 + 1.3382 w_k1_DIS_s103
                                 + 1.18387 w_k1_FDX_s103
                                 + 1.13538 w_k1_IBM_s103
                                 + 1.28539 w_k1_INTC_s103
                                 + 1.19732 w_k1_KO_s103 + 1.11861 w_k1_MAT_s103
                                 + 1.17088 w_k1_MSFT_s103
                                 + 1.08291 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.615595 w_k1_ADM_s103
                                 + 0.709808 w_k1_AXP_s103
                                 + 0.665927 w_k1_DIS_s103
                                 + 0.757538 w_k1_FDX_s103
                                 + 0.83267 w_k1_IBM_s103
                                 + 0.809507 w_k1_INTC_s103
                                 + 0.775635 w_k1_KO_s103
                                 + 0.663053 w_k1_MAT_s103
                                 + 0.730912 w_k1_MSFT_s103
                                 + 0.776653 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.14337 w_k1_ADM_s103
                                 + 1.12687 w_k1_AXP_s103
                                 + 1.23067 w_k1_DIS_s103
                                 + 1.08407 w_k1_FDX_s103
                                 + 1.45883 w_k1_IBM_s103
                                 + 0.85033 w_k1_INTC_s103
                                 + 1.49734 w_k1_KO_s103 + 1.59974 w_k1_MAT_s103
                                 + 1.05893 w_k1_MSFT_s103
                                 + 1.10834 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.16681 w_k1_ADM_s104
                                 + 1.04312 w_k1_AXP_s104
                                 + 0.973788 w_k1_DIS_s104
                                 + 1.01433 w_k1_FDX_s104
                                 + 1.18263 w_k1_IBM_s104
                                 + 0.90526 w_k1_INTC_s104
                                 + 1.04237 w_k1_KO_s104 + 1.02064 w_k1_MAT_s104
                                 + 0.937582 w_k1_MSFT_s104
                                 + 1.01113 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.871626 w_k1_ADM_s104
                                 + 1.04488 w_k1_AXP_s104
                                 + 1.29605 w_k1_DIS_s104
                                 + 1.12623 w_k1_FDX_s104
                                 + 0.954822 w_k1_IBM_s104
                                 + 1.09617 w_k1_INTC_s104
                                 + 1.09024 w_k1_KO_s104
                                 + 0.882204 w_k1_MAT_s104
                                 + 1.09998 w_k1_MSFT_s104
                                 + 1.14568 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.9889 w_k1_ADM_s104
                                 + 1.36594 w_k1_AXP_s104
                                 + 1.31842 w_k1_DIS_s104
                                 + 1.11463 w_k1_FDX_s104
                                 + 1.51659 w_k1_IBM_s104
                                 + 1.33387 w_k1_INTC_s104
                                 + 1.40905 w_k1_KO_s104 + 1.63339 w_k1_MAT_s104
                                 + 1.19293 w_k1_MSFT_s104
                                 + 1.23452 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.825514 w_k1_ADM_s104
                                 + 0.973504 w_k1_AXP_s104
                                 + 0.850187 w_k1_DIS_s104
                                 + 0.923232 w_k1_FDX_s104
                                 + 0.84956 w_k1_IBM_s104
                                 + 0.987045 w_k1_INTC_s104
                                 + 0.864661 w_k1_KO_s104
                                 + 0.799672 w_k1_MAT_s104
                                 + 0.920964 w_k1_MSFT_s104
                                 + 1.05492 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.05636 w_k1_ADM_s104
                                 + 1.41347 w_k1_AXP_s104
                                 + 1.35339 w_k1_DIS_s104
                                 + 1.35285 w_k1_FDX_s104
                                 + 0.789442 w_k1_IBM_s104
                                 + 1.33767 w_k1_INTC_s104
                                 + 1.14658 w_k1_KO_s104
                                 + 0.773833 w_k1_MAT_s104
                                 + 1.32768 w_k1_MSFT_s104
                                 + 1.21979 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.897123 w_k1_ADM_s104
                                 + 0.894834 w_k1_AXP_s104
                                 + 0.919486 w_k1_DIS_s104
                                 + 0.89725 w_k1_FDX_s104
                                 + 1.26288 w_k1_IBM_s104
                                 + 1.2002 w_k1_INTC_s104 + 1.20273 w_k1_KO_s104
                                 + 1.00737 w_k1_MAT_s104
                                 + 1.01856 w_k1_MSFT_s104
                                 + 1.04168 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.934811 w_k1_ADM_s104
                                 + 0.710022 w_k1_AXP_s104
                                 + 0.72844 w_k1_DIS_s104
                                 + 0.729742 w_k1_FDX_s104
                                 + 1.15119 w_k1_IBM_s104
                                 + 0.937417 w_k1_INTC_s104
                                 + 0.967275 w_k1_KO_s104
                                 + 0.91626 w_k1_MAT_s104
                                 + 0.837713 w_k1_MSFT_s104
                                 + 0.936907 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.787209 w_k1_ADM_s104
                                 + 0.632706 w_k1_AXP_s104
                                 + 0.761213 w_k1_DIS_s104
                                 + 0.790115 w_k1_FDX_s104
                                 + 0.753886 w_k1_IBM_s104
                                 + 0.942686 w_k1_INTC_s104
                                 + 0.761616 w_k1_KO_s104
                                 + 0.411419 w_k1_MAT_s104
                                 + 0.815157 w_k1_MSFT_s104
                                 + 0.831014 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.82148 w_k1_ADM_s105
                                 + 0.789107 w_k1_AXP_s105
                                 + 0.840474 w_k1_DIS_s105
                                 + 0.84574 w_k1_FDX_s105
                                 + 0.883327 w_k1_IBM_s105
                                 + 0.814723 w_k1_INTC_s105
                                 + 0.870047 w_k1_KO_s105
                                 + 0.68863 w_k1_MAT_s105
                                 + 0.996632 w_k1_MSFT_s105
                                 + 0.919725 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.893045 w_k1_ADM_s105
                                 + 0.982409 w_k1_AXP_s105
                                 + 1.13751 w_k1_DIS_s105
                                 + 1.13275 w_k1_FDX_s105
                                 + 0.803359 w_k1_IBM_s105
                                 + 0.993709 w_k1_INTC_s105
                                 + 0.829581 w_k1_KO_s105
                                 + 0.819717 w_k1_MAT_s105
                                 + 1.19275 w_k1_MSFT_s105
                                 + 1.01692 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.15376 w_k1_ADM_s105
                                 + 1.38525 w_k1_AXP_s105
                                 + 1.79244 w_k1_DIS_s105 + 1.3159 w_k1_FDX_s105
                                 + 1.45892 w_k1_IBM_s105
                                 + 1.22767 w_k1_INTC_s105
                                 + 1.40259 w_k1_KO_s105 + 1.25541 w_k1_MAT_s105
                                 + 1.41243 w_k1_MSFT_s105
                                 + 1.31823 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.665528 w_k1_ADM_s105
                                 + 0.696298 w_k1_AXP_s105
                                 + 0.593291 w_k1_DIS_s105
                                 + 0.783187 w_k1_FDX_s105
                                 + 0.68633 w_k1_IBM_s105
                                 + 0.758856 w_k1_INTC_s105
                                 + 0.649104 w_k1_KO_s105
                                 + 0.709991 w_k1_MAT_s105
                                 + 0.81948 w_k1_MSFT_s105
                                 + 0.785988 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.15607 w_k1_ADM_s105
                                 + 0.786316 w_k1_AXP_s105
                                 + 1.21149 w_k1_DIS_s105
                                 + 1.04126 w_k1_FDX_s105 + 1.0228 w_k1_IBM_s105
                                 + 0.808739 w_k1_INTC_s105
                                 + 0.96396 w_k1_KO_s105
                                 + 0.869278 w_k1_MAT_s105
                                 + 1.04193 w_k1_MSFT_s105
                                 + 1.03076 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.12604 w_k1_ADM_s105
                                 + 1.40077 w_k1_AXP_s105
                                 + 1.48557 w_k1_DIS_s105
                                 + 1.28505 w_k1_FDX_s105
                                 + 0.963953 w_k1_IBM_s105
                                 + 1.21303 w_k1_INTC_s105
                                 + 1.06793 w_k1_KO_s105 + 1.13348 w_k1_MAT_s105
                                 + 1.31444 w_k1_MSFT_s105
                                 + 1.10586 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.89885 w_k1_ADM_s105
                                 + 0.880721 w_k1_AXP_s105
                                 + 1.1773 w_k1_DIS_s105
                                 + 0.993803 w_k1_FDX_s105
                                 + 0.912052 w_k1_IBM_s105
                                 + 0.968551 w_k1_INTC_s105
                                 + 1.09115 w_k1_KO_s105
                                 + 0.988898 w_k1_MAT_s105
                                 + 1.11945 w_k1_MSFT_s105
                                 + 1.03541 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.856855 w_k1_ADM_s105
                                 + 0.994722 w_k1_AXP_s105
                                 + 0.879271 w_k1_DIS_s105
                                 + 0.943843 w_k1_FDX_s105
                                 + 1.24339 w_k1_IBM_s105
                                 + 0.836548 w_k1_INTC_s105
                                 + 0.900381 w_k1_KO_s105
                                 + 1.01614 w_k1_MAT_s105
                                 + 0.986052 w_k1_MSFT_s105
                                 + 1.00584 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.923548 w_k1_ADM_s106
                                 + 1.09385 w_k1_AXP_s106
                                 + 1.27521 w_k1_DIS_s106 + 1.0654 w_k1_FDX_s106
                                 + 1.02795 w_k1_IBM_s106
                                 + 0.998833 w_k1_INTC_s106
                                 + 1.08038 w_k1_KO_s106 + 1.12701 w_k1_MAT_s106
                                 + 1.06521 w_k1_MSFT_s106
                                 + 1.00696 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.709902 w_k1_ADM_s106
                                 + 0.863378 w_k1_AXP_s106
                                 + 0.848917 w_k1_DIS_s106
                                 + 0.714226 w_k1_FDX_s106
                                 + 1.07622 w_k1_IBM_s106
                                 + 0.743502 w_k1_INTC_s106
                                 + 0.975037 w_k1_KO_s106
                                 + 1.22696 w_k1_MAT_s106
                                 + 0.779402 w_k1_MSFT_s106
                                 + 0.839482 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.856291 w_k1_ADM_s106
                                 + 1.04494 w_k1_AXP_s106
                                 + 1.36544 w_k1_DIS_s106
                                 + 1.02147 w_k1_FDX_s106
                                 + 1.53935 w_k1_IBM_s106
                                 + 0.875209 w_k1_INTC_s106
                                 + 1.32556 w_k1_KO_s106 + 1.6367 w_k1_MAT_s106
                                 + 0.89423 w_k1_MSFT_s106
                                 + 1.05804 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.28392 w_k1_ADM_s106
                                 + 1.46054 w_k1_AXP_s106
                                 + 1.59088 w_k1_DIS_s106
                                 + 1.35249 w_k1_FDX_s106
                                 + 1.27829 w_k1_IBM_s106
                                 + 1.30269 w_k1_INTC_s106
                                 + 1.53709 w_k1_KO_s106 + 1.42029 w_k1_MAT_s106
                                 + 1.2866 w_k1_MSFT_s106 + 1.15944 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.939219 w_k1_ADM_s106
                                 + 0.757311 w_k1_AXP_s106
                                 + 0.793698 w_k1_DIS_s106
                                 + 0.904507 w_k1_FDX_s106
                                 + 0.819404 w_k1_IBM_s106
                                 + 0.733255 w_k1_INTC_s106
                                 + 0.863433 w_k1_KO_s106
                                 + 0.905566 w_k1_MAT_s106
                                 + 0.869643 w_k1_MSFT_s106
                                 + 0.864547 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.01243 w_k1_ADM_s106
                                 + 1.49795 w_k1_AXP_s106
                                 + 1.43082 w_k1_DIS_s106
                                 + 1.35061 w_k1_FDX_s106
                                 + 1.02016 w_k1_IBM_s106
                                 + 0.862847 w_k1_INTC_s106
                                 + 1.09721 w_k1_KO_s106 + 1.42084 w_k1_MAT_s106
                                 + 1.26771 w_k1_MSFT_s106
                                 + 1.19935 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.39665 w_k1_ADM_s106
                                 + 1.11947 w_k1_AXP_s106 + 1.1594 w_k1_DIS_s106
                                 + 1.1461 w_k1_FDX_s106 + 0.76136 w_k1_IBM_s106
                                 + 0.972624 w_k1_INTC_s106
                                 + 0.892173 w_k1_KO_s106
                                 + 0.912744 w_k1_MAT_s106
                                 + 1.01926 w_k1_MSFT_s106
                                 + 0.900423 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.82587 w_k1_ADM_s106
                                 + 0.736043 w_k1_AXP_s106
                                 + 0.658258 w_k1_DIS_s106
                                 + 0.848451 w_k1_FDX_s106
                                 + 0.636332 w_k1_IBM_s106
                                 + 0.83018 w_k1_INTC_s106
                                 + 0.616885 w_k1_KO_s106
                                 + 0.695823 w_k1_MAT_s106
                                 + 0.822366 w_k1_MSFT_s106
                                 + 0.795146 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.985395 w_k1_ADM_s107
                                 + 0.84158 w_k1_AXP_s107
                                 + 1.12075 w_k1_DIS_s107
                                 + 1.04626 w_k1_FDX_s107
                                 + 0.960871 w_k1_IBM_s107
                                 + 1.01513 w_k1_INTC_s107 + 1.0532 w_k1_KO_s107
                                 + 0.78325 w_k1_MAT_s107
                                 + 1.14458 w_k1_MSFT_s107
                                 + 0.984275 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.923799 w_k1_ADM_s107
                                 + 0.935865 w_k1_AXP_s107
                                 + 0.663626 w_k1_DIS_s107
                                 + 0.865696 w_k1_FDX_s107
                                 + 0.680687 w_k1_IBM_s107
                                 + 0.87791 w_k1_INTC_s107
                                 + 0.773563 w_k1_KO_s107
                                 + 0.642724 w_k1_MAT_s107
                                 + 0.863245 w_k1_MSFT_s107
                                 + 0.882227 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.44015 w_k1_ADM_s107
                                 + 1.3682 w_k1_AXP_s107 + 1.41279 w_k1_DIS_s107
                                 + 1.21667 w_k1_FDX_s107
                                 + 1.38466 w_k1_IBM_s107
                                 + 1.17707 w_k1_INTC_s107
                                 + 1.13898 w_k1_KO_s107 + 1.21087 w_k1_MAT_s107
                                 + 1.1608 w_k1_MSFT_s107 + 1.04616 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.1437 w_k1_ADM_s107
                                 + 1.46154 w_k1_AXP_s107
                                 + 1.23939 w_k1_DIS_s107
                                 + 1.28533 w_k1_FDX_s107
                                 + 0.885981 w_k1_IBM_s107
                                 + 1.11484 w_k1_INTC_s107
                                 + 0.802842 w_k1_KO_s107
                                 + 0.916193 w_k1_MAT_s107
                                 + 1.11128 w_k1_MSFT_s107
                                 + 1.09377 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.996683 w_k1_ADM_s107
                                 + 0.919971 w_k1_AXP_s107
                                 + 0.808945 w_k1_DIS_s107
                                 + 0.875945 w_k1_FDX_s107
                                 + 1.0665 w_k1_IBM_s107
                                 + 0.983572 w_k1_INTC_s107
                                 + 0.806209 w_k1_KO_s107
                                 + 0.86487 w_k1_MAT_s107
                                 + 0.889951 w_k1_MSFT_s107
                                 + 0.940888 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.02839 w_k1_ADM_s107
                                 + 1.14237 w_k1_AXP_s107
                                 + 1.13972 w_k1_DIS_s107
                                 + 1.05878 w_k1_FDX_s107
                                 + 1.45236 w_k1_IBM_s107
                                 + 0.897419 w_k1_INTC_s107
                                 + 1.00111 w_k1_KO_s107 + 1.20155 w_k1_MAT_s107
                                 + 1.07269 w_k1_MSFT_s107
                                 + 1.04644 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.79089 w_k1_ADM_s107
                                 + 1.78392 w_k1_AXP_s107 + 1.7801 w_k1_DIS_s107
                                 + 1.40337 w_k1_FDX_s107 + 1.3587 w_k1_IBM_s107
                                 + 1.18094 w_k1_INTC_s107
                                 + 1.25963 w_k1_KO_s107 + 1.51797 w_k1_MAT_s107
                                 + 1.35741 w_k1_MSFT_s107
                                 + 1.34909 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.755985 w_k1_ADM_s107
                                 + 0.554125 w_k1_AXP_s107
                                 + 0.562681 w_k1_DIS_s107
                                 + 0.638269 w_k1_FDX_s107
                                 + 0.70369 w_k1_IBM_s107
                                 + 0.708028 w_k1_INTC_s107
                                 + 0.583232 w_k1_KO_s107
                                 + 0.673018 w_k1_MAT_s107
                                 + 0.792536 w_k1_MSFT_s107
                                 + 0.78283 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.849922 w_k1_ADM_s110
                                 + 1.2275 w_k1_AXP_s110
                                 + 0.784019 w_k1_DIS_s110
                                 + 0.905708 w_k1_FDX_s110
                                 + 1.12061 w_k1_IBM_s110
                                 + 1.02717 w_k1_INTC_s110
                                 + 1.15451 w_k1_KO_s110 + 1.47751 w_k1_MAT_s110
                                 + 1.00562 w_k1_MSFT_s110
                                 + 0.904775 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.04972 w_k1_ADM_s110
                                 + 1.03989 w_k1_AXP_s110
                                 + 0.94837 w_k1_DIS_s110
                                 + 0.992573 w_k1_FDX_s110
                                 + 0.873112 w_k1_IBM_s110
                                 + 0.973608 w_k1_INTC_s110
                                 + 0.984103 w_k1_KO_s110
                                 + 1.19114 w_k1_MAT_s110
                                 + 1.06885 w_k1_MSFT_s110
                                 + 0.868559 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.991154 w_k1_ADM_s110
                                 + 0.943793 w_k1_AXP_s110
                                 + 0.724078 w_k1_DIS_s110
                                 + 0.996716 w_k1_FDX_s110
                                 + 0.629751 w_k1_IBM_s110
                                 + 0.869044 w_k1_INTC_s110
                                 + 0.790061 w_k1_KO_s110
                                 + 0.764052 w_k1_MAT_s110
                                 + 0.841402 w_k1_MSFT_s110
                                 + 0.853613 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.801386 w_k1_ADM_s110
                                 + 0.747852 w_k1_AXP_s110
                                 + 0.755797 w_k1_DIS_s110
                                 + 0.721229 w_k1_FDX_s110
                                 + 0.976706 w_k1_IBM_s110
                                 + 0.966838 w_k1_INTC_s110
                                 + 0.948672 w_k1_KO_s110
                                 + 0.625603 w_k1_MAT_s110
                                 + 0.700278 w_k1_MSFT_s110
                                 + 0.713885 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.9987 w_k1_ADM_s110
                                 + 0.968106 w_k1_AXP_s110
                                 + 0.969609 w_k1_DIS_s110
                                 + 0.92158 w_k1_FDX_s110
                                 + 1.23119 w_k1_IBM_s110
                                 + 0.922704 w_k1_INTC_s110
                                 + 1.28565 w_k1_KO_s110 + 1.39452 w_k1_MAT_s110
                                 + 0.928315 w_k1_MSFT_s110
                                 + 0.994625 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.07054 w_k1_ADM_s110
                                 + 1.2945 w_k1_AXP_s110 + 1.47983 w_k1_DIS_s110
                                 + 1.40944 w_k1_FDX_s110
                                 + 0.899014 w_k1_IBM_s110
                                 + 1.48885 w_k1_INTC_s110
                                 + 1.39121 w_k1_KO_s110
                                 + 0.843989 w_k1_MAT_s110
                                 + 1.34513 w_k1_MSFT_s110
                                 + 1.03679 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.847921 w_k1_ADM_s110
                                 + 0.698185 w_k1_AXP_s110
                                 + 0.751942 w_k1_DIS_s110
                                 + 0.760939 w_k1_FDX_s110
                                 + 0.77125 w_k1_IBM_s110
                                 + 0.780111 w_k1_INTC_s110
                                 + 0.816616 w_k1_KO_s110
                                 + 0.727839 w_k1_MAT_s110
                                 + 0.78339 w_k1_MSFT_s110
                                 + 0.792824 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.733741 w_k1_ADM_s110
                                 + 1.51613 w_k1_AXP_s110
                                 + 1.20355 w_k1_DIS_s110
                                 + 1.07812 w_k1_FDX_s110
                                 + 1.59687 w_k1_IBM_s110
                                 + 1.35661 w_k1_INTC_s110 + 1.5054 w_k1_KO_s110
                                 + 1.65446 w_k1_MAT_s110
                                 + 0.99974 w_k1_MSFT_s110
                                 + 1.05263 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.02015 w_k1_ADM_s111
                                 + 1.43897 w_k1_AXP_s111
                                 + 1.38584 w_k1_DIS_s111
                                 + 1.36539 w_k1_FDX_s111
                                 + 0.936756 w_k1_IBM_s111
                                 + 1.34126 w_k1_INTC_s111
                                 + 1.16751 w_k1_KO_s111 + 1.09794 w_k1_MAT_s111
                                 + 1.05584 w_k1_MSFT_s111
                                 + 1.19463 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.908925 w_k1_ADM_s111
                                 + 0.871987 w_k1_AXP_s111
                                 + 0.931951 w_k1_DIS_s111
                                 + 1.04214 w_k1_FDX_s111
                                 + 0.98242 w_k1_IBM_s111
                                 + 1.20547 w_k1_INTC_s111
                                 + 0.938625 w_k1_KO_s111
                                 + 0.674976 w_k1_MAT_s111
                                 + 1.14608 w_k1_MSFT_s111
                                 + 1.01602 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.15512 w_k1_ADM_s111
                                 + 0.978558 w_k1_AXP_s111
                                 + 1.25118 w_k1_DIS_s111
                                 + 1.09563 w_k1_FDX_s111
                                 + 1.60236 w_k1_IBM_s111
                                 + 1.10992 w_k1_INTC_s111
                                 + 1.50623 w_k1_KO_s111 + 1.59942 w_k1_MAT_s111
                                 + 1.09491 w_k1_MSFT_s111
                                 + 1.18895 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.762186 w_k1_ADM_s111
                                 + 0.659101 w_k1_AXP_s111
                                 + 0.67479 w_k1_DIS_s111
                                 + 0.777698 w_k1_FDX_s111
                                 + 1.0009 w_k1_IBM_s111
                                 + 0.896055 w_k1_INTC_s111
                                 + 0.872014 w_k1_KO_s111
                                 + 0.758995 w_k1_MAT_s111
                                 + 0.809536 w_k1_MSFT_s111
                                 + 0.98866 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.11764 w_k1_ADM_s111
                                 + 1.13554 w_k1_AXP_s111
                                 + 1.32557 w_k1_DIS_s111 + 1.1302 w_k1_FDX_s111
                                 + 1.29479 w_k1_IBM_s111
                                 + 1.03936 w_k1_INTC_s111
                                 + 1.34555 w_k1_KO_s111 + 1.56126 w_k1_MAT_s111
                                 + 1.18164 w_k1_MSFT_s111
                                 + 1.23668 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.966631 w_k1_ADM_s111
                                 + 0.891185 w_k1_AXP_s111
                                 + 0.88454 w_k1_DIS_s111
                                 + 0.976532 w_k1_FDX_s111
                                 + 0.729688 w_k1_IBM_s111
                                 + 0.889613 w_k1_INTC_s111
                                 + 0.876935 w_k1_KO_s111
                                 + 0.715533 w_k1_MAT_s111
                                 + 0.972143 w_k1_MSFT_s111
                                 + 0.970898 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.803525 w_k1_ADM_s111
                                 + 0.770021 w_k1_AXP_s111
                                 + 0.819901 w_k1_DIS_s111
                                 + 0.849114 w_k1_FDX_s111
                                 + 0.880414 w_k1_IBM_s111
                                 + 1.12858 w_k1_INTC_s111
                                 + 0.890699 w_k1_KO_s111
                                 + 0.869357 w_k1_MAT_s111
                                 + 1.02295 w_k1_MSFT_s111
                                 + 0.997842 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.803285 w_k1_ADM_s111
                                 + 0.524488 w_k1_AXP_s111
                                 + 0.581645 w_k1_DIS_s111
                                 + 0.686664 w_k1_FDX_s111
                                 + 0.687773 w_k1_IBM_s111
                                 + 0.736105 w_k1_INTC_s111
                                 + 0.721021 w_k1_KO_s111
                                 + 0.420777 w_k1_MAT_s111
                                 + 0.771114 w_k1_MSFT_s111
                                 + 0.765361 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.952324 w_k1_ADM_s112
                                 + 0.874852 w_k1_AXP_s112
                                 + 0.719032 w_k1_DIS_s112
                                 + 0.874964 w_k1_FDX_s112
                                 + 0.9835 w_k1_IBM_s112
                                 + 0.964232 w_k1_INTC_s112
                                 + 0.971506 w_k1_KO_s112
                                 + 1.14651 w_k1_MAT_s112
                                 + 1.03982 w_k1_MSFT_s112
                                 + 1.00802 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.895447 w_k1_ADM_s112
                                 + 0.656513 w_k1_AXP_s112
                                 + 0.84028 w_k1_DIS_s112
                                 + 0.850334 w_k1_FDX_s112
                                 + 0.769409 w_k1_IBM_s112
                                 + 0.948107 w_k1_INTC_s112
                                 + 0.990841 w_k1_KO_s112
                                 + 0.828245 w_k1_MAT_s112
                                 + 1.01575 w_k1_MSFT_s112
                                 + 0.89566 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.51663 w_k1_ADM_s112
                                 + 1.29794 w_k1_AXP_s112
                                 + 1.72681 w_k1_DIS_s112
                                 + 1.38664 w_k1_FDX_s112
                                 + 1.32274 w_k1_IBM_s112
                                 + 1.25967 w_k1_INTC_s112
                                 + 1.67472 w_k1_KO_s112 + 2.28935 w_k1_MAT_s112
                                 + 1.3699 w_k1_MSFT_s112 + 1.24702 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.04978 w_k1_ADM_s112
                                 + 0.960548 w_k1_AXP_s112
                                 + 1.14302 w_k1_DIS_s112
                                 + 0.977301 w_k1_FDX_s112
                                 + 1.17734 w_k1_IBM_s112
                                 + 1.10937 w_k1_INTC_s112
                                 + 1.44161 w_k1_KO_s112 + 1.45454 w_k1_MAT_s112
                                 + 1.21714 w_k1_MSFT_s112 + 1.1159 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.16388 w_k1_ADM_s112
                                 + 0.945417 w_k1_AXP_s112
                                 + 0.963994 w_k1_DIS_s112
                                 + 0.849033 w_k1_FDX_s112
                                 + 1.33821 w_k1_IBM_s112
                                 + 0.845513 w_k1_INTC_s112
                                 + 1.4155 w_k1_KO_s112 + 1.53939 w_k1_MAT_s112
                                 + 0.939369 w_k1_MSFT_s112
                                 + 0.931498 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.19717 w_k1_ADM_s112
                                 + 0.908111 w_k1_AXP_s112
                                 + 1.26423 w_k1_DIS_s112
                                 + 1.19794 w_k1_FDX_s112
                                 + 1.06626 w_k1_IBM_s112
                                 + 1.16566 w_k1_INTC_s112
                                 + 1.31042 w_k1_KO_s112 + 1.14396 w_k1_MAT_s112
                                 + 1.36014 w_k1_MSFT_s112
                                 + 1.13848 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.40206 w_k1_ADM_s112
                                 + 0.962875 w_k1_AXP_s112
                                 + 0.835654 w_k1_DIS_s112
                                 + 1.11958 w_k1_FDX_s112
                                 + 0.722813 w_k1_IBM_s112
                                 + 1.03413 w_k1_INTC_s112
                                 + 1.02518 w_k1_KO_s112
                                 + 0.859933 w_k1_MAT_s112
                                 + 1.11292 w_k1_MSFT_s112
                                 + 0.968973 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.11982 w_k1_ADM_s112
                                 + 1.47105 w_k1_AXP_s112
                                 + 1.10612 w_k1_DIS_s112
                                 + 1.17003 w_k1_FDX_s112
                                 + 1.01125 w_k1_IBM_s112
                                 + 1.43458 w_k1_INTC_s112
                                 + 1.17757 w_k1_KO_s112 + 1.21004 w_k1_MAT_s112
                                 + 1.06443 w_k1_MSFT_s112
                                 + 1.11266 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.834105 w_k1_ADM_s113
                                 + 0.926038 w_k1_AXP_s113
                                 + 0.935685 w_k1_DIS_s113
                                 + 1.0014 w_k1_FDX_s113 + 1.17608 w_k1_IBM_s113
                                 + 1.14603 w_k1_INTC_s113
                                 + 1.17769 w_k1_KO_s113 + 1.11735 w_k1_MAT_s113
                                 + 1.12011 w_k1_MSFT_s113
                                 + 1.19751 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.904696 w_k1_ADM_s113
                                 + 1.36494 w_k1_AXP_s113
                                 + 1.13609 w_k1_DIS_s113
                                 + 1.05535 w_k1_FDX_s113
                                 + 1.42157 w_k1_IBM_s113
                                 + 1.1696 w_k1_INTC_s113 + 1.24311 w_k1_KO_s113
                                 + 1.56838 w_k1_MAT_s113
                                 + 1.08312 w_k1_MSFT_s113
                                 + 1.21096 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.74286 w_k1_ADM_s113
                                 + 0.934849 w_k1_AXP_s113
                                 + 0.867411 w_k1_DIS_s113
                                 + 0.962196 w_k1_FDX_s113
                                 + 0.976896 w_k1_IBM_s113
                                 + 1.08211 w_k1_INTC_s113
                                 + 1.06128 w_k1_KO_s113 + 1.04841 w_k1_MAT_s113
                                 + 0.977642 w_k1_MSFT_s113
                                 + 0.938357 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.770136 w_k1_ADM_s113
                                 + 1.21963 w_k1_AXP_s113
                                 + 1.68396 w_k1_DIS_s113
                                 + 1.28956 w_k1_FDX_s113
                                 + 1.46145 w_k1_IBM_s113
                                 + 1.38855 w_k1_INTC_s113
                                 + 1.42495 w_k1_KO_s113 + 1.36558 w_k1_MAT_s113
                                 + 1.27898 w_k1_MSFT_s113
                                 + 1.33997 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.848715 w_k1_ADM_s113
                                 + 0.961177 w_k1_AXP_s113
                                 + 0.782868 w_k1_DIS_s113
                                 + 0.930074 w_k1_FDX_s113
                                 + 0.848994 w_k1_IBM_s113
                                 + 0.993643 w_k1_INTC_s113
                                 + 0.795922 w_k1_KO_s113
                                 + 0.677081 w_k1_MAT_s113
                                 + 0.895884 w_k1_MSFT_s113
                                 + 0.935979 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.599913 w_k1_ADM_s113
                                 + 0.698582 w_k1_AXP_s113
                                 + 0.659195 w_k1_DIS_s113
                                 + 0.846158 w_k1_FDX_s113
                                 + 0.676922 w_k1_IBM_s113
                                 + 0.723698 w_k1_INTC_s113
                                 + 0.722491 w_k1_KO_s113
                                 + 0.789095 w_k1_MAT_s113
                                 + 0.842459 w_k1_MSFT_s113
                                 + 0.879967 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.986521 w_k1_ADM_s113
                                 + 0.939969 w_k1_AXP_s113
                                 + 1.20703 w_k1_DIS_s113
                                 + 1.16335 w_k1_FDX_s113
                                 + 0.981213 w_k1_IBM_s113
                                 + 1.04091 w_k1_INTC_s113 + 1.1365 w_k1_KO_s113
                                 + 0.891984 w_k1_MAT_s113
                                 + 1.14682 w_k1_MSFT_s113
                                 + 1.11216 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.840494 w_k1_ADM_s113
                                 + 0.699475 w_k1_AXP_s113
                                 + 0.726778 w_k1_DIS_s113
                                 + 0.733249 w_k1_FDX_s113
                                 + 1.07281 w_k1_IBM_s113
                                 + 0.803715 w_k1_INTC_s113
                                 + 0.888713 w_k1_KO_s113
                                 + 0.916156 w_k1_MAT_s113
                                 + 0.849125 w_k1_MSFT_s113
                                 + 0.907477 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.783973 w_k1_ADM_s114
                                 + 0.662968 w_k1_AXP_s114
                                 + 0.834821 w_k1_DIS_s114
                                 + 0.934385 w_k1_FDX_s114
                                 + 0.654043 w_k1_IBM_s114
                                 + 0.84711 w_k1_INTC_s114
                                 + 0.673678 w_k1_KO_s114
                                 + 0.688714 w_k1_MAT_s114
                                 + 0.896276 w_k1_MSFT_s114
                                 + 1.07291 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.985648 w_k1_ADM_s114
                                 + 1.12025 w_k1_AXP_s114
                                 + 0.979132 w_k1_DIS_s114
                                 + 0.960562 w_k1_FDX_s114
                                 + 1.19909 w_k1_IBM_s114
                                 + 0.919564 w_k1_INTC_s114
                                 + 0.99093 w_k1_KO_s114 + 1.08513 w_k1_MAT_s114
                                 + 0.747818 w_k1_MSFT_s114
                                 + 0.976326 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.10144 w_k1_ADM_s114
                                 + 1.27874 w_k1_AXP_s114
                                 + 1.92725 w_k1_DIS_s114
                                 + 1.45658 w_k1_FDX_s114
                                 + 0.996213 w_k1_IBM_s114
                                 + 1.29117 w_k1_INTC_s114 + 1.1248 w_k1_KO_s114
                                 + 0.688874 w_k1_MAT_s114
                                 + 1.28141 w_k1_MSFT_s114
                                 + 1.21478 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.753992 w_k1_ADM_s114
                                 + 1.35718 w_k1_AXP_s114
                                 + 1.73081 w_k1_DIS_s114
                                 + 1.31423 w_k1_FDX_s114
                                 + 1.46863 w_k1_IBM_s114
                                 + 1.42411 w_k1_INTC_s114
                                 + 1.27795 w_k1_KO_s114 + 1.83039 w_k1_MAT_s114
                                 + 1.35161 w_k1_MSFT_s114
                                 + 1.30216 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.797921 w_k1_ADM_s114
                                 + 0.920522 w_k1_AXP_s114
                                 + 0.900862 w_k1_DIS_s114
                                 + 0.86135 w_k1_FDX_s114
                                 + 0.944183 w_k1_IBM_s114
                                 + 1.03521 w_k1_INTC_s114
                                 + 0.771333 w_k1_KO_s114
                                 + 0.882521 w_k1_MAT_s114
                                 + 0.94439 w_k1_MSFT_s114
                                 + 1.09948 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.02595 w_k1_ADM_s114
                                 + 1.2033 w_k1_AXP_s114 + 1.22008 w_k1_DIS_s114
                                 + 1.18391 w_k1_FDX_s114
                                 + 0.864647 w_k1_IBM_s114
                                 + 1.29672 w_k1_INTC_s114
                                 + 0.965756 w_k1_KO_s114
                                 + 1.12043 w_k1_MAT_s114
                                 + 1.15947 w_k1_MSFT_s114
                                 + 1.09351 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.739434 w_k1_ADM_s114
                                 + 0.750714 w_k1_AXP_s114
                                 + 0.567217 w_k1_DIS_s114
                                 + 0.70705 w_k1_FDX_s114
                                 + 0.773102 w_k1_IBM_s114
                                 + 0.784592 w_k1_INTC_s114
                                 + 0.622781 w_k1_KO_s114
                                 + 1.10645 w_k1_MAT_s114
                                 + 0.671673 w_k1_MSFT_s114
                                 + 0.901628 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.07633 w_k1_ADM_s114
                                 + 0.931392 w_k1_AXP_s114
                                 + 1.27016 w_k1_DIS_s114
                                 + 0.942358 w_k1_FDX_s114
                                 + 1.54148 w_k1_IBM_s114
                                 + 0.996601 w_k1_INTC_s114
                                 + 1.01274 w_k1_KO_s114 + 1.36234 w_k1_MAT_s114
                                 + 1.14097 w_k1_MSFT_s114
                                 + 1.22796 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.09038 w_k1_ADM_s115
                                 + 1.22717 w_k1_AXP_s115
                                 + 1.38777 w_k1_DIS_s115 + 1.07 w_k1_FDX_s115
                                 + 1.60378 w_k1_IBM_s115
                                 + 0.92062 w_k1_INTC_s115
                                 + 1.58492 w_k1_KO_s115 + 1.76147 w_k1_MAT_s115
                                 + 1.05781 w_k1_MSFT_s115
                                 + 1.30802 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.813423 w_k1_ADM_s115
                                 + 0.824817 w_k1_AXP_s115
                                 + 1.15448 w_k1_DIS_s115
                                 + 0.896608 w_k1_FDX_s115
                                 + 0.868165 w_k1_IBM_s115
                                 + 0.762253 w_k1_INTC_s115
                                 + 1.11225 w_k1_KO_s115
                                 + 0.838318 w_k1_MAT_s115
                                 + 0.973652 w_k1_MSFT_s115
                                 + 1.10615 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.816957 w_k1_ADM_s115
                                 + 0.836766 w_k1_AXP_s115
                                 + 0.969661 w_k1_DIS_s115
                                 + 1.0621 w_k1_FDX_s115
                                 + 0.712079 w_k1_IBM_s115
                                 + 1.13409 w_k1_INTC_s115
                                 + 0.975379 w_k1_KO_s115
                                 + 0.501174 w_k1_MAT_s115
                                 + 1.02698 w_k1_MSFT_s115
                                 + 0.856972 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.18249 w_k1_ADM_s115
                                 + 1.09911 w_k1_AXP_s115
                                 + 1.21092 w_k1_DIS_s115
                                 + 1.05003 w_k1_FDX_s115
                                 + 1.03933 w_k1_IBM_s115
                                 + 1.16216 w_k1_INTC_s115
                                 + 1.27759 w_k1_KO_s115 + 1.05134 w_k1_MAT_s115
                                 + 1.08896 w_k1_MSFT_s115
                                 + 1.09361 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.873138 w_k1_ADM_s115
                                 + 0.734574 w_k1_AXP_s115
                                 + 0.721598 w_k1_DIS_s115
                                 + 0.78542 w_k1_FDX_s115
                                 + 0.677397 w_k1_IBM_s115
                                 + 0.665625 w_k1_INTC_s115
                                 + 0.932036 w_k1_KO_s115
                                 + 0.673167 w_k1_MAT_s115
                                 + 0.763502 w_k1_MSFT_s115
                                 + 0.890121 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.00633 w_k1_ADM_s115
                                 + 0.866127 w_k1_AXP_s115
                                 + 0.839065 w_k1_DIS_s115
                                 + 0.841779 w_k1_FDX_s115
                                 + 1.07725 w_k1_IBM_s115
                                 + 0.77207 w_k1_INTC_s115 + 1.1328 w_k1_KO_s115
                                 + 1.29269 w_k1_MAT_s115
                                 + 0.803808 w_k1_MSFT_s115
                                 + 0.915648 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.28255 w_k1_ADM_s115
                                 + 1.38316 w_k1_AXP_s115
                                 + 1.46271 w_k1_DIS_s115
                                 + 1.29016 w_k1_FDX_s115
                                 + 1.12287 w_k1_IBM_s115
                                 + 1.18065 w_k1_INTC_s115
                                 + 1.44035 w_k1_KO_s115 + 1.41349 w_k1_MAT_s115
                                 + 1.21218 w_k1_MSFT_s115 + 1.2707 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.986506 w_k1_ADM_s115
                                 + 1.09711 w_k1_AXP_s115
                                 + 1.06656 w_k1_DIS_s115 + 1.1861 w_k1_FDX_s115
                                 + 0.741347 w_k1_IBM_s115
                                 + 1.10358 w_k1_INTC_s115
                                 + 1.03683 w_k1_KO_s115
                                 + 0.958029 w_k1_MAT_s115
                                 + 1.08671 w_k1_MSFT_s115
                                 + 1.07838 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.08199 w_k1_ADM_s116
                                 + 1.35677 w_k1_AXP_s116
                                 + 1.52816 w_k1_DIS_s116
                                 + 1.17074 w_k1_FDX_s116
                                 + 1.39154 w_k1_IBM_s116
                                 + 1.27339 w_k1_INTC_s116
                                 + 1.37311 w_k1_KO_s116 + 1.99305 w_k1_MAT_s116
                                 + 1.14612 w_k1_MSFT_s116
                                 + 1.14728 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.978303 w_k1_ADM_s116
                                 + 1.52256 w_k1_AXP_s116
                                 + 1.19741 w_k1_DIS_s116
                                 + 1.20679 w_k1_FDX_s116
                                 + 1.13675 w_k1_IBM_s116
                                 + 1.27627 w_k1_INTC_s116
                                 + 1.17471 w_k1_KO_s116 + 1.53767 w_k1_MAT_s116
                                 + 1.09492 w_k1_MSFT_s116
                                 + 1.01671 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.871826 w_k1_ADM_s116
                                 + 1.22889 w_k1_AXP_s116
                                 + 1.41739 w_k1_DIS_s116 + 1.2406 w_k1_FDX_s116
                                 + 0.833846 w_k1_IBM_s116
                                 + 1.25679 w_k1_INTC_s116 + 1.0527 w_k1_KO_s116
                                 + 1.20816 w_k1_MAT_s116
                                 + 1.10674 w_k1_MSFT_s116
                                 + 1.10922 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.69413 w_k1_ADM_s116
                                 + 0.702918 w_k1_AXP_s116
                                 + 0.773108 w_k1_DIS_s116
                                 + 0.817118 w_k1_FDX_s116
                                 + 0.674665 w_k1_IBM_s116
                                 + 0.76409 w_k1_INTC_s116
                                 + 0.636674 w_k1_KO_s116
                                 + 0.600161 w_k1_MAT_s116
                                 + 0.672695 w_k1_MSFT_s116
                                 + 0.777259 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.01006 w_k1_ADM_s116
                                 + 0.935221 w_k1_AXP_s116
                                 + 1.03749 w_k1_DIS_s116
                                 + 1.01381 w_k1_FDX_s116
                                 + 0.892183 w_k1_IBM_s116
                                 + 0.82072 w_k1_INTC_s116
                                 + 0.927901 w_k1_KO_s116
                                 + 1.12285 w_k1_MAT_s116
                                 + 0.99565 w_k1_MSFT_s116
                                 + 0.958805 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.810286 w_k1_ADM_s116
                                 + 1.29493 w_k1_AXP_s116
                                 + 0.937797 w_k1_DIS_s116
                                 + 0.905229 w_k1_FDX_s116
                                 + 1.38836 w_k1_IBM_s116
                                 + 1.12687 w_k1_INTC_s116 + 1.0365 w_k1_KO_s116
                                 + 1.58236 w_k1_MAT_s116
                                 + 0.755705 w_k1_MSFT_s116
                                 + 1.0782 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.787131 w_k1_ADM_s116
                                 + 0.851535 w_k1_AXP_s116
                                 + 0.764673 w_k1_DIS_s116
                                 + 0.796354 w_k1_FDX_s116
                                 + 0.951039 w_k1_IBM_s116
                                 + 0.846089 w_k1_INTC_s116
                                 + 0.907028 w_k1_KO_s116
                                 + 0.884997 w_k1_MAT_s116
                                 + 0.899708 w_k1_MSFT_s116
                                 + 0.870268 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.817181 w_k1_ADM_s116
                                 + 0.901529 w_k1_AXP_s116
                                 + 0.920958 w_k1_DIS_s116
                                 + 0.921477 w_k1_FDX_s116
                                 + 0.702038 w_k1_IBM_s116
                                 + 0.902874 w_k1_INTC_s116
                                 + 0.774959 w_k1_KO_s116
                                 + 1.01517 w_k1_MAT_s116
                                 + 0.843602 w_k1_MSFT_s116
                                 + 0.920741 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.08182 w_k1_ADM_s117
                                 + 0.959992 w_k1_AXP_s117
                                 + 1.11673 w_k1_DIS_s117
                                 + 1.13261 w_k1_FDX_s117
                                 + 1.11933 w_k1_IBM_s117
                                 + 1.20745 w_k1_INTC_s117
                                 + 1.02907 w_k1_KO_s117 + 1.03699 w_k1_MAT_s117
                                 + 1.00418 w_k1_MSFT_s117
                                 + 1.06657 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.923242 w_k1_ADM_s117
                                 + 1.25364 w_k1_AXP_s117
                                 + 1.21557 w_k1_DIS_s117
                                 + 1.27544 w_k1_FDX_s117
                                 + 0.877757 w_k1_IBM_s117
                                 + 1.1768 w_k1_INTC_s117
                                 + 0.935961 w_k1_KO_s117
                                 + 0.837124 w_k1_MAT_s117
                                 + 1.07214 w_k1_MSFT_s117
                                 + 1.01657 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.43126 w_k1_ADM_s117
                                 + 1.15005 w_k1_AXP_s117
                                 + 1.72987 w_k1_DIS_s117
                                 + 1.52585 w_k1_FDX_s117
                                 + 0.977961 w_k1_IBM_s117
                                 + 1.41511 w_k1_INTC_s117
                                 + 1.21786 w_k1_KO_s117
                                 + 0.791897 w_k1_MAT_s117
                                 + 1.26176 w_k1_MSFT_s117
                                 + 1.32492 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.12881 w_k1_ADM_s117
                                 + 1.02117 w_k1_AXP_s117
                                 + 0.819647 w_k1_DIS_s117
                                 + 0.963022 w_k1_FDX_s117
                                 + 0.899503 w_k1_IBM_s117
                                 + 0.866463 w_k1_INTC_s117
                                 + 0.859034 w_k1_KO_s117
                                 + 0.801201 w_k1_MAT_s117
                                 + 0.871891 w_k1_MSFT_s117
                                 + 0.86647 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.18359 w_k1_ADM_s117
                                 + 1.09857 w_k1_AXP_s117
                                 + 1.26293 w_k1_DIS_s117
                                 + 1.09487 w_k1_FDX_s117
                                 + 1.62385 w_k1_IBM_s117
                                 + 1.04898 w_k1_INTC_s117
                                 + 1.49608 w_k1_KO_s117 + 1.77604 w_k1_MAT_s117
                                 + 1.00123 w_k1_MSFT_s117
                                 + 1.27392 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.03374 w_k1_ADM_s117
                                 + 0.694004 w_k1_AXP_s117
                                 + 0.578094 w_k1_DIS_s117
                                 + 0.721519 w_k1_FDX_s117
                                 + 1.20366 w_k1_IBM_s117
                                 + 0.725738 w_k1_INTC_s117
                                 + 0.91792 w_k1_KO_s117 + 0.90824 w_k1_MAT_s117
                                 + 0.697108 w_k1_MSFT_s117
                                 + 0.905264 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.759483 w_k1_ADM_s117
                                 + 0.926447 w_k1_AXP_s117
                                 + 0.897329 w_k1_DIS_s117
                                 + 1.06777 w_k1_FDX_s117
                                 + 0.62476 w_k1_IBM_s117
                                 + 0.990294 w_k1_INTC_s117
                                 + 0.837065 w_k1_KO_s117
                                 + 0.53275 w_k1_MAT_s117
                                 + 0.7733 w_k1_MSFT_s117 + 0.8209 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.13452 w_k1_ADM_s117
                                 + 0.700366 w_k1_AXP_s117
                                 + 0.5515 w_k1_DIS_s117
                                 + 0.746677 w_k1_FDX_s117
                                 + 0.651221 w_k1_IBM_s117
                                 + 0.808457 w_k1_INTC_s117
                                 + 0.666357 w_k1_KO_s117
                                 + 0.51284 w_k1_MAT_s117
                                 + 0.670236 w_k1_MSFT_s117
                                 + 0.829862 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0296422412653045 w_k1_ADM_s000
                                 + 0.0472919734188286 w_k1_ADM_s001
                                 + 0.0190520251549586 w_k1_ADM_s002
                                 + 0.023507553179536 w_k1_ADM_s003
                                 + 0.086626684818784 w_k1_ADM_s004
                                 + 0.0337003831733866 w_k1_ADM_s005
                                 + 0.02688691328105 w_k1_ADM_s006
                                 + 0.0535232523772706 w_k1_ADM_s007
                                 + 0.0278067928655061 w_k1_ADM_s010
                                 + 0.0246202289058012 w_k1_ADM_s011
                                 + 0.0169269493589656 w_k1_ADM_s012
                                 + 0.0675016628279425 w_k1_ADM_s013
                                 + 0.0107389663417406 w_k1_ADM_s014
                                 + 0.0596243627753828 w_k1_ADM_s015
                                 + 0.0226148810772644 w_k1_ADM_s016
                                 + 0.0353915663194338 w_k1_ADM_s017
                                 + 0.0247396698611759 w_k1_ADM_s100
                                 + 0.0402590657220885 w_k1_ADM_s101
                                 + 0.0318257056341651 w_k1_ADM_s102
                                 + 0.0216108991678549 w_k1_ADM_s103
                                 + 0.0409011745245823 w_k1_ADM_s104
                                 + 0.0477456891947241 w_k1_ADM_s105
                                 + 0.0229814555947624 w_k1_ADM_s106
                                 + 0.0998223299145188 w_k1_ADM_s107
                                 + 0.0200196280132519 w_k1_ADM_s110
                                 + 0.0109236430230642 w_k1_ADM_s111
                                 + 0.00988350259349475 w_k1_ADM_s112
                                 + 0.00435015307988074 w_k1_ADM_s113
                                 + 0.0112397702275823 w_k1_ADM_s114
                                 + 0.0265754296913243 w_k1_ADM_s115
                                 + 0.0136670893528852 w_k1_ADM_s116
                                 + 0.00618395274694641 w_k1_ADM_s117
                                 + 0.0282233642113988 w_k1_AXP_s000
                                 + 0.0600559398720488 w_k1_AXP_s001
                                 + 0.0194954864270377 w_k1_AXP_s002
                                 + 0.0254841127765606 w_k1_AXP_s003
                                 + 0.0785362837723191 w_k1_AXP_s004
                                 + 0.0384805340693977 w_k1_AXP_s005
                                 + 0.029272164424604 w_k1_AXP_s006
                                 + 0.0551352148322704 w_k1_AXP_s007
                                 + 0.0240710411337324 w_k1_AXP_s010
                                 + 0.0192851781015483 w_k1_AXP_s011
                                 + 0.0174303135149956 w_k1_AXP_s012
                                 + 0.0707129091286195 w_k1_AXP_s013
                                 + 0.0138084085592563 w_k1_AXP_s014
                                 + 0.0538503351061214 w_k1_AXP_s015
                                 + 0.0169019184782255 w_k1_AXP_s016
                                 + 0.0369896013892038 w_k1_AXP_s017
                                 + 0.0250481949317976 w_k1_AXP_s100
                                 + 0.0425565763386999 w_k1_AXP_s101
                                 + 0.0325548511717393 w_k1_AXP_s102
                                 + 0.0207944925285771 w_k1_AXP_s103
                                 + 0.0467621059158378 w_k1_AXP_s104
                                 + 0.0514010907282228 w_k1_AXP_s105
                                 + 0.0249975407120048 w_k1_AXP_s106
                                 + 0.0985313049021859 w_k1_AXP_s107
                                 + 0.0229819829968511 w_k1_AXP_s110
                                 + 0.0105358127435732 w_k1_AXP_s111
                                 + 0.00840295770873168 w_k1_AXP_s112
                                 + 0.00518202695436717 w_k1_AXP_s113
                                 + 0.0127767264991365 w_k1_AXP_s114
                                 + 0.0256704631192374 w_k1_AXP_s115
                                 + 0.0168416405595216 w_k1_AXP_s116
                                 + 0.00583735668639654 w_k1_AXP_s117
                                 + 0.0253757775569405 w_k1_DIS_s000
                                 + 0.0558144672108586 w_k1_DIS_s001
                                 + 0.0202047420725334 w_k1_DIS_s002
                                 + 0.0245312534130205 w_k1_DIS_s003
                                 + 0.0886216219784698 w_k1_DIS_s004
                                 + 0.0346700199076399 w_k1_DIS_s005
                                 + 0.0269194824551744 w_k1_DIS_s006
                                 + 0.0578235824033122 w_k1_DIS_s007
                                 + 0.0244783922310867 w_k1_DIS_s010
                                 + 0.0206123190582492 w_k1_DIS_s011
                                 + 0.0168285555812636 w_k1_DIS_s012
                                 + 0.0635043234163772 w_k1_DIS_s013
                                 + 0.0136494886988242 w_k1_DIS_s014
                                 + 0.0536162624183686 w_k1_DIS_s015
                                 + 0.0209361770334445 w_k1_DIS_s016
                                 + 0.0335364080929986 w_k1_DIS_s017
                                 + 0.0249340281438754 w_k1_DIS_s100
                                 + 0.0412453950410333 w_k1_DIS_s101
                                 + 0.0331343179950746 w_k1_DIS_s102
                                 + 0.0232237610701229 w_k1_DIS_s103
                                 + 0.0479023470687819 w_k1_DIS_s104
                                 + 0.0570911050483911 w_k1_DIS_s105
                                 + 0.0272088767750488 w_k1_DIS_s106
                                 + 0.0935266615577342 w_k1_DIS_s107
                                 + 0.0212313881059986 w_k1_DIS_s110
                                 + 0.0115322825471641 w_k1_DIS_s111
                                 + 0.00892934969814715 w_k1_DIS_s112
                                 + 0.00537576338893461 w_k1_DIS_s113
                                 + 0.0137834795628963 w_k1_DIS_s114
                                 + 0.0271725320670931 w_k1_DIS_s115
                                 + 0.0161950877606812 w_k1_DIS_s116
                                 + 0.00588716952094488 w_k1_DIS_s117
                                 + 0.0264419360488007 w_k1_FDX_s000
                                 + 0.0558462882549641 w_k1_FDX_s001
                                 + 0.0202977013333889 w_k1_FDX_s002
                                 + 0.0239000351629816 w_k1_FDX_s003
                                 + 0.0841641880459989 w_k1_FDX_s004
                                 + 0.0349763596845039 w_k1_FDX_s005
                                 + 0.0279843762092158 w_k1_FDX_s006
                                 + 0.0563164911221378 w_k1_FDX_s007
                                 + 0.0239904732955727 w_k1_FDX_s010
                                 + 0.0197078371664427 w_k1_FDX_s011
                                 + 0.0172952505183866 w_k1_FDX_s012
                                 + 0.0645411869753073 w_k1_FDX_s013
                                 + 0.0130314371667886 w_k1_FDX_s014
                                 + 0.0540676351300268 w_k1_FDX_s015
                                 + 0.019493770223191 w_k1_FDX_s016
                                 + 0.033648373419765 w_k1_FDX_s017
                                 + 0.0230745360641274 w_k1_FDX_s100
                                 + 0.0383613582605195 w_k1_FDX_s101
                                 + 0.0331971258469911 w_k1_FDX_s102
                                 + 0.0230202598305976 w_k1_FDX_s103
                                 + 0.0455365750886946 w_k1_FDX_s104
                                 + 0.0526875802169205 w_k1_FDX_s105
                                 + 0.0245103863871261 w_k1_FDX_s106
                                 + 0.0931169534999297 w_k1_FDX_s107
                                 + 0.0214208208186329 w_k1_FDX_s110
                                 + 0.0114829902460623 w_k1_FDX_s111
                                 + 0.00891503136004002 w_k1_FDX_s112
                                 + 0.00541297947112051 w_k1_FDX_s113
                                 + 0.012772528539006 w_k1_FDX_s114
                                 + 0.0256371079126277 w_k1_FDX_s115
                                 + 0.0152210027735966 w_k1_FDX_s116
                                 + 0.00622245977457121 w_k1_FDX_s117
                                 + 0.0259728795108246 w_k1_IBM_s000
                                 + 0.0584568396688386 w_k1_IBM_s001
                                 + 0.0197966984721182 w_k1_IBM_s002
                                 + 0.0249494513758578 w_k1_IBM_s003
                                 + 0.0856950174595229 w_k1_IBM_s004
                                 + 0.0364054772492919 w_k1_IBM_s005
                                 + 0.0279124820768231 w_k1_IBM_s006
                                 + 0.056543399030871 w_k1_IBM_s007
                                 + 0.0251334521547638 w_k1_IBM_s010
                                 + 0.0209258467588419 w_k1_IBM_s011
                                 + 0.01739815241321 w_k1_IBM_s012
                                 + 0.0673617041076012 w_k1_IBM_s013
                                 + 0.012634225503636 w_k1_IBM_s014
                                 + 0.0553748994271188 w_k1_IBM_s015
                                 + 0.0198401503592426 w_k1_IBM_s016
                                 + 0.0342698511500055 w_k1_IBM_s017
                                 + 0.0231956989988128 w_k1_IBM_s100
                                 + 0.0403776183488606 w_k1_IBM_s101
                                 + 0.0333944440955474 w_k1_IBM_s102
                                 + 0.0240419823528417 w_k1_IBM_s103
                                 + 0.0464062838394798 w_k1_IBM_s104
                                 + 0.0528994946186881 w_k1_IBM_s105
                                 + 0.0251408446761265 w_k1_IBM_s106
                                 + 0.0957461610823234 w_k1_IBM_s107
                                 + 0.0217978809626679 w_k1_IBM_s110
                                 + 0.0117578626466684 w_k1_IBM_s111
                                 + 0.00898571042959942 w_k1_IBM_s112
                                 + 0.00569234049661945 w_k1_IBM_s113
                                 + 0.0125042025840529 w_k1_IBM_s114
                                 + 0.0257015414297091 w_k1_IBM_s115
                                 + 0.0161550293726753 w_k1_IBM_s116
                                 + 0.00583481904307795 w_k1_IBM_s117
                                 + 0.0250095189010561 w_k1_INTC_s000
                                 + 0.0552754117315048 w_k1_INTC_s001
                                 + 0.0173405065555501 w_k1_INTC_s002
                                 + 0.0253846987129921 w_k1_INTC_s003
                                 + 0.0739035266125676 w_k1_INTC_s004
                                 + 0.0389275626980902 w_k1_INTC_s005
                                 + 0.0254009484827957 w_k1_INTC_s006
                                 + 0.060381103272914 w_k1_INTC_s007
                                 + 0.0252621152592436 w_k1_INTC_s010
                                 + 0.0187536646774655 w_k1_INTC_s011
                                 + 0.0185608935105624 w_k1_INTC_s012
                                 + 0.064186968456298 w_k1_INTC_s013
                                 + 0.0127474501725518 w_k1_INTC_s014
                                 + 0.0625659011224248 w_k1_INTC_s015
                                 + 0.0169508176762497 w_k1_INTC_s016
                                 + 0.0364299965026817 w_k1_INTC_s017
                                 + 0.0287591462069718 w_k1_INTC_s100
                                 + 0.0374146307264498 w_k1_INTC_s101
                                 + 0.0388185490148608 w_k1_INTC_s102
                                 + 0.0210120649470162 w_k1_INTC_s103
                                 + 0.0497622261843215 w_k1_INTC_s104
                                 + 0.0486099962695031 w_k1_INTC_s105
                                 + 0.0220819375930943 w_k1_INTC_s106
                                 + 0.0901903120997715 w_k1_INTC_s107
                                 + 0.0228005513684101 w_k1_INTC_s110
                                 + 0.0119930331126946 w_k1_INTC_s111
                                 + 0.00926907770044151 w_k1_INTC_s112
                                 + 0.00559135357030287 w_k1_INTC_s113
                                 + 0.0134157740148144 w_k1_INTC_s114
                                 + 0.0242177184669625 w_k1_INTC_s115
                                 + 0.0157677005929699 w_k1_INTC_s116
                                 + 0.00600888206003356 w_k1_INTC_s117
                                 + 0.0282442462377374 w_k1_KO_s000
                                 + 0.0541313171070565 w_k1_KO_s001
                                 + 0.0196012368058793 w_k1_KO_s002
                                 + 0.0271733915874896 w_k1_KO_s003
                                 + 0.0752436084404068 w_k1_KO_s004
                                 + 0.033209876226046 w_k1_KO_s005
                                 + 0.0265506824921621 w_k1_KO_s006
                                 + 0.0618516963057676 w_k1_KO_s007
                                 + 0.0261178468644924 w_k1_KO_s010
                                 + 0.0216022381964771 w_k1_KO_s011
                                 + 0.0180822074120224 w_k1_KO_s012
                                 + 0.0644034894227016 w_k1_KO_s013
                                 + 0.0133619678509454 w_k1_KO_s014
                                 + 0.05742434051908 w_k1_KO_s015
                                 + 0.0176454405058557 w_k1_KO_s016
                                 + 0.0334623043028964 w_k1_KO_s017
                                 + 0.024523837147265 w_k1_KO_s100
                                 + 0.0389786242878819 w_k1_KO_s101
                                 + 0.0357336176080402 w_k1_KO_s102
                                 + 0.024682945894531 w_k1_KO_s103
                                 + 0.0479068515459602 w_k1_KO_s104
                                 + 0.0499150479240063 w_k1_KO_s105
                                 + 0.0255077684903529 w_k1_KO_s106
                                 + 0.0828305724917803 w_k1_KO_s107
                                 + 0.0243852565706717 w_k1_KO_s110
                                 + 0.0120471195064523 w_k1_KO_s111
                                 + 0.0105689017340623 w_k1_KO_s112
                                 + 0.00570065203455157 w_k1_KO_s113
                                 + 0.0112760937658192 w_k1_KO_s114
                                 + 0.0304746030858909 w_k1_KO_s115
                                 + 0.0154192882838301 w_k1_KO_s116
                                 + 0.00574653636254788 w_k1_KO_s117
                                 + 0.0301357630832657 w_k1_MAT_s000
                                 + 0.0624876619974024 w_k1_MAT_s001
                                 + 0.0209009073948585 w_k1_MAT_s002
                                 + 0.0258486980703248 w_k1_MAT_s003
                                 + 0.081719937467891 w_k1_MAT_s004
                                 + 0.0311872930439059 w_k1_MAT_s005
                                 + 0.0291275512197974 w_k1_MAT_s006
                                 + 0.0706242890585599 w_k1_MAT_s007
                                 + 0.0300588585582499 w_k1_MAT_s010
                                 + 0.0216679256669785 w_k1_MAT_s011
                                 + 0.0152081723389649 w_k1_MAT_s012
                                 + 0.0828259651833377 w_k1_MAT_s013
                                 + 0.0135810275422097 w_k1_MAT_s014
                                 + 0.058189718268874 w_k1_MAT_s015
                                 + 0.0189694264500466 w_k1_MAT_s016
                                 + 0.0370961577583216 w_k1_MAT_s017
                                 + 0.0220919682660035 w_k1_MAT_s100
                                 + 0.0363842385498718 w_k1_MAT_s101
                                 + 0.0291886567784967 w_k1_MAT_s102
                                 + 0.0241092105380552 w_k1_MAT_s103
                                 + 0.0423944396690734 w_k1_MAT_s104
                                 + 0.0485869869696896 w_k1_MAT_s105
                                 + 0.0281797871475453 w_k1_MAT_s106
                                 + 0.0864436394024646 w_k1_MAT_s107
                                 + 0.0239211025343485 w_k1_MAT_s110
                                 + 0.0112735580481195 w_k1_MAT_s111
                                 + 0.0110913451424208 w_k1_MAT_s112
                                 + 0.00564081386000453 w_k1_MAT_s113
                                 + 0.0132900430190986 w_k1_MAT_s114
                                 + 0.0287471667673732 w_k1_MAT_s115
                                 + 0.0195366793857312 w_k1_MAT_s116
                                 + 0.00537720199142559 w_k1_MAT_s117
                                 + 0.0234363586067971 w_k1_MSFT_s000
                                 + 0.0529216280396957 w_k1_MSFT_s001
                                 + 0.0197165043467677 w_k1_MSFT_s002
                                 + 0.0248372778125491 w_k1_MSFT_s003
                                 + 0.0846715082337439 w_k1_MSFT_s004
                                 + 0.034321318820388 w_k1_MSFT_s005
                                 + 0.0247330784407934 w_k1_MSFT_s006
                                 + 0.0556730293323244 w_k1_MSFT_s007
                                 + 0.0260465614419345 w_k1_MSFT_s010
                                 + 0.0200470146744658 w_k1_MSFT_s011
                                 + 0.0159240383930921 w_k1_MSFT_s012
                                 + 0.0726127340733489 w_k1_MSFT_s013
                                 + 0.0135379576564305 w_k1_MSFT_s014
                                 + 0.0555764981176704 w_k1_MSFT_s015
                                 + 0.0203484046174724 w_k1_MSFT_s016
                                 + 0.0343604333866728 w_k1_MSFT_s017
                                 + 0.0249269586526872 w_k1_MSFT_s100
                                 + 0.0407355252818104 w_k1_MSFT_s101
                                 + 0.0345908848596908 w_k1_MSFT_s102
                                 + 0.0226758745271344 w_k1_MSFT_s103
                                 + 0.046588432709034 w_k1_MSFT_s104
                                 + 0.0561514314705657 w_k1_MSFT_s105
                                 + 0.0235138361630815 w_k1_MSFT_s106
                                 + 0.0941483841052331 w_k1_MSFT_s107
                                 + 0.021246197909298 w_k1_MSFT_s110
                                 + 0.0118690482009796 w_k1_MSFT_s111
                                 + 0.0098232139271857 w_k1_MSFT_s112
                                 + 0.00553847820888362 w_k1_MSFT_s113
                                 + 0.0124963501646344 w_k1_MSFT_s114
                                 + 0.0249906061045223 w_k1_MSFT_s115
                                 + 0.0143908415021679 w_k1_MSFT_s116
                                 + 0.00539026694619203 w_k1_MSFT_s117
                                 + 0.0261543611916657 w_k1_PG_s000
                                 + 0.0571373500610175 w_k1_PG_s001
                                 + 0.0186537230628608 w_k1_PG_s002
                                 + 0.0265540072403181 w_k1_PG_s003
                                 + 0.0836472892339487 w_k1_PG_s004
                                 + 0.0329685256446333 w_k1_PG_s005
                                 + 0.0260966181856708 w_k1_PG_s006
                                 + 0.0560022294980742 w_k1_PG_s007
                                 + 0.0239746360313524 w_k1_PG_s010
                                 + 0.0201371650548401 w_k1_PG_s011
                                 + 0.0159306112259098 w_k1_PG_s012
                                 + 0.0645690674541524 w_k1_PG_s013
                                 + 0.0131217647137044 w_k1_PG_s014
                                 + 0.0520242928193583 w_k1_PG_s015
                                 + 0.0192881587803201 w_k1_PG_s016
                                 + 0.0341434949231114 w_k1_PG_s017
                                 + 0.0226069887219356 w_k1_PG_s100
                                 + 0.0388631812471781 w_k1_PG_s101
                                 + 0.0342834525350108 w_k1_PG_s102
                                 + 0.0216242052346128 w_k1_PG_s103
                                 + 0.0482973801952691 w_k1_PG_s104
                                 + 0.0525609713393846 w_k1_PG_s105
                                 + 0.0232225050798703 w_k1_PG_s106
                                 + 0.0925028539252189 w_k1_PG_s107
                                 + 0.0196130890467525 w_k1_PG_s110
                                 + 0.0120099587104407 w_k1_PG_s111
                                 + 0.00905651267999886 w_k1_PG_s112
                                 + 0.00572632853449857 w_k1_PG_s113
                                 + 0.0136101728455362 w_k1_PG_s114
                                 + 0.0268333160782661 w_k1_PG_s115
                                 + 0.0153214688343236 w_k1_PG_s116
                                 + 0.00580596017685069 w_k1_PG_s117 >= 1.08
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
