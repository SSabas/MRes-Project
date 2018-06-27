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
 w_k1_ADM_s0_return_constraint:  1.20355 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.925979 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.841051 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.03766 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.24169 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.876068 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.31149 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.822845 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.03262 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.27625 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.826232 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.12956 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.39444 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.930668 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.21673 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.45408 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.04923 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.27397 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.02417 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.668342 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.039 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.800575 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.19637 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.768142 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.03733 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.820659 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.700529 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.886664 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.06576 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.33959 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.9121 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.759387 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.21804 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.10759 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.777651 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.05638 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.07961 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.20205 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.26269 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.923875 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.899819 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.14549 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.18 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.811018 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.56895 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.716682 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.947615 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.21788 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.987801 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.26549 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.4182 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.727427 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.47891 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.44426 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.930682 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.21661 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.86487 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.871586 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.722842 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.706028 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.45359 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.18266 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.07506 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.922206 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.808053 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.796481 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.2485 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.46492 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.976533 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.708106 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.52343 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.13522 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.663977 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.837337 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.44146 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.01875 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.35511 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.879928 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.785876 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.23263 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.30605 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.758995 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.65038 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.722577 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.08255 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.24932 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.970879 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.15798 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.21415 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.728447 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.5552 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.42926 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.04835 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.982854 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.878699 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.764289 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.821794 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.650408 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.598 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.17996 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.09509 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.944117 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.658458 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.730678 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.05626 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.6345 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.829779 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.590128 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.51553 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.22797 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.662052 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.854412 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.48535 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.10017 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20439 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.94039 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.875718 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.1038 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.14479 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.84412 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.54755 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.728247 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.12026 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.981707 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.948201 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.18119 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.28923 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.843222 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.34697 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.33572 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.09819 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.07049 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.840672 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.799932 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.980288 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.811078 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.29631 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.981286 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.16284 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.883212 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.707863 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.848427 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.05629 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.4352 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.838124 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.702778 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.37643 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.05198 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.723086 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.945669 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.13685 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.19914 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.24533 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.916818 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.829276 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.17754 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.18178 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.871805 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.11491 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.36378 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.77988 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.85107 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.2211 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.15445 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.954936 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.822928 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.69293 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.937395 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.824366 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.027 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.11034 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.47218 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.574789 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.689652 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.49109 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.85963 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.764336 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.20497 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.29685 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.677576 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.30627 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.977753 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.16316 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.629191 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.02431 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.43184 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.88528 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.691665 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.95567 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.709843 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.08193 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.957756 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.01869 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.02951 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.996824 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.01172 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.05246 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.744219 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.868566 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.984449 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.948461 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.13847 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.17896 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.801368 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.6155 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.41309 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.14023 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.03722 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.901368 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.04144 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.943265 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.894444 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.32775 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.18293 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.12193 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.01114 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.904373 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.955156 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.24407 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.37197 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.788141 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.64735 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.12673 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.97502 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.730112 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.805231 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.923698 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.954154 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.23279 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.880272 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.877382 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.10112 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.29763 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.866383 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.21464 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.919724 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.904135 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.33222 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.0157 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.12429 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.08661 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.693308 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.67435 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.2236 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.00702 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.996551 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.997095 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.1002 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.805962 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.747088 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.39168 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.50399 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.870454 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.04433 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.961776 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.705428 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.13149 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.29904 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.986094 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.674057 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.21521 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.40187 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.806494 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.837433 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.45918 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.946555 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.33706 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.919177 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.80357 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.23339 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.34439 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.786491 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.4698 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.10813 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.867376 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.95052 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.1773 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.46739 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.26311 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.825115 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.77908 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.09805 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.740793 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.17112 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.894626 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.32232 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.429407 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.6007 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.45922 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.82396 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.698904 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.10546 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.09278 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.469045 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.19629 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.07914 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.06948 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.679628 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.14439 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.66481 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.782188 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.807158 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 2.31824 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.743202 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.13736 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.894741 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.917043 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.15018 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.24871 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.934766 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.29375 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.762518 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.04195 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.959286 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.01155 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.16057 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.16007 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.866508 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.50969 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.23332 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.09719 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.05074 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.827025 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.864344 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.980424 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.784221 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.22973 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.02208 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.03472 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.947668 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.719563 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.814651 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.98279 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.31674 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.881027 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.748493 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.20398 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.06284 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.747531 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.911416 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.18676 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.08382 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13023 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.926893 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.920693 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.12795 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.14774 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.907944 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.2905 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.868356 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.973688 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.07468 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.04603 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.0122 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.16647 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.838793 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.32521 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.12038 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.999821 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.966753 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.852902 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.890453 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.857084 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.810482 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.20644 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.21774 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.05575 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.97322 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.876054 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.821842 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.06139 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.14049 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.944133 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.77114 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.16455 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.17909 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.799199 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.99322 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.14181 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.966102 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.175000430325902 z_k1_s0000
                                 - 0.0760000091129933 z_k1_s0001
                                 - 0.084999947586911 z_k1_s0002
                                 - 0.208000204730267 z_k1_s0003
                                 - 0.123000379143335 z_k1_s0004
                                 - 0.181999569139087 z_k1_s0005
                                 - 0.211000184221573 z_k1_s0006
                                 - 0.160000532869372 z_k1_s0007
                                 - 0.12899974596788 z_k1_s0010
                                 - 0.117999990187628 z_k1_s0011
                                 - 0.178999795877881 z_k1_s0012
                                 - 0.161999651308507 z_k1_s0013
                                 - 0.0649999372830272 z_k1_s0014
                                 - 0.30199979105651 z_k1_s0015
                                 - 0.100999973218228 z_k1_s0016
                                 - 0.220000219604005 z_k1_s0017
                                 - 0.130999986814602 z_k1_s0020
                                 - 0.392999113243855 z_k1_s0021
                                 - 0.359000979212476 z_k1_s0022
                                 - 0.140999911435568 z_k1_s0023
                                 - 0.156000010167005 z_k1_s0024
                                 - 0.248000841639807 z_k1_s0025
                                 - 0.288998965265859 z_k1_s0026
                                 - 0.40100091678052 z_k1_s0027
                                 - 0.0659999843245324 z_k1_s0030
                                 - 0.244999488561257 z_k1_s0031
                                 - 0.112999886211794 z_k1_s0032
                                 - 0.209999561623935 z_k1_s0033
                                 - 0.106999898736824 z_k1_s0034
                                 - 0.0859999425744309 z_k1_s0035
                                 - 0.299999373748478 z_k1_s0036
                                 - 0.0939999258743904 z_k1_s0037
                                 - 0.203999788815438 z_k1_s0040
                                 - 0.197999864379614 z_k1_s0041
                                 - 0.2149998467809 z_k1_s0042
                                 - 0.803999306390103 z_k1_s0043
                                 - 0.304000101410785 z_k1_s0044
                                 - 0.279000612761304 z_k1_s0045
                                 - 0.1799998552724 z_k1_s0046
                                 - 0.173999695036833 z_k1_s0047
                                 - 0.0850000136063853 z_k1_s0050
                                 - 0.164999387565714 z_k1_s0051
                                 - 0.193999496440772 z_k1_s0052
                                 - 0.15099948283284 z_k1_s0053
                                 - 0.217999537268435 z_k1_s0054
                                 - 0.178000421888433 z_k1_s0055
                                 - 0.211000299401576 z_k1_s0056
                                 - 0.226999373953914 z_k1_s0057
                                 - 0.0699999612172898 z_k1_s0060
                                 - 0.175000170792982 z_k1_s0061
                                 - 0.0749999201971308 z_k1_s0062
                                 - 0.202999298480673 z_k1_s0063
                                 - 0.10699987186792 z_k1_s0064
                                 - 0.170999989409303 z_k1_s0065
                                 - 0.0699999612172898 z_k1_s0066
                                 - 0.199999430192672 z_k1_s0067
                                 - 0.762001380896853 z_k1_s0070
                                 - 0.540000585832629 z_k1_s0071
                                 - 0.321000071133992 z_k1_s0072
                                 - 0.90499728699929 z_k1_s0073
                                 - 0.230999640962068 z_k1_s0074
                                 - 0.964000306857171 z_k1_s0075
                                 - 0.604999178424399 z_k1_s0076
                                 - 0.660001824461499 z_k1_s0077
                                 - 0.131000405647005 z_k1_s0100
                                 - 0.223999680673304 z_k1_s0101
                                 - 0.120000311904743 z_k1_s0102
                                 - 0.119000226546167 z_k1_s0103
                                 - 0.565000684978739 z_k1_s0104
                                 - 0.122000271321376 z_k1_s0105
                                 - 0.460001230851601 z_k1_s0106
                                 - 0.372000903514547 z_k1_s0107
                                 - 0.497001916038345 z_k1_s0110
                                 - 0.342000946317417 z_k1_s0111
                                 - 0.862002463829431 z_k1_s0112
                                 - 0.392002127003265 z_k1_s0113
                                 - 0.267000520491246 z_k1_s0114
                                 - 0.258000442488053 z_k1_s0115
                                 - 0.515000726842129 z_k1_s0116
                                 - 0.230000386645148 z_k1_s0117
                                 - 0.402002502645201 z_k1_s0120
                                 - 0.53200328759007 z_k1_s0121
                                 - 0.220000913920434 z_k1_s0122
                                 - 0.158000558400649 z_k1_s0123
                                 - 0.258000728918052 z_k1_s0124
                                 - 0.328000774809942 z_k1_s0125
                                 - 0.291001135397191 z_k1_s0126
                                 - 0.260001569889737 z_k1_s0127
                                 - 0.39999949326522 z_k1_s0130
                                 - 0.31300039315029 z_k1_s0131
                                 - 0.0880000752384087 z_k1_s0132
                                 - 0.103000075685861 z_k1_s0133
                                 - 0.14400006653556 z_k1_s0134
                                 - 0.145999993981863 z_k1_s0135
                                 - 0.130999993534411 z_k1_s0136
                                 - 0.231000722650993 z_k1_s0137
                                 - 0.585999592647212 z_k1_s0140
                                 - 0.227000083071124 z_k1_s0141
                                 - 0.275000011618347 z_k1_s0142
                                 - 0.275000011618347 z_k1_s0143
                                 - 0.33699870035884 z_k1_s0144
                                 - 0.267999877521916 z_k1_s0145
                                 - 0.55799905626149 z_k1_s0146
                                 - 0.490000338752604 z_k1_s0147
                                 - 0.14500014017711 z_k1_s0150
                                 - 0.406000880536394 z_k1_s0151
                                 - 0.323000880835784 z_k1_s0152
                                 - 0.188000167322032 z_k1_s0153
                                 - 0.0890001088790764 z_k1_s0154
                                 - 0.175000860369614 z_k1_s0155
                                 - 0.124000106652826 z_k1_s0156
                                 - 0.293000334943453 z_k1_s0157
                                 - 0.118000104474559 z_k1_s0160
                                 - 0.176999350711207 z_k1_s0161
                                 - 0.461000121277502 z_k1_s0162
                                 - 0.198000503169433 z_k1_s0163
                                 - 0.209000799790167 z_k1_s0164
                                 - 0.207000306222416 z_k1_s0165
                                 - 0.138000042948151 z_k1_s0166
                                 - 0.104000034702956 z_k1_s0167
                                 - 0.207999475233823 z_k1_s0170
                                 - 0.110999724619961 z_k1_s0171
                                 - 0.385998382599038 z_k1_s0172
                                 - 0.354999687296613 z_k1_s0173
                                 - 0.501997959364391 z_k1_s0174
                                 - 0.314000317705899 z_k1_s0175
                                 - 0.436998426478042 z_k1_s0176
                                 - 0.11199979204879 z_k1_s0177
                                 - 0.32700053019267 z_k1_s1000
                                 - 0.4279987418064 z_k1_s1001
                                 - 0.332000952391186 z_k1_s1002
                                 - 0.211000084783742 z_k1_s1003
                                 - 0.937999894070081 z_k1_s1004
                                 - 0.231999923618204 z_k1_s1005
                                 - 0.499000868226714 z_k1_s1006
                                 - 0.256999777811595 z_k1_s1007
                                 - 0.280000536447261 z_k1_s1010
                                 - 0.110999888513292 z_k1_s1011
                                 - 0.244998848642706 z_k1_s1012
                                 - 0.403998613018565 z_k1_s1013
                                 - 0.130999926830953 z_k1_s1014
                                 - 0.191999935649911 z_k1_s1015
                                 - 0.507998812270405 z_k1_s1016
                                 - 0.289999475106993 z_k1_s1017
                                 - 1.07899668978095 z_k1_s1020
                                 - 1.12100111385028 z_k1_s1021
                                 - 0.425000207101487 z_k1_s1022
                                 - 1.22499916202193 z_k1_s1023
                                 - 0.710998905173544 z_k1_s1024
                                 - 0.864997057427727 z_k1_s1025
                                 - 0.738997337219758 z_k1_s1026
                                 - 0.611000052208491 z_k1_s1027
                                 - 0.90900089750611 z_k1_s1030
                                 - 0.376999546185141 z_k1_s1031
                                 - 0.48299964118344 z_k1_s1032
                                 - 0.518998338627307 z_k1_s1033
                                 - 1.06499704206146 z_k1_s1034
                                 - 0.758000845029221 z_k1_s1035
                                 - 0.39399992044119 z_k1_s1036
                                 - 0.373999939339595 z_k1_s1037
                                 - 0.173999741212061 z_k1_s1040
                                 - 0.440998297764377 z_k1_s1041
                                 - 0.486000417635993 z_k1_s1042
                                 - 0.321998757097799 z_k1_s1043
                                 - 0.342000056040382 z_k1_s1044
                                 - 0.393998937895605 z_k1_s1045
                                 - 0.846999941623599 z_k1_s1046
                                 - 0.204999667425157 z_k1_s1047
                                 - 2.33999640321062 z_k1_s1050
                                 - 0.669998762896481 z_k1_s1051
                                 - 0.49899895665426 z_k1_s1052
                                 - 0.600999628885402 z_k1_s1053
                                 - 0.685999290979229 z_k1_s1054
                                 - 0.741999343071263 z_k1_s1055
                                 - 2.21299625299837 z_k1_s1056
                                 - 1.23000018191561 z_k1_s1057
                                 - 0.40800001933561 z_k1_s1060
                                 - 0.366999999863165 z_k1_s1061
                                 - 0.199000047995543 z_k1_s1062
                                 - 0.189999973945516 z_k1_s1063
                                 - 0.235999955090223 z_k1_s1064
                                 - 0.621000818253225 z_k1_s1065
                                 - 0.147000017484151 z_k1_s1066
                                 - 0.215999870001248 z_k1_s1067
                                 - 0.250999788709977 z_k1_s1070
                                 - 0.805998340343846 z_k1_s1071
                                 - 0.20999998037484 z_k1_s1072
                                 - 0.177999803698721 z_k1_s1073
                                 - 0.33100005890032 z_k1_s1074
                                 - 0.352000228437759 z_k1_s1075
                                 - 0.865999314736402 z_k1_s1076
                                 - 0.436000906587517 z_k1_s1077
                                 - 0.471000199346318 z_k1_s1100
                                 - 0.558000106048457 z_k1_s1101
                                 - 0.526000294461519 z_k1_s1102
                                 - 1.00800204499143 z_k1_s1103
                                 - 1.05300287728596 z_k1_s1104
                                 - 1.49199879624372 z_k1_s1105
                                 - 0.374000391423275 z_k1_s1106
                                 - 0.902002070609483 z_k1_s1107
                                 - 0.174000090772785 z_k1_s1110
                                 - 0.297999389033682 z_k1_s1111
                                 - 0.223999976084474 z_k1_s1112
                                 - 0.196000149173951 z_k1_s1113
                                 - 0.238999624157911 z_k1_s1114
                                 - 0.386000076238449 z_k1_s1115
                                 - 0.560001074211431 z_k1_s1116
                                 - 0.191000115282773 z_k1_s1117
                                 - 0.671001896160465 z_k1_s1120
                                 - 0.863002661336695 z_k1_s1121
                                 - 0.903004065502086 z_k1_s1122
                                 - 0.441001290731525 z_k1_s1123
                                 - 0.680003954752826 z_k1_s1124
                                 - 0.371001322949443 z_k1_s1125
                                 - 0.296000930695953 z_k1_s1126
                                 - 0.754002071345563 z_k1_s1127
                                 - 0.496001166128337 z_k1_s1130
                                 - 0.533999358601617 z_k1_s1131
                                 - 0.691000750539401 z_k1_s1132
                                 - 0.788996331759482 z_k1_s1133
                                 - 0.425998985667458 z_k1_s1134
                                 - 0.426999326479141 z_k1_s1135
                                 - 0.234999380905876 z_k1_s1136
                                 - 1.01399691653936 z_k1_s1137
                                 - 0.245999793041521 z_k1_s1140
                                 - 0.510001470649227 z_k1_s1141
                                 - 0.21000016171388 z_k1_s1142
                                 - 0.308001145874729 z_k1_s1143
                                 - 0.121000119836123 z_k1_s1144
                                 - 0.209000273371562 z_k1_s1145
                                 - 0.438001234392832 z_k1_s1146
                                 - 0.392001015840059 z_k1_s1147
                                 - 0.398001056377306 z_k1_s1150
                                 - 0.716001619541375 z_k1_s1151
                                 - 0.216000743314247 z_k1_s1152
                                 - 0.361000963099027 z_k1_s1153
                                 - 0.82500048968733 z_k1_s1154
                                 - 0.821999846057744 z_k1_s1155
                                 - 0.815003247267844 z_k1_s1156
                                 - 0.750004010997549 z_k1_s1157
                                 - 0.173000637891074 z_k1_s1160
                                 - 0.0950000686230153 z_k1_s1161
                                 - 0.127000162045606 z_k1_s1162
                                 - 0.174001001336051 z_k1_s1163
                                 - 0.164000402890669 z_k1_s1164
                                 - 0.29400059266967 z_k1_s1165
                                 - 0.393000141669742 z_k1_s1166
                                 - 0.0980000963564106 z_k1_s1167
                                 - 0.632999713290541 z_k1_s1170
                                 - 0.424000780914369 z_k1_s1171
                                 - 1.15300037682722 z_k1_s1172
                                 - 0.276999888699643 z_k1_s1173
                                 - 0.245999894119543 z_k1_s1174
                                 - 0.301999695715973 z_k1_s1175
                                 - 0.371999934961199 z_k1_s1176
                                 - 0.490999562078582 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.879489 w_k1_ADM_s000
                                 + 1.11334 w_k1_AXP_s000
                                 + 0.886132 w_k1_DIS_s000
                                 + 0.952368 w_k1_FDX_s000
                                 + 0.84721 w_k1_IBM_s000
                                 + 1.27607 w_k1_INTC_s000
                                 + 0.837855 w_k1_KO_s000
                                 + 1.02486 w_k1_MAT_s000
                                 + 0.957218 w_k1_MSFT_s000
                                 + 0.878871 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.03034 w_k1_ADM_s000
                                 + 1.40278 w_k1_AXP_s000 + 1.1125 w_k1_DIS_s000
                                 + 0.886697 w_k1_FDX_s000
                                 + 1.89949 w_k1_IBM_s000
                                 + 1.24588 w_k1_INTC_s000
                                 + 1.46858 w_k1_KO_s000 + 2.03282 w_k1_MAT_s000
                                 + 1.00142 w_k1_MSFT_s000
                                 + 1.12455 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.974086 w_k1_ADM_s000
                                 + 0.971587 w_k1_AXP_s000
                                 + 1.2653 w_k1_DIS_s000 + 1.14104 w_k1_FDX_s000
                                 + 0.675171 w_k1_IBM_s000
                                 + 1.57137 w_k1_INTC_s000
                                 + 1.01584 w_k1_KO_s000
                                 + 0.675929 w_k1_MAT_s000
                                 + 1.29059 w_k1_MSFT_s000
                                 + 0.999965 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.775336 w_k1_ADM_s000
                                 + 0.765618 w_k1_AXP_s000
                                 + 0.778989 w_k1_DIS_s000
                                 + 0.750514 w_k1_FDX_s000
                                 + 0.949636 w_k1_IBM_s000
                                 + 1.23836 w_k1_INTC_s000
                                 + 0.906635 w_k1_KO_s000
                                 + 0.711454 w_k1_MAT_s000
                                 + 0.857941 w_k1_MSFT_s000
                                 + 0.784833 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.00319 w_k1_ADM_s000
                                 + 1.46842 w_k1_AXP_s000
                                 + 1.67062 w_k1_DIS_s000 + 1.3724 w_k1_FDX_s000
                                 + 1.13367 w_k1_IBM_s000
                                 + 1.62558 w_k1_INTC_s000
                                 + 1.26805 w_k1_KO_s000 + 1.15878 w_k1_MAT_s000
                                 + 1.42411 w_k1_MSFT_s000
                                 + 1.09195 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.17044 w_k1_ADM_s000
                                 + 1.00363 w_k1_AXP_s000
                                 + 1.18357 w_k1_DIS_s000
                                 + 1.04724 w_k1_FDX_s000
                                 + 0.982539 w_k1_IBM_s000
                                 + 1.07202 w_k1_INTC_s000
                                 + 1.07967 w_k1_KO_s000 + 1.04324 w_k1_MAT_s000
                                 + 1.09398 w_k1_MSFT_s000 + 0.9427 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.10324 w_k1_ADM_s000
                                 + 1.30057 w_k1_AXP_s000
                                 + 1.29569 w_k1_DIS_s000
                                 + 1.05781 w_k1_FDX_s000
                                 + 1.38513 w_k1_IBM_s000
                                 + 1.13299 w_k1_INTC_s000
                                 + 1.30715 w_k1_KO_s000 + 1.1626 w_k1_MAT_s000
                                 + 1.08672 w_k1_MSFT_s000
                                 + 1.18742 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.866018 w_k1_ADM_s000
                                 + 0.673616 w_k1_AXP_s000
                                 + 0.859399 w_k1_DIS_s000
                                 + 0.903723 w_k1_FDX_s000
                                 + 0.627421 w_k1_IBM_s000
                                 + 1.19528 w_k1_INTC_s000
                                 + 0.792654 w_k1_KO_s000
                                 + 0.332046 w_k1_MAT_s000
                                 + 1.03036 w_k1_MSFT_s000
                                 + 0.870694 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.898656 w_k1_ADM_s001
                                 + 0.819271 w_k1_AXP_s001
                                 + 0.824837 w_k1_DIS_s001
                                 + 0.819232 w_k1_FDX_s001
                                 + 0.827136 w_k1_IBM_s001
                                 + 0.802722 w_k1_INTC_s001
                                 + 0.813955 w_k1_KO_s001
                                 + 1.01778 w_k1_MAT_s001
                                 + 0.8936 w_k1_MSFT_s001
                                 + 0.838524 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.68767 w_k1_ADM_s001
                                 + 0.728168 w_k1_AXP_s001
                                 + 0.606081 w_k1_DIS_s001
                                 + 0.667415 w_k1_FDX_s001
                                 + 0.71541 w_k1_IBM_s001
                                 + 0.66943 w_k1_INTC_s001
                                 + 0.757053 w_k1_KO_s001
                                 + 0.81933 w_k1_MAT_s001
                                 + 0.702032 w_k1_MSFT_s001
                                 + 0.7512 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.979972 w_k1_ADM_s001
                                 + 0.999252 w_k1_AXP_s001
                                 + 1.0209 w_k1_DIS_s001
                                 + 0.972725 w_k1_FDX_s001
                                 + 1.02207 w_k1_IBM_s001
                                 + 1.00451 w_k1_INTC_s001
                                 + 1.04242 w_k1_KO_s001 + 1.39204 w_k1_MAT_s001
                                 + 1.04135 w_k1_MSFT_s001
                                 + 1.12467 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.02618 w_k1_ADM_s001
                                 + 0.896735 w_k1_AXP_s001
                                 + 0.759491 w_k1_DIS_s001
                                 + 0.988756 w_k1_FDX_s001
                                 + 0.718777 w_k1_IBM_s001
                                 + 0.787902 w_k1_INTC_s001
                                 + 0.760461 w_k1_KO_s001
                                 + 0.878284 w_k1_MAT_s001
                                 + 0.883665 w_k1_MSFT_s001
                                 + 0.98788 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.03734 w_k1_ADM_s001
                                 + 0.771066 w_k1_AXP_s001
                                 + 0.933523 w_k1_DIS_s001
                                 + 0.929236 w_k1_FDX_s001
                                 + 0.927535 w_k1_IBM_s001
                                 + 1.0701 w_k1_INTC_s001 + 1.064 w_k1_KO_s001
                                 + 0.706813 w_k1_MAT_s001
                                 + 1.13815 w_k1_MSFT_s001
                                 + 0.996055 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.02781 w_k1_ADM_s001
                                 + 1.54051 w_k1_AXP_s001
                                 + 1.36862 w_k1_DIS_s001 + 1.1684 w_k1_FDX_s001
                                 + 1.30803 w_k1_IBM_s001
                                 + 1.12405 w_k1_INTC_s001
                                 + 1.24874 w_k1_KO_s001 + 1.41429 w_k1_MAT_s001
                                 + 1.26344 w_k1_MSFT_s001
                                 + 1.27309 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.10573 w_k1_ADM_s001
                                 + 1.45892 w_k1_AXP_s001
                                 + 1.34031 w_k1_DIS_s001
                                 + 1.25119 w_k1_FDX_s001
                                 + 0.870598 w_k1_IBM_s001
                                 + 1.20117 w_k1_INTC_s001
                                 + 1.09407 w_k1_KO_s001 + 1.09625 w_k1_MAT_s001
                                 + 1.15109 w_k1_MSFT_s001 + 1.2368 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.02453 w_k1_ADM_s001
                                 + 0.915256 w_k1_AXP_s001
                                 + 0.969071 w_k1_DIS_s001
                                 + 0.821555 w_k1_FDX_s001
                                 + 1.30811 w_k1_IBM_s001
                                 + 0.835476 w_k1_INTC_s001
                                 + 1.10492 w_k1_KO_s001 + 1.51874 w_k1_MAT_s001
                                 + 0.8792 w_k1_MSFT_s001
                                 + 0.975115 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.35567 w_k1_ADM_s002
                                 + 1.26641 w_k1_AXP_s002
                                 + 1.71834 w_k1_DIS_s002 + 1.2742 w_k1_FDX_s002
                                 + 0.991281 w_k1_IBM_s002
                                 + 1.28843 w_k1_INTC_s002
                                 + 1.31629 w_k1_KO_s002
                                 + 0.808657 w_k1_MAT_s002
                                 + 1.30757 w_k1_MSFT_s002
                                 + 1.11384 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.853525 w_k1_ADM_s002
                                 + 0.754245 w_k1_AXP_s002
                                 + 0.734483 w_k1_DIS_s002
                                 + 0.86356 w_k1_FDX_s002
                                 + 0.776616 w_k1_IBM_s002
                                 + 1.13635 w_k1_INTC_s002
                                 + 0.743654 w_k1_KO_s002
                                 + 0.729036 w_k1_MAT_s002
                                 + 0.890617 w_k1_MSFT_s002
                                 + 0.899847 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.14232 w_k1_ADM_s002
                                 + 1.27146 w_k1_AXP_s002
                                 + 0.901994 w_k1_DIS_s002
                                 + 1.12551 w_k1_FDX_s002
                                 + 0.840155 w_k1_IBM_s002
                                 + 1.40846 w_k1_INTC_s002
                                 + 0.819475 w_k1_KO_s002
                                 + 0.569792 w_k1_MAT_s002
                                 + 0.943685 w_k1_MSFT_s002
                                 + 0.924274 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.103 w_k1_ADM_s002
                                 + 1.35734 w_k1_AXP_s002 + 1.3122 w_k1_DIS_s002
                                 + 1.2879 w_k1_FDX_s002
                                 + 0.957787 w_k1_IBM_s002
                                 + 1.36328 w_k1_INTC_s002
                                 + 1.23304 w_k1_KO_s002
                                 + 0.966619 w_k1_MAT_s002
                                 + 1.38646 w_k1_MSFT_s002
                                 + 1.06415 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.32121 w_k1_ADM_s002
                                 + 1.19562 w_k1_AXP_s002
                                 + 1.53716 w_k1_DIS_s002
                                 + 1.30232 w_k1_FDX_s002
                                 + 1.49919 w_k1_IBM_s002
                                 + 1.43293 w_k1_INTC_s002
                                 + 1.41178 w_k1_KO_s002 + 1.40753 w_k1_MAT_s002
                                 + 1.31909 w_k1_MSFT_s002
                                 + 1.35972 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.927018 w_k1_ADM_s002
                                 + 0.993857 w_k1_AXP_s002
                                 + 0.980264 w_k1_DIS_s002
                                 + 0.895254 w_k1_FDX_s002
                                 + 1.4696 w_k1_IBM_s002
                                 + 1.05927 w_k1_INTC_s002
                                 + 1.16692 w_k1_KO_s002 + 1.12813 w_k1_MAT_s002
                                 + 0.87994 w_k1_MSFT_s002
                                 + 1.05419 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.0876 w_k1_ADM_s002
                                 + 1.03791 w_k1_AXP_s002
                                 + 1.30356 w_k1_DIS_s002
                                 + 1.15911 w_k1_FDX_s002
                                 + 0.950154 w_k1_IBM_s002
                                 + 1.01781 w_k1_INTC_s002 + 1.004 w_k1_KO_s002
                                 + 0.890963 w_k1_MAT_s002
                                 + 1.17667 w_k1_MSFT_s002
                                 + 0.955404 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.998429 w_k1_ADM_s002
                                 + 0.769681 w_k1_AXP_s002
                                 + 0.905422 w_k1_DIS_s002
                                 + 0.886238 w_k1_FDX_s002
                                 + 1.05091 w_k1_IBM_s002
                                 + 0.973318 w_k1_INTC_s002
                                 + 0.991271 w_k1_KO_s002
                                 + 0.87846 w_k1_MAT_s002
                                 + 1.06715 w_k1_MSFT_s002
                                 + 0.953356 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.38642 w_k1_ADM_s003
                                 + 1.77987 w_k1_AXP_s003
                                 + 1.62364 w_k1_DIS_s003
                                 + 1.58952 w_k1_FDX_s003
                                 + 1.03896 w_k1_IBM_s003
                                 + 1.43701 w_k1_INTC_s003
                                 + 1.31624 w_k1_KO_s003 + 1.11554 w_k1_MAT_s003
                                 + 1.36219 w_k1_MSFT_s003
                                 + 1.16597 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.874324 w_k1_ADM_s003
                                 + 0.807065 w_k1_AXP_s003
                                 + 0.730872 w_k1_DIS_s003
                                 + 0.938914 w_k1_FDX_s003
                                 + 0.707324 w_k1_IBM_s003
                                 + 0.885197 w_k1_INTC_s003
                                 + 0.702857 w_k1_KO_s003
                                 + 0.49229 w_k1_MAT_s003
                                 + 0.970326 w_k1_MSFT_s003
                                 + 0.799278 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.992582 w_k1_ADM_s003
                                 + 1.2317 w_k1_AXP_s003 + 1.22686 w_k1_DIS_s003
                                 + 1.1668 w_k1_FDX_s003 + 1.39632 w_k1_IBM_s003
                                 + 0.923008 w_k1_INTC_s003
                                 + 1.17844 w_k1_KO_s003 + 1.11566 w_k1_MAT_s003
                                 + 0.960365 w_k1_MSFT_s003
                                 + 1.12807 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.866445 w_k1_ADM_s003
                                 + 1.12502 w_k1_AXP_s003
                                 + 0.713926 w_k1_DIS_s003
                                 + 0.895382 w_k1_FDX_s003
                                 + 0.946327 w_k1_IBM_s003
                                 + 0.947712 w_k1_INTC_s003
                                 + 0.92737 w_k1_KO_s003 + 1.25086 w_k1_MAT_s003
                                 + 0.977983 w_k1_MSFT_s003
                                 + 0.912231 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.11206 w_k1_ADM_s003
                                 + 1.60374 w_k1_AXP_s003
                                 + 1.50068 w_k1_DIS_s003
                                 + 1.36215 w_k1_FDX_s003
                                 + 1.56994 w_k1_IBM_s003
                                 + 1.22172 w_k1_INTC_s003
                                 + 1.43053 w_k1_KO_s003 + 2.05485 w_k1_MAT_s003
                                 + 1.35325 w_k1_MSFT_s003
                                 + 1.49036 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.736669 w_k1_ADM_s003
                                 + 1.01813 w_k1_AXP_s003
                                 + 0.718906 w_k1_DIS_s003
                                 + 0.837052 w_k1_FDX_s003
                                 + 1.25447 w_k1_IBM_s003
                                 + 0.987204 w_k1_INTC_s003
                                 + 1.0168 w_k1_KO_s003 + 1.32304 w_k1_MAT_s003
                                 + 1.03239 w_k1_MSFT_s003
                                 + 0.903187 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.887188 w_k1_ADM_s003
                                 + 0.909771 w_k1_AXP_s003
                                 + 1.10449 w_k1_DIS_s003
                                 + 1.01601 w_k1_FDX_s003 + 1.0134 w_k1_IBM_s003
                                 + 0.81808 w_k1_INTC_s003
                                 + 1.03282 w_k1_KO_s003 + 1.0135 w_k1_MAT_s003
                                 + 1.22547 w_k1_MSFT_s003
                                 + 1.00176 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.968362 w_k1_ADM_s003
                                 + 1.41371 w_k1_AXP_s003
                                 + 1.27575 w_k1_DIS_s003
                                 + 1.31877 w_k1_FDX_s003
                                 + 1.02949 w_k1_IBM_s003
                                 + 1.37027 w_k1_INTC_s003
                                 + 1.25534 w_k1_KO_s003 + 1.21 w_k1_MAT_s003
                                 + 1.40509 w_k1_MSFT_s003
                                 + 1.22192 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.43034 w_k1_ADM_s004
                                 + 0.942929 w_k1_AXP_s004
                                 + 0.914745 w_k1_DIS_s004
                                 + 0.944304 w_k1_FDX_s004
                                 + 0.981766 w_k1_IBM_s004
                                 + 0.941853 w_k1_INTC_s004
                                 + 0.977779 w_k1_KO_s004
                                 + 0.834674 w_k1_MAT_s004
                                 + 0.962655 w_k1_MSFT_s004
                                 + 0.901033 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.01726 w_k1_ADM_s004
                                 + 0.91023 w_k1_AXP_s004
                                 + 0.944322 w_k1_DIS_s004
                                 + 0.819253 w_k1_FDX_s004
                                 + 1.37438 w_k1_IBM_s004
                                 + 0.818836 w_k1_INTC_s004
                                 + 1.08124 w_k1_KO_s004 + 1.38149 w_k1_MAT_s004
                                 + 0.790156 w_k1_MSFT_s004
                                 + 1.00722 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.07961 w_k1_ADM_s004
                                 + 0.954289 w_k1_AXP_s004
                                 + 1.07531 w_k1_DIS_s004
                                 + 1.17304 w_k1_FDX_s004
                                 + 0.710216 w_k1_IBM_s004
                                 + 1.01818 w_k1_INTC_s004
                                 + 0.837292 w_k1_KO_s004
                                 + 0.896775 w_k1_MAT_s004
                                 + 0.938844 w_k1_MSFT_s004
                                 + 0.869712 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.67916 w_k1_ADM_s004
                                 + 1.15911 w_k1_AXP_s004
                                 + 1.55344 w_k1_DIS_s004
                                 + 1.21138 w_k1_FDX_s004
                                 + 1.34001 w_k1_IBM_s004
                                 + 1.05701 w_k1_INTC_s004
                                 + 1.28182 w_k1_KO_s004 + 1.17531 w_k1_MAT_s004
                                 + 1.07212 w_k1_MSFT_s004
                                 + 1.03741 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.18441 w_k1_ADM_s004
                                 + 0.975688 w_k1_AXP_s004
                                 + 1.04786 w_k1_DIS_s004
                                 + 0.96029 w_k1_FDX_s004
                                 + 0.92746 w_k1_IBM_s004
                                 + 0.820864 w_k1_INTC_s004
                                 + 0.926402 w_k1_KO_s004
                                 + 0.915787 w_k1_MAT_s004
                                 + 0.921395 w_k1_MSFT_s004
                                 + 0.940059 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.12002 w_k1_ADM_s004
                                 + 0.687203 w_k1_AXP_s004
                                 + 0.638044 w_k1_DIS_s004
                                 + 0.795524 w_k1_FDX_s004
                                 + 0.830835 w_k1_IBM_s004
                                 + 0.817018 w_k1_INTC_s004
                                 + 0.691389 w_k1_KO_s004
                                 + 0.698266 w_k1_MAT_s004
                                 + 0.742305 w_k1_MSFT_s004
                                 + 0.827258 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.14486 w_k1_ADM_s004
                                 + 1.37907 w_k1_AXP_s004
                                 + 0.932074 w_k1_DIS_s004
                                 + 1.03045 w_k1_FDX_s004
                                 + 0.926346 w_k1_IBM_s004
                                 + 1.15173 w_k1_INTC_s004
                                 + 0.981407 w_k1_KO_s004
                                 + 1.27527 w_k1_MAT_s004
                                 + 0.953661 w_k1_MSFT_s004
                                 + 0.966158 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.17148 w_k1_ADM_s004
                                 + 0.838052 w_k1_AXP_s004
                                 + 0.894378 w_k1_DIS_s004
                                 + 0.937063 w_k1_FDX_s004
                                 + 0.755226 w_k1_IBM_s004
                                 + 0.878651 w_k1_INTC_s004
                                 + 0.789554 w_k1_KO_s004
                                 + 0.82675 w_k1_MAT_s004
                                 + 0.923816 w_k1_MSFT_s004
                                 + 0.831049 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.09012 w_k1_ADM_s005
                                 + 0.99011 w_k1_AXP_s005
                                 + 1.20121 w_k1_DIS_s005
                                 + 0.945107 w_k1_FDX_s005
                                 + 1.58067 w_k1_IBM_s005
                                 + 0.783051 w_k1_INTC_s005
                                 + 1.19236 w_k1_KO_s005 + 1.46469 w_k1_MAT_s005
                                 + 0.895247 w_k1_MSFT_s005
                                 + 1.07521 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.72734 w_k1_ADM_s005
                                 + 0.687123 w_k1_AXP_s005
                                 + 0.567391 w_k1_DIS_s005
                                 + 0.711786 w_k1_FDX_s005
                                 + 0.810473 w_k1_IBM_s005
                                 + 0.643358 w_k1_INTC_s005
                                 + 0.667629 w_k1_KO_s005
                                 + 0.601264 w_k1_MAT_s005
                                 + 0.651805 w_k1_MSFT_s005
                                 + 0.829562 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.16095 w_k1_ADM_s005
                                 + 1.05573 w_k1_AXP_s005
                                 + 1.21821 w_k1_DIS_s005
                                 + 1.24161 w_k1_FDX_s005
                                 + 0.840162 w_k1_IBM_s005
                                 + 1.1279 w_k1_INTC_s005
                                 + 0.957638 w_k1_KO_s005
                                 + 0.686954 w_k1_MAT_s005
                                 + 1.09194 w_k1_MSFT_s005
                                 + 1.24572 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.13096 w_k1_ADM_s005
                                 + 1.1623 w_k1_AXP_s005 + 1.26306 w_k1_DIS_s005
                                 + 1.04289 w_k1_FDX_s005
                                 + 1.31622 w_k1_IBM_s005
                                 + 1.13051 w_k1_INTC_s005
                                 + 1.15858 w_k1_KO_s005 + 1.06747 w_k1_MAT_s005
                                 + 1.06516 w_k1_MSFT_s005
                                 + 1.27857 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.23912 w_k1_ADM_s005
                                 + 1.54424 w_k1_AXP_s005
                                 + 1.59392 w_k1_DIS_s005 + 1.2087 w_k1_FDX_s005
                                 + 1.34982 w_k1_IBM_s005
                                 + 1.18049 w_k1_INTC_s005
                                 + 1.20836 w_k1_KO_s005 + 1.33915 w_k1_MAT_s005
                                 + 1.14402 w_k1_MSFT_s005 + 1.1509 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.14246 w_k1_ADM_s005
                                 + 0.888161 w_k1_AXP_s005
                                 + 0.78644 w_k1_DIS_s005
                                 + 0.97347 w_k1_FDX_s005
                                 + 0.703333 w_k1_IBM_s005
                                 + 0.850687 w_k1_INTC_s005
                                 + 0.748914 w_k1_KO_s005
                                 + 0.478128 w_k1_MAT_s005
                                 + 0.891849 w_k1_MSFT_s005
                                 + 0.898843 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.11477 w_k1_ADM_s005
                                 + 0.920652 w_k1_AXP_s005
                                 + 1.00778 w_k1_DIS_s005
                                 + 0.964219 w_k1_FDX_s005
                                 + 0.89267 w_k1_IBM_s005
                                 + 0.795788 w_k1_INTC_s005
                                 + 0.850854 w_k1_KO_s005
                                 + 0.767568 w_k1_MAT_s005
                                 + 0.923101 w_k1_MSFT_s005
                                 + 1.08243 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.932895 w_k1_ADM_s005
                                 + 0.930184 w_k1_AXP_s005
                                 + 0.919488 w_k1_DIS_s005
                                 + 0.883609 w_k1_FDX_s005
                                 + 1.07187 w_k1_IBM_s005
                                 + 0.952688 w_k1_INTC_s005
                                 + 1.07876 w_k1_KO_s005
                                 + 0.995923 w_k1_MAT_s005
                                 + 1.01048 w_k1_MSFT_s005
                                 + 1.04917 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.05967 w_k1_ADM_s006
                                 + 1.32678 w_k1_AXP_s006 + 1.2978 w_k1_DIS_s006
                                 + 1.13673 w_k1_FDX_s006
                                 + 1.21943 w_k1_IBM_s006
                                 + 1.07921 w_k1_INTC_s006
                                 + 1.01241 w_k1_KO_s006 + 1.21488 w_k1_MAT_s006
                                 + 1.06653 w_k1_MSFT_s006
                                 + 0.99799 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.758308 w_k1_ADM_s006
                                 + 0.833137 w_k1_AXP_s006
                                 + 0.949296 w_k1_DIS_s006
                                 + 0.746272 w_k1_FDX_s006
                                 + 1.04254 w_k1_IBM_s006
                                 + 0.902117 w_k1_INTC_s006
                                 + 0.873111 w_k1_KO_s006
                                 + 0.975958 w_k1_MAT_s006
                                 + 0.831298 w_k1_MSFT_s006
                                 + 0.916649 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.33079 w_k1_ADM_s006
                                 + 1.50568 w_k1_AXP_s006
                                 + 1.68107 w_k1_DIS_s006
                                 + 1.37007 w_k1_FDX_s006
                                 + 1.61381 w_k1_IBM_s006
                                 + 0.98901 w_k1_INTC_s006
                                 + 1.34578 w_k1_KO_s006 + 1.40186 w_k1_MAT_s006
                                 + 1.29756 w_k1_MSFT_s006 + 1.2082 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.10651 w_k1_ADM_s006
                                 + 1.16266 w_k1_AXP_s006
                                 + 1.32624 w_k1_DIS_s006
                                 + 1.25862 w_k1_FDX_s006
                                 + 0.939663 w_k1_IBM_s006
                                 + 1.02301 w_k1_INTC_s006
                                 + 0.924315 w_k1_KO_s006
                                 + 0.808972 w_k1_MAT_s006
                                 + 1.17704 w_k1_MSFT_s006
                                 + 1.09113 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.787759 w_k1_ADM_s006
                                 + 0.759043 w_k1_AXP_s006
                                 + 0.881958 w_k1_DIS_s006
                                 + 0.947826 w_k1_FDX_s006
                                 + 0.651572 w_k1_IBM_s006
                                 + 0.81749 w_k1_INTC_s006
                                 + 0.740524 w_k1_KO_s006
                                 + 0.733999 w_k1_MAT_s006
                                 + 0.972397 w_k1_MSFT_s006
                                 + 0.791653 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.943034 w_k1_ADM_s006
                                 + 0.968383 w_k1_AXP_s006
                                 + 0.929766 w_k1_DIS_s006
                                 + 0.938404 w_k1_FDX_s006
                                 + 0.907002 w_k1_IBM_s006
                                 + 0.831175 w_k1_INTC_s006
                                 + 0.822049 w_k1_KO_s006
                                 + 0.60749 w_k1_MAT_s006
                                 + 0.909267 w_k1_MSFT_s006
                                 + 0.846682 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.07508 w_k1_ADM_s006
                                 + 1.20418 w_k1_AXP_s006
                                 + 1.23525 w_k1_DIS_s006
                                 + 1.04408 w_k1_FDX_s006
                                 + 1.57163 w_k1_IBM_s006
                                 + 0.941799 w_k1_INTC_s006
                                 + 1.26309 w_k1_KO_s006 + 1.70464 w_k1_MAT_s006
                                 + 1.1106 w_k1_MSFT_s006 + 1.02957 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.970289 w_k1_ADM_s006
                                 + 0.856484 w_k1_AXP_s006
                                 + 1.20755 w_k1_DIS_s006
                                 + 1.00904 w_k1_FDX_s006
                                 + 1.04027 w_k1_IBM_s006
                                 + 0.949743 w_k1_INTC_s006
                                 + 1.19059 w_k1_KO_s006
                                 + 0.854676 w_k1_MAT_s006
                                 + 1.12407 w_k1_MSFT_s006
                                 + 0.993797 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.12697 w_k1_ADM_s007
                                 + 1.24704 w_k1_AXP_s007
                                 + 1.24409 w_k1_DIS_s007
                                 + 1.05691 w_k1_FDX_s007
                                 + 1.35896 w_k1_IBM_s007
                                 + 1.11731 w_k1_INTC_s007
                                 + 1.32364 w_k1_KO_s007 + 1.42637 w_k1_MAT_s007
                                 + 1.16207 w_k1_MSFT_s007
                                 + 1.25174 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.936809 w_k1_ADM_s007
                                 + 0.836559 w_k1_AXP_s007
                                 + 0.809557 w_k1_DIS_s007
                                 + 0.949637 w_k1_FDX_s007
                                 + 0.552863 w_k1_IBM_s007
                                 + 1.0267 w_k1_INTC_s007
                                 + 0.869035 w_k1_KO_s007
                                 + 0.388326 w_k1_MAT_s007
                                 + 0.85235 w_k1_MSFT_s007
                                 + 0.880549 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.740978 w_k1_ADM_s007
                                 + 0.84367 w_k1_AXP_s007
                                 + 0.697286 w_k1_DIS_s007
                                 + 0.827649 w_k1_FDX_s007
                                 + 0.71134 w_k1_IBM_s007
                                 + 0.879954 w_k1_INTC_s007
                                 + 0.833912 w_k1_KO_s007
                                 + 0.782302 w_k1_MAT_s007
                                 + 0.809182 w_k1_MSFT_s007
                                 + 0.95135 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.21349 w_k1_ADM_s007
                                 + 1.25662 w_k1_AXP_s007
                                 + 1.09823 w_k1_DIS_s007
                                 + 1.17303 w_k1_FDX_s007
                                 + 0.872981 w_k1_IBM_s007
                                 + 1.09418 w_k1_INTC_s007
                                 + 0.998863 w_k1_KO_s007
                                 + 0.775084 w_k1_MAT_s007
                                 + 1.04265 w_k1_MSFT_s007
                                 + 1.04789 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.816983 w_k1_ADM_s007
                                 + 0.790909 w_k1_AXP_s007
                                 + 0.810758 w_k1_DIS_s007
                                 + 0.700053 w_k1_FDX_s007
                                 + 1.32529 w_k1_IBM_s007
                                 + 0.961262 w_k1_INTC_s007
                                 + 1.27389 w_k1_KO_s007 + 1.09234 w_k1_MAT_s007
                                 + 0.886123 w_k1_MSFT_s007
                                 + 0.989135 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.93343 w_k1_ADM_s007
                                 + 0.781862 w_k1_AXP_s007
                                 + 0.800632 w_k1_DIS_s007
                                 + 0.815304 w_k1_FDX_s007
                                 + 0.827606 w_k1_IBM_s007
                                 + 0.894502 w_k1_INTC_s007
                                 + 1.06032 w_k1_KO_s007
                                 + 0.655593 w_k1_MAT_s007
                                 + 0.85266 w_k1_MSFT_s007
                                 + 0.948457 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.916743 w_k1_ADM_s007
                                 + 1.12404 w_k1_AXP_s007
                                 + 0.86178 w_k1_DIS_s007
                                 + 0.923873 w_k1_FDX_s007
                                 + 1.15856 w_k1_IBM_s007
                                 + 0.961601 w_k1_INTC_s007
                                 + 1.06495 w_k1_KO_s007
                                 + 0.991201 w_k1_MAT_s007
                                 + 0.894255 w_k1_MSFT_s007
                                 + 1.07453 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.27523 w_k1_ADM_s007
                                 + 1.71862 w_k1_AXP_s007
                                 + 1.59005 w_k1_DIS_s007
                                 + 1.45571 w_k1_FDX_s007
                                 + 1.10468 w_k1_IBM_s007
                                 + 1.17767 w_k1_INTC_s007
                                 + 1.49534 w_k1_KO_s007 + 1.54017 w_k1_MAT_s007
                                 + 1.27868 w_k1_MSFT_s007
                                 + 1.40822 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.2487 w_k1_ADM_s010
                                 + 1.20917 w_k1_AXP_s010
                                 + 1.19882 w_k1_DIS_s010 + 1.1925 w_k1_FDX_s010
                                 + 0.841684 w_k1_IBM_s010
                                 + 1.36149 w_k1_INTC_s010
                                 + 0.956931 w_k1_KO_s010
                                 + 0.901054 w_k1_MAT_s010
                                 + 1.12472 w_k1_MSFT_s010
                                 + 1.04575 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.77759 w_k1_ADM_s010
                                 + 0.948176 w_k1_AXP_s010
                                 + 1.13086 w_k1_DIS_s010
                                 + 0.967344 w_k1_FDX_s010
                                 + 0.686476 w_k1_IBM_s010
                                 + 0.890839 w_k1_INTC_s010
                                 + 0.895192 w_k1_KO_s010
                                 + 0.652554 w_k1_MAT_s010
                                 + 0.785775 w_k1_MSFT_s010
                                 + 0.967701 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.996718 w_k1_ADM_s010
                                 + 1.11361 w_k1_AXP_s010
                                 + 0.801483 w_k1_DIS_s010
                                 + 0.87073 w_k1_FDX_s010
                                 + 0.630341 w_k1_IBM_s010
                                 + 0.77268 w_k1_INTC_s010
                                 + 0.758761 w_k1_KO_s010
                                 + 0.79447 w_k1_MAT_s010
                                 + 0.7227 w_k1_MSFT_s010 + 0.96009 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.651602 w_k1_ADM_s010
                                 + 0.609569 w_k1_AXP_s010
                                 + 0.749397 w_k1_DIS_s010
                                 + 0.678746 w_k1_FDX_s010
                                 + 0.718268 w_k1_IBM_s010
                                 + 0.720747 w_k1_INTC_s010
                                 + 0.751654 w_k1_KO_s010
                                 + 0.653553 w_k1_MAT_s010
                                 + 0.668495 w_k1_MSFT_s010
                                 + 0.710853 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.920928 w_k1_ADM_s010
                                 + 1.16218 w_k1_AXP_s010
                                 + 1.06289 w_k1_DIS_s010
                                 + 0.868925 w_k1_FDX_s010
                                 + 1.1123 w_k1_IBM_s010
                                 + 0.632285 w_k1_INTC_s010
                                 + 0.937627 w_k1_KO_s010
                                 + 1.27163 w_k1_MAT_s010
                                 + 0.630206 w_k1_MSFT_s010
                                 + 0.911774 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.09298 w_k1_ADM_s010
                                 + 1.12358 w_k1_AXP_s010 + 0.9224 w_k1_DIS_s010
                                 + 0.741747 w_k1_FDX_s010
                                 + 1.90672 w_k1_IBM_s010
                                 + 0.664283 w_k1_INTC_s010
                                 + 1.11378 w_k1_KO_s010 + 1.73642 w_k1_MAT_s010
                                 + 0.678367 w_k1_MSFT_s010
                                 + 1.11329 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.35634 w_k1_ADM_s010
                                 + 1.4278 w_k1_AXP_s010 + 1.66398 w_k1_DIS_s010
                                 + 1.23319 w_k1_FDX_s010
                                 + 1.31984 w_k1_IBM_s010
                                 + 1.12445 w_k1_INTC_s010
                                 + 1.43255 w_k1_KO_s010 + 1.28481 w_k1_MAT_s010
                                 + 1.10343 w_k1_MSFT_s010
                                 + 1.10407 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.919561 w_k1_ADM_s010
                                 + 0.77629 w_k1_AXP_s010
                                 + 0.677802 w_k1_DIS_s010
                                 + 0.736374 w_k1_FDX_s010
                                 + 0.938061 w_k1_IBM_s010
                                 + 0.714963 w_k1_INTC_s010
                                 + 0.754436 w_k1_KO_s010
                                 + 0.844539 w_k1_MAT_s010
                                 + 0.591874 w_k1_MSFT_s010
                                 + 0.748275 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.2249 w_k1_ADM_s011
                                 + 1.15599 w_k1_AXP_s011
                                 + 1.44151 w_k1_DIS_s011 + 1.3821 w_k1_FDX_s011
                                 + 0.940473 w_k1_IBM_s011
                                 + 1.09767 w_k1_INTC_s011
                                 + 1.26058 w_k1_KO_s011 + 1.05714 w_k1_MAT_s011
                                 + 1.3048 w_k1_MSFT_s011 + 1.2837 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.856659 w_k1_ADM_s011
                                 + 0.986068 w_k1_AXP_s011
                                 + 0.749769 w_k1_DIS_s011
                                 + 0.915313 w_k1_FDX_s011
                                 + 0.896676 w_k1_IBM_s011
                                 + 0.866042 w_k1_INTC_s011
                                 + 0.841684 w_k1_KO_s011
                                 + 0.897319 w_k1_MAT_s011
                                 + 0.890107 w_k1_MSFT_s011
                                 + 0.932018 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.947065 w_k1_ADM_s011
                                 + 0.931568 w_k1_AXP_s011
                                 + 1.02605 w_k1_DIS_s011
                                 + 0.947364 w_k1_FDX_s011
                                 + 1.10534 w_k1_IBM_s011
                                 + 0.794057 w_k1_INTC_s011
                                 + 0.953789 w_k1_KO_s011
                                 + 1.09926 w_k1_MAT_s011
                                 + 0.889864 w_k1_MSFT_s011
                                 + 0.941426 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.950157 w_k1_ADM_s011
                                 + 1.19898 w_k1_AXP_s011
                                 + 1.06003 w_k1_DIS_s011
                                 + 1.18342 w_k1_FDX_s011
                                 + 1.28244 w_k1_IBM_s011
                                 + 1.0886 w_k1_INTC_s011 + 1.10854 w_k1_KO_s011
                                 + 1.13353 w_k1_MAT_s011
                                 + 0.975466 w_k1_MSFT_s011
                                 + 0.989822 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.674674 w_k1_ADM_s011
                                 + 0.729457 w_k1_AXP_s011
                                 + 0.620085 w_k1_DIS_s011
                                 + 0.699792 w_k1_FDX_s011
                                 + 1.05815 w_k1_IBM_s011
                                 + 0.746991 w_k1_INTC_s011
                                 + 0.829119 w_k1_KO_s011
                                 + 1.09979 w_k1_MAT_s011
                                 + 0.7808 w_k1_MSFT_s011
                                 + 0.822971 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.795402 w_k1_ADM_s011
                                 + 0.643643 w_k1_AXP_s011
                                 + 0.770214 w_k1_DIS_s011
                                 + 0.890939 w_k1_FDX_s011
                                 + 0.565775 w_k1_IBM_s011
                                 + 0.803284 w_k1_INTC_s011
                                 + 0.71323 w_k1_KO_s011
                                 + 0.450065 w_k1_MAT_s011
                                 + 0.921156 w_k1_MSFT_s011
                                 + 0.792938 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.971035 w_k1_ADM_s011
                                 + 0.901516 w_k1_AXP_s011
                                 + 1.04111 w_k1_DIS_s011 + 1.1083 w_k1_FDX_s011
                                 + 0.705679 w_k1_IBM_s011
                                 + 0.876765 w_k1_INTC_s011
                                 + 0.899474 w_k1_KO_s011
                                 + 0.794479 w_k1_MAT_s011
                                 + 1.06674 w_k1_MSFT_s011
                                 + 1.03441 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.09387 w_k1_ADM_s011
                                 + 1.24093 w_k1_AXP_s011
                                 + 1.61802 w_k1_DIS_s011
                                 + 1.24225 w_k1_FDX_s011
                                 + 1.63624 w_k1_IBM_s011
                                 + 1.0291 w_k1_INTC_s011 + 1.44632 w_k1_KO_s011
                                 + 2.03951 w_k1_MAT_s011
                                 + 1.22712 w_k1_MSFT_s011 + 1.3689 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.11596 w_k1_ADM_s012
                                 + 1.23076 w_k1_AXP_s012 + 1.2917 w_k1_DIS_s012
                                 + 1.01747 w_k1_FDX_s012
                                 + 1.44587 w_k1_IBM_s012
                                 + 1.07406 w_k1_INTC_s012
                                 + 1.26796 w_k1_KO_s012 + 1.32032 w_k1_MAT_s012
                                 + 0.967863 w_k1_MSFT_s012
                                 + 1.04087 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.1223 w_k1_ADM_s012
                                 + 1.14846 w_k1_AXP_s012
                                 + 1.16305 w_k1_DIS_s012
                                 + 1.10981 w_k1_FDX_s012
                                 + 0.796376 w_k1_IBM_s012
                                 + 1.00018 w_k1_INTC_s012
                                 + 1.04629 w_k1_KO_s012
                                 + 0.842659 w_k1_MAT_s012
                                 + 0.905949 w_k1_MSFT_s012
                                 + 1.05394 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.33117 w_k1_ADM_s012
                                 + 1.36028 w_k1_AXP_s012
                                 + 1.57522 w_k1_DIS_s012
                                 + 1.20674 w_k1_FDX_s012 + 1.3913 w_k1_IBM_s012
                                 + 1.22068 w_k1_INTC_s012
                                 + 1.35888 w_k1_KO_s012 + 1.20902 w_k1_MAT_s012
                                 + 1.19944 w_k1_MSFT_s012
                                 + 1.21312 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.40622 w_k1_ADM_s012
                                 + 1.25152 w_k1_AXP_s012 + 1.6527 w_k1_DIS_s012
                                 + 1.45795 w_k1_FDX_s012
                                 + 0.850015 w_k1_IBM_s012
                                 + 1.33966 w_k1_INTC_s012 + 1.156 w_k1_KO_s012
                                 + 1.44156 w_k1_MAT_s012
                                 + 1.42475 w_k1_MSFT_s012
                                 + 1.18924 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.946384 w_k1_ADM_s012
                                 + 0.621386 w_k1_AXP_s012
                                 + 0.690386 w_k1_DIS_s012
                                 + 0.728179 w_k1_FDX_s012
                                 + 0.947606 w_k1_IBM_s012
                                 + 0.990774 w_k1_INTC_s012
                                 + 0.924961 w_k1_KO_s012
                                 + 0.856919 w_k1_MAT_s012
                                 + 0.790479 w_k1_MSFT_s012
                                 + 0.878625 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.821514 w_k1_ADM_s012
                                 + 0.885482 w_k1_AXP_s012
                                 + 1.03471 w_k1_DIS_s012
                                 + 0.910602 w_k1_FDX_s012
                                 + 1.07769 w_k1_IBM_s012
                                 + 1.12644 w_k1_INTC_s012
                                 + 1.11705 w_k1_KO_s012
                                 + 0.967174 w_k1_MAT_s012
                                 + 1.02198 w_k1_MSFT_s012
                                 + 0.935195 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.880482 w_k1_ADM_s012
                                 + 0.821126 w_k1_AXP_s012
                                 + 0.846484 w_k1_DIS_s012
                                 + 0.878208 w_k1_FDX_s012
                                 + 0.82459 w_k1_IBM_s012
                                 + 1.11547 w_k1_INTC_s012
                                 + 0.859288 w_k1_KO_s012
                                 + 0.677731 w_k1_MAT_s012
                                 + 0.87799 w_k1_MSFT_s012
                                 + 0.965183 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.10836 w_k1_ADM_s012
                                 + 0.741137 w_k1_AXP_s012
                                 + 0.811058 w_k1_DIS_s012
                                 + 0.953466 w_k1_FDX_s012
                                 + 0.542151 w_k1_IBM_s012
                                 + 1.0059 w_k1_INTC_s012
                                 + 0.787157 w_k1_KO_s012
                                 + 0.599568 w_k1_MAT_s012
                                 + 0.988536 w_k1_MSFT_s012
                                 + 0.941768 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.671119 w_k1_ADM_s013
                                 + 1.01697 w_k1_AXP_s013
                                 + 0.977728 w_k1_DIS_s013
                                 + 0.973366 w_k1_FDX_s013
                                 + 1.0812 w_k1_IBM_s013 + 1.235 w_k1_INTC_s013
                                 + 1.15676 w_k1_KO_s013 + 1.09085 w_k1_MAT_s013
                                 + 0.927868 w_k1_MSFT_s013
                                 + 1.06082 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.990897 w_k1_ADM_s013
                                 + 1.09576 w_k1_AXP_s013
                                 + 1.36676 w_k1_DIS_s013
                                 + 1.23588 w_k1_FDX_s013 + 1.138 w_k1_IBM_s013
                                 + 1.29979 w_k1_INTC_s013
                                 + 1.07755 w_k1_KO_s013 + 1.14139 w_k1_MAT_s013
                                 + 1.14798 w_k1_MSFT_s013
                                 + 1.17687 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.759535 w_k1_ADM_s013
                                 + 1.2979 w_k1_AXP_s013 + 1.03504 w_k1_DIS_s013
                                 + 1.18852 w_k1_FDX_s013
                                 + 0.658406 w_k1_IBM_s013
                                 + 1.37562 w_k1_INTC_s013
                                 + 0.842856 w_k1_KO_s013
                                 + 0.866117 w_k1_MAT_s013
                                 + 0.930497 w_k1_MSFT_s013
                                 + 1.10193 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.89399 w_k1_ADM_s013
                                 + 0.897887 w_k1_AXP_s013
                                 + 1.09307 w_k1_DIS_s013
                                 + 0.931413 w_k1_FDX_s013
                                 + 0.896238 w_k1_IBM_s013
                                 + 1.22998 w_k1_INTC_s013
                                 + 1.05799 w_k1_KO_s013
                                 + 0.825283 w_k1_MAT_s013
                                 + 0.953046 w_k1_MSFT_s013
                                 + 1.02585 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.97547 w_k1_ADM_s013
                                 + 1.3597 w_k1_AXP_s013 + 1.40856 w_k1_DIS_s013
                                 + 1.20039 w_k1_FDX_s013
                                 + 1.17167 w_k1_IBM_s013
                                 + 1.56924 w_k1_INTC_s013
                                 + 1.30705 w_k1_KO_s013 + 1.2118 w_k1_MAT_s013
                                 + 1.21132 w_k1_MSFT_s013
                                 + 1.39154 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.794585 w_k1_ADM_s013
                                 + 0.673983 w_k1_AXP_s013
                                 + 0.660734 w_k1_DIS_s013
                                 + 0.775114 w_k1_FDX_s013
                                 + 0.715721 w_k1_IBM_s013
                                 + 0.824726 w_k1_INTC_s013
                                 + 0.704673 w_k1_KO_s013
                                 + 0.531847 w_k1_MAT_s013
                                 + 0.704912 w_k1_MSFT_s013
                                 + 0.875953 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.941613 w_k1_ADM_s013
                                 + 0.861012 w_k1_AXP_s013
                                 + 0.78055 w_k1_DIS_s013
                                 + 0.99519 w_k1_FDX_s013
                                 + 0.663341 w_k1_IBM_s013
                                 + 1.06529 w_k1_INTC_s013
                                 + 0.850895 w_k1_KO_s013
                                 + 0.640591 w_k1_MAT_s013
                                 + 0.84645 w_k1_MSFT_s013
                                 + 0.970984 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.856397 w_k1_ADM_s013
                                 + 1.03116 w_k1_AXP_s013
                                 + 0.967207 w_k1_DIS_s013
                                 + 0.874674 w_k1_FDX_s013
                                 + 1.41861 w_k1_IBM_s013
                                 + 1.09224 w_k1_INTC_s013
                                 + 1.17925 w_k1_KO_s013 + 1.09395 w_k1_MAT_s013
                                 + 0.831867 w_k1_MSFT_s013
                                 + 0.999034 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.991367 w_k1_ADM_s014
                                 + 1.13839 w_k1_AXP_s014
                                 + 1.14729 w_k1_DIS_s014
                                 + 1.13701 w_k1_FDX_s014
                                 + 1.09757 w_k1_IBM_s014
                                 + 1.38758 w_k1_INTC_s014
                                 + 1.14963 w_k1_KO_s014 + 1.34948 w_k1_MAT_s014
                                 + 1.33067 w_k1_MSFT_s014
                                 + 1.13185 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.831659 w_k1_ADM_s014
                                 + 0.726973 w_k1_AXP_s014
                                 + 0.800557 w_k1_DIS_s014
                                 + 0.93127 w_k1_FDX_s014
                                 + 0.565874 w_k1_IBM_s014
                                 + 1.03401 w_k1_INTC_s014
                                 + 0.847049 w_k1_KO_s014
                                 + 0.571382 w_k1_MAT_s014
                                 + 1.03772 w_k1_MSFT_s014
                                 + 0.978917 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.832321 w_k1_ADM_s014
                                 + 1.14754 w_k1_AXP_s014
                                 + 0.875359 w_k1_DIS_s014
                                 + 0.975604 w_k1_FDX_s014
                                 + 0.865575 w_k1_IBM_s014
                                 + 0.996504 w_k1_INTC_s014
                                 + 0.859479 w_k1_KO_s014
                                 + 1.06251 w_k1_MAT_s014
                                 + 0.921187 w_k1_MSFT_s014
                                 + 1.11443 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.66303 w_k1_ADM_s014
                                 + 0.611356 w_k1_AXP_s014
                                 + 0.56431 w_k1_DIS_s014
                                 + 0.684009 w_k1_FDX_s014
                                 + 0.861154 w_k1_IBM_s014
                                 + 1.06131 w_k1_INTC_s014
                                 + 0.804315 w_k1_KO_s014
                                 + 0.66299 w_k1_MAT_s014
                                 + 0.793919 w_k1_MSFT_s014
                                 + 0.917777 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.1928 w_k1_ADM_s014
                                 + 1.28461 w_k1_AXP_s014
                                 + 1.36599 w_k1_DIS_s014
                                 + 1.36022 w_k1_FDX_s014
                                 + 0.901927 w_k1_IBM_s014
                                 + 1.28213 w_k1_INTC_s014
                                 + 1.24852 w_k1_KO_s014 + 1.07338 w_k1_MAT_s014
                                 + 1.27489 w_k1_MSFT_s014
                                 + 1.25488 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.00134 w_k1_ADM_s014
                                 + 0.952314 w_k1_AXP_s014
                                 + 1.38385 w_k1_DIS_s014
                                 + 1.04419 w_k1_FDX_s014
                                 + 1.27441 w_k1_IBM_s014
                                 + 1.11806 w_k1_INTC_s014
                                 + 1.45413 w_k1_KO_s014 + 1.35951 w_k1_MAT_s014
                                 + 1.37626 w_k1_MSFT_s014
                                 + 1.42343 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.992981 w_k1_ADM_s014
                                 + 1.14824 w_k1_AXP_s014
                                 + 0.910851 w_k1_DIS_s014
                                 + 0.975113 w_k1_FDX_s014
                                 + 1.36272 w_k1_IBM_s014
                                 + 1.20096 w_k1_INTC_s014
                                 + 1.17752 w_k1_KO_s014 + 1.52738 w_k1_MAT_s014
                                 + 1.08889 w_k1_MSFT_s014
                                 + 1.11568 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.685882 w_k1_ADM_s014
                                 + 0.798278 w_k1_AXP_s014 + 0.736 w_k1_DIS_s014
                                 + 0.846086 w_k1_FDX_s014
                                 + 0.783902 w_k1_IBM_s014
                                 + 0.961365 w_k1_INTC_s014
                                 + 0.843172 w_k1_KO_s014
                                 + 0.852694 w_k1_MAT_s014
                                 + 0.883347 w_k1_MSFT_s014
                                 + 1.03647 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.0607 w_k1_ADM_s015
                                 + 0.64733 w_k1_AXP_s015
                                 + 0.532295 w_k1_DIS_s015
                                 + 0.610906 w_k1_FDX_s015
                                 + 0.690319 w_k1_IBM_s015
                                 + 0.526566 w_k1_INTC_s015
                                 + 0.686887 w_k1_KO_s015
                                 + 0.632937 w_k1_MAT_s015
                                 + 0.585247 w_k1_MSFT_s015
                                 + 0.896068 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.07892 w_k1_ADM_s015
                                 + 0.992088 w_k1_AXP_s015
                                 + 0.846104 w_k1_DIS_s015
                                 + 0.888656 w_k1_FDX_s015
                                 + 1.05542 w_k1_IBM_s015
                                 + 0.85967 w_k1_INTC_s015
                                 + 0.834966 w_k1_KO_s015
                                 + 1.09023 w_k1_MAT_s015
                                 + 0.886379 w_k1_MSFT_s015
                                 + 0.914746 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.32316 w_k1_ADM_s015
                                 + 0.764601 w_k1_AXP_s015
                                 + 0.840156 w_k1_DIS_s015
                                 + 0.888333 w_k1_FDX_s015
                                 + 0.756081 w_k1_IBM_s015
                                 + 0.792025 w_k1_INTC_s015
                                 + 0.781438 w_k1_KO_s015
                                 + 0.714772 w_k1_MAT_s015
                                 + 0.797633 w_k1_MSFT_s015
                                 + 0.868876 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.79392 w_k1_ADM_s015
                                 + 1.26667 w_k1_AXP_s015
                                 + 1.60813 w_k1_DIS_s015
                                 + 1.16503 w_k1_FDX_s015
                                 + 1.57619 w_k1_IBM_s015
                                 + 0.936917 w_k1_INTC_s015
                                 + 1.52917 w_k1_KO_s015 + 1.44335 w_k1_MAT_s015
                                 + 1.20394 w_k1_MSFT_s015
                                 + 1.34826 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.766469 w_k1_ADM_s015
                                 + 0.871301 w_k1_AXP_s015
                                 + 0.780274 w_k1_DIS_s015
                                 + 0.731099 w_k1_FDX_s015
                                 + 1.43436 w_k1_IBM_s015
                                 + 1.10977 w_k1_INTC_s015
                                 + 1.09256 w_k1_KO_s015 + 1.21104 w_k1_MAT_s015
                                 + 0.842054 w_k1_MSFT_s015
                                 + 0.991283 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.6443 w_k1_ADM_s015
                                 + 1.04197 w_k1_AXP_s015
                                 + 1.21412 w_k1_DIS_s015
                                 + 0.980261 w_k1_FDX_s015
                                 + 1.38067 w_k1_IBM_s015
                                 + 0.802384 w_k1_INTC_s015
                                 + 1.1931 w_k1_KO_s015 + 1.39442 w_k1_MAT_s015
                                 + 1.05371 w_k1_MSFT_s015
                                 + 0.977195 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.11236 w_k1_ADM_s015
                                 + 1.00999 w_k1_AXP_s015 + 1.4646 w_k1_DIS_s015
                                 + 1.09555 w_k1_FDX_s015
                                 + 0.848763 w_k1_IBM_s015
                                 + 1.19812 w_k1_INTC_s015
                                 + 1.09468 w_k1_KO_s015
                                 + 0.688667 w_k1_MAT_s015
                                 + 1.20417 w_k1_MSFT_s015 + 1.0021 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.52133 w_k1_ADM_s015
                                 + 1.35475 w_k1_AXP_s015
                                 + 1.28961 w_k1_DIS_s015
                                 + 1.28622 w_k1_FDX_s015
                                 + 0.978662 w_k1_IBM_s015
                                 + 1.20097 w_k1_INTC_s015
                                 + 1.13747 w_k1_KO_s015 + 1.17736 w_k1_MAT_s015
                                 + 1.17252 w_k1_MSFT_s015
                                 + 1.19248 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.52364 w_k1_ADM_s016
                                 + 1.48988 w_k1_AXP_s016
                                 + 1.73168 w_k1_DIS_s016
                                 + 1.51705 w_k1_FDX_s016
                                 + 1.07874 w_k1_IBM_s016
                                 + 1.29926 w_k1_INTC_s016
                                 + 1.26686 w_k1_KO_s016 + 1.34754 w_k1_MAT_s016
                                 + 1.21425 w_k1_MSFT_s016
                                 + 1.22177 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.922829 w_k1_ADM_s016
                                 + 1.43151 w_k1_AXP_s016
                                 + 1.16249 w_k1_DIS_s016
                                 + 1.10754 w_k1_FDX_s016
                                 + 1.06423 w_k1_IBM_s016
                                 + 1.18977 w_k1_INTC_s016
                                 + 1.02121 w_k1_KO_s016 + 1.67024 w_k1_MAT_s016
                                 + 1.04278 w_k1_MSFT_s016
                                 + 1.01483 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.73688 w_k1_ADM_s016
                                 + 0.929043 w_k1_AXP_s016
                                 + 0.857103 w_k1_DIS_s016
                                 + 0.910804 w_k1_FDX_s016
                                 + 0.804984 w_k1_IBM_s016
                                 + 1.06424 w_k1_INTC_s016
                                 + 0.815584 w_k1_KO_s016
                                 + 0.965374 w_k1_MAT_s016
                                 + 0.900777 w_k1_MSFT_s016
                                 + 0.905225 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.840945 w_k1_ADM_s016
                                 + 1.15704 w_k1_AXP_s016
                                 + 1.11158 w_k1_DIS_s016
                                 + 0.976338 w_k1_FDX_s016
                                 + 1.48143 w_k1_IBM_s016
                                 + 1.17087 w_k1_INTC_s016
                                 + 1.19386 w_k1_KO_s016 + 1.99819 w_k1_MAT_s016
                                 + 0.982441 w_k1_MSFT_s016
                                 + 1.05237 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.968069 w_k1_ADM_s016
                                 + 1.18168 w_k1_AXP_s016
                                 + 1.31508 w_k1_DIS_s016
                                 + 1.10151 w_k1_FDX_s016
                                 + 0.861454 w_k1_IBM_s016
                                 + 1.26454 w_k1_INTC_s016
                                 + 1.20789 w_k1_KO_s016 + 1.64346 w_k1_MAT_s016
                                 + 1.12914 w_k1_MSFT_s016
                                 + 1.14976 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.728037 w_k1_ADM_s016
                                 + 0.722724 w_k1_AXP_s016
                                 + 0.751413 w_k1_DIS_s016
                                 + 0.761731 w_k1_FDX_s016
                                 + 1.01261 w_k1_IBM_s016
                                 + 0.943123 w_k1_INTC_s016
                                 + 0.897745 w_k1_KO_s016
                                 + 0.725568 w_k1_MAT_s016
                                 + 0.733646 w_k1_MSFT_s016
                                 + 0.888383 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.19037 w_k1_ADM_s016
                                 + 1.30234 w_k1_AXP_s016
                                 + 1.01036 w_k1_DIS_s016
                                 + 1.24529 w_k1_FDX_s016
                                 + 0.728548 w_k1_IBM_s016
                                 + 1.22036 w_k1_INTC_s016
                                 + 0.788492 w_k1_KO_s016
                                 + 0.714975 w_k1_MAT_s016
                                 + 0.991034 w_k1_MSFT_s016
                                 + 1.02467 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.93344 w_k1_ADM_s016
                                 + 1.17019 w_k1_AXP_s016
                                 + 1.44791 w_k1_DIS_s016 + 1.0562 w_k1_FDX_s016
                                 + 1.4674 w_k1_IBM_s016
                                 + 1.16066 w_k1_INTC_s016
                                 + 1.48248 w_k1_KO_s016 + 1.38477 w_k1_MAT_s016
                                 + 1.1926 w_k1_MSFT_s016 + 1.2774 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.811439 w_k1_ADM_s017
                                 + 0.626591 w_k1_AXP_s017
                                 + 0.686944 w_k1_DIS_s017
                                 + 0.665652 w_k1_FDX_s017
                                 + 0.932018 w_k1_IBM_s017
                                 + 0.824879 w_k1_INTC_s017
                                 + 0.780611 w_k1_KO_s017
                                 + 0.58027 w_k1_MAT_s017
                                 + 0.699549 w_k1_MSFT_s017
                                 + 0.774917 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.14226 w_k1_ADM_s017
                                 + 1.72671 w_k1_AXP_s017
                                 + 1.85879 w_k1_DIS_s017
                                 + 1.39259 w_k1_FDX_s017
                                 + 1.20948 w_k1_IBM_s017
                                 + 1.5314 w_k1_INTC_s017 + 1.52943 w_k1_KO_s017
                                 + 1.59418 w_k1_MAT_s017
                                 + 1.39608 w_k1_MSFT_s017
                                 + 1.31844 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.51098 w_k1_ADM_s017
                                 + 1.55929 w_k1_AXP_s017
                                 + 1.53799 w_k1_DIS_s017
                                 + 1.27754 w_k1_FDX_s017
                                 + 1.44646 w_k1_IBM_s017
                                 + 1.19063 w_k1_INTC_s017
                                 + 1.43683 w_k1_KO_s017 + 1.98893 w_k1_MAT_s017
                                 + 1.12683 w_k1_MSFT_s017
                                 + 1.20248 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.872882 w_k1_ADM_s017
                                 + 0.758064 w_k1_AXP_s017
                                 + 0.775399 w_k1_DIS_s017
                                 + 0.840284 w_k1_FDX_s017
                                 + 0.823709 w_k1_IBM_s017
                                 + 1.01416 w_k1_INTC_s017
                                 + 0.811278 w_k1_KO_s017
                                 + 0.621395 w_k1_MAT_s017
                                 + 0.948126 w_k1_MSFT_s017
                                 + 0.855531 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.926123 w_k1_ADM_s017
                                 + 1.18796 w_k1_AXP_s017
                                 + 1.15488 w_k1_DIS_s017 + 1.1298 w_k1_FDX_s017
                                 + 0.975168 w_k1_IBM_s017
                                 + 1.291 w_k1_INTC_s017 + 1.13354 w_k1_KO_s017
                                 + 1.1703 w_k1_MAT_s017
                                 + 1.32232 w_k1_MSFT_s017
                                 + 1.06864 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.953661 w_k1_ADM_s017
                                 + 0.93171 w_k1_AXP_s017
                                 + 0.825481 w_k1_DIS_s017
                                 + 0.936814 w_k1_FDX_s017
                                 + 0.681864 w_k1_IBM_s017
                                 + 1.09007 w_k1_INTC_s017
                                 + 0.830878 w_k1_KO_s017
                                 + 0.820659 w_k1_MAT_s017
                                 + 1.02931 w_k1_MSFT_s017
                                 + 0.996209 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.2223 w_k1_ADM_s017
                                 + 1.07054 w_k1_AXP_s017
                                 + 0.985102 w_k1_DIS_s017
                                 + 0.855044 w_k1_FDX_s017
                                 + 1.35165 w_k1_IBM_s017
                                 + 0.984297 w_k1_INTC_s017
                                 + 1.14812 w_k1_KO_s017 + 1.10079 w_k1_MAT_s017
                                 + 0.947998 w_k1_MSFT_s017
                                 + 1.08325 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.742415 w_k1_ADM_s017
                                 + 0.70704 w_k1_AXP_s017
                                 + 1.02745 w_k1_DIS_s017
                                 + 0.776543 w_k1_FDX_s017
                                 + 1.16936 w_k1_IBM_s017
                                 + 0.917291 w_k1_INTC_s017
                                 + 1.11111 w_k1_KO_s017
                                 + 0.933613 w_k1_MAT_s017
                                 + 0.905642 w_k1_MSFT_s017
                                 + 0.892611 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.84763 w_k1_ADM_s100
                                 + 0.767322 w_k1_AXP_s100
                                 + 0.974025 w_k1_DIS_s100
                                 + 0.895228 w_k1_FDX_s100
                                 + 1.01704 w_k1_IBM_s100
                                 + 0.977411 w_k1_INTC_s100
                                 + 1.0622 w_k1_KO_s100 + 0.901186 w_k1_MAT_s100
                                 + 1.00375 w_k1_MSFT_s100
                                 + 1.00852 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.8878 w_k1_ADM_s100
                                 + 1.31688 w_k1_AXP_s100
                                 + 1.33039 w_k1_DIS_s100
                                 + 1.29704 w_k1_FDX_s100
                                 + 0.89039 w_k1_IBM_s100
                                 + 1.32436 w_k1_INTC_s100
                                 + 0.956003 w_k1_KO_s100
                                 + 0.951078 w_k1_MAT_s100
                                 + 1.1718 w_k1_MSFT_s100 + 1.09318 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.09795 w_k1_ADM_s100
                                 + 1.2149 w_k1_AXP_s100
                                 + 0.932096 w_k1_DIS_s100
                                 + 0.909849 w_k1_FDX_s100
                                 + 1.31679 w_k1_IBM_s100
                                 + 1.10627 w_k1_INTC_s100 + 1.0933 w_k1_KO_s100
                                 + 1.76786 w_k1_MAT_s100
                                 + 1.01991 w_k1_MSFT_s100
                                 + 0.978116 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.992677 w_k1_ADM_s100
                                 + 1.77176 w_k1_AXP_s100 + 1.7799 w_k1_DIS_s100
                                 + 1.36469 w_k1_FDX_s100
                                 + 1.57333 w_k1_IBM_s100
                                 + 1.54324 w_k1_INTC_s100
                                 + 1.36812 w_k1_KO_s100 + 1.76625 w_k1_MAT_s100
                                 + 1.25379 w_k1_MSFT_s100
                                 + 1.25208 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.745738 w_k1_ADM_s100
                                 + 0.941967 w_k1_AXP_s100
                                 + 0.796843 w_k1_DIS_s100
                                 + 0.930982 w_k1_FDX_s100
                                 + 0.703037 w_k1_IBM_s100
                                 + 1.04497 w_k1_INTC_s100
                                 + 0.752357 w_k1_KO_s100
                                 + 0.713659 w_k1_MAT_s100
                                 + 0.836698 w_k1_MSFT_s100
                                 + 0.906748 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.85109 w_k1_ADM_s100
                                 + 0.710638 w_k1_AXP_s100
                                 + 0.530053 w_k1_DIS_s100
                                 + 0.662971 w_k1_FDX_s100
                                 + 0.715225 w_k1_IBM_s100
                                 + 0.805949 w_k1_INTC_s100
                                 + 0.616922 w_k1_KO_s100
                                 + 0.704437 w_k1_MAT_s100
                                 + 0.694876 w_k1_MSFT_s100
                                 + 0.832714 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.0897 w_k1_ADM_s100
                                 + 1.45612 w_k1_AXP_s100
                                 + 1.17218 w_k1_DIS_s100
                                 + 1.12329 w_k1_FDX_s100 + 1.1807 w_k1_IBM_s100
                                 + 1.22668 w_k1_INTC_s100
                                 + 1.02509 w_k1_KO_s100 + 1.12742 w_k1_MAT_s100
                                 + 1.20598 w_k1_MSFT_s100
                                 + 1.10625 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.760308 w_k1_ADM_s100
                                 + 0.925335 w_k1_AXP_s100
                                 + 0.725029 w_k1_DIS_s100
                                 + 0.778025 w_k1_FDX_s100
                                 + 1.10718 w_k1_IBM_s100
                                 + 0.77469 w_k1_INTC_s100
                                 + 0.918096 w_k1_KO_s100
                                 + 1.28221 w_k1_MAT_s100
                                 + 0.761943 w_k1_MSFT_s100
                                 + 0.918526 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.09084 w_k1_ADM_s101
                                 + 0.812519 w_k1_AXP_s101
                                 + 0.905963 w_k1_DIS_s101
                                 + 0.846808 w_k1_FDX_s101
                                 + 1.11458 w_k1_IBM_s101
                                 + 0.895538 w_k1_INTC_s101
                                 + 0.944917 w_k1_KO_s101
                                 + 0.797589 w_k1_MAT_s101
                                 + 0.887708 w_k1_MSFT_s101
                                 + 0.881551 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.05985 w_k1_ADM_s101
                                 + 0.715893 w_k1_AXP_s101
                                 + 0.732496 w_k1_DIS_s101
                                 + 0.982208 w_k1_FDX_s101
                                 + 0.706392 w_k1_IBM_s101
                                 + 1.0036 w_k1_INTC_s101
                                 + 0.706161 w_k1_KO_s101
                                 + 0.501882 w_k1_MAT_s101
                                 + 0.948216 w_k1_MSFT_s101
                                 + 0.800648 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.43836 w_k1_ADM_s101
                                 + 0.68177 w_k1_AXP_s101
                                 + 0.786942 w_k1_DIS_s101
                                 + 0.841399 w_k1_FDX_s101
                                 + 0.851196 w_k1_IBM_s101
                                 + 0.924207 w_k1_INTC_s101
                                 + 0.771805 w_k1_KO_s101
                                 + 0.756664 w_k1_MAT_s101
                                 + 0.929446 w_k1_MSFT_s101
                                 + 0.88285 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.30026 w_k1_ADM_s101
                                 + 1.33359 w_k1_AXP_s101
                                 + 1.18111 w_k1_DIS_s101
                                 + 0.939086 w_k1_FDX_s101
                                 + 1.55515 w_k1_IBM_s101
                                 + 0.994438 w_k1_INTC_s101
                                 + 1.08448 w_k1_KO_s101 + 1.3663 w_k1_MAT_s101
                                 + 0.961167 w_k1_MSFT_s101
                                 + 0.925415 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.31308 w_k1_ADM_s101
                                 + 0.629591 w_k1_AXP_s101
                                 + 0.880182 w_k1_DIS_s101
                                 + 1.01332 w_k1_FDX_s101
                                 + 0.535605 w_k1_IBM_s101
                                 + 0.871358 w_k1_INTC_s101
                                 + 0.702192 w_k1_KO_s101
                                 + 0.439305 w_k1_MAT_s101
                                 + 0.972586 w_k1_MSFT_s101
                                 + 0.768452 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.49572 w_k1_ADM_s101
                                 + 0.936777 w_k1_AXP_s101
                                 + 0.966259 w_k1_DIS_s101
                                 + 0.897328 w_k1_FDX_s101
                                 + 0.744438 w_k1_IBM_s101
                                 + 0.85749 w_k1_INTC_s101
                                 + 0.76753 w_k1_KO_s101
                                 + 0.703593 w_k1_MAT_s101
                                 + 0.971987 w_k1_MSFT_s101
                                 + 0.902556 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.31737 w_k1_ADM_s101
                                 + 1.3626 w_k1_AXP_s101 + 1.41792 w_k1_DIS_s101
                                 + 1.2191 w_k1_FDX_s101 + 1.09784 w_k1_IBM_s101
                                 + 1.20998 w_k1_INTC_s101 + 1.1631 w_k1_KO_s101
                                 + 1.3867 w_k1_MAT_s101
                                 + 1.33418 w_k1_MSFT_s101
                                 + 1.03755 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.13613 w_k1_ADM_s101
                                 + 0.887779 w_k1_AXP_s101
                                 + 1.41959 w_k1_DIS_s101
                                 + 1.14044 w_k1_FDX_s101
                                 + 0.923632 w_k1_IBM_s101
                                 + 1.19964 w_k1_INTC_s101
                                 + 0.90262 w_k1_KO_s101 + 0.65038 w_k1_MAT_s101
                                 + 1.17629 w_k1_MSFT_s101
                                 + 0.98162 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.926611 w_k1_ADM_s102
                                 + 1.1238 w_k1_AXP_s102
                                 + 0.908754 w_k1_DIS_s102
                                 + 0.98154 w_k1_FDX_s102
                                 + 1.00447 w_k1_IBM_s102
                                 + 0.92204 w_k1_INTC_s102
                                 + 0.912483 w_k1_KO_s102
                                 + 0.858548 w_k1_MAT_s102
                                 + 0.871773 w_k1_MSFT_s102
                                 + 0.894173 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.09338 w_k1_ADM_s102
                                 + 1.46765 w_k1_AXP_s102
                                 + 1.41681 w_k1_DIS_s102
                                 + 1.25561 w_k1_FDX_s102
                                 + 1.30888 w_k1_IBM_s102
                                 + 1.45302 w_k1_INTC_s102
                                 + 1.39953 w_k1_KO_s102 + 1.12693 w_k1_MAT_s102
                                 + 1.25975 w_k1_MSFT_s102
                                 + 1.09234 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.835326 w_k1_ADM_s102
                                 + 1.02975 w_k1_AXP_s102
                                 + 0.734477 w_k1_DIS_s102
                                 + 0.857152 w_k1_FDX_s102
                                 + 1.38627 w_k1_IBM_s102
                                 + 1.02147 w_k1_INTC_s102
                                 + 0.851393 w_k1_KO_s102 + 1.4662 w_k1_MAT_s102
                                 + 0.803491 w_k1_MSFT_s102
                                 + 0.912023 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.743875 w_k1_ADM_s102
                                 + 0.857942 w_k1_AXP_s102
                                 + 0.699623 w_k1_DIS_s102
                                 + 0.722104 w_k1_FDX_s102
                                 + 0.973539 w_k1_IBM_s102
                                 + 0.975301 w_k1_INTC_s102
                                 + 0.957791 w_k1_KO_s102
                                 + 1.10145 w_k1_MAT_s102
                                 + 0.83991 w_k1_MSFT_s102
                                 + 0.870702 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.03568 w_k1_ADM_s102
                                 + 1.25368 w_k1_AXP_s102
                                 + 1.10255 w_k1_DIS_s102
                                 + 0.995522 w_k1_FDX_s102
                                 + 1.61257 w_k1_IBM_s102
                                 + 1.14582 w_k1_INTC_s102
                                 + 1.58211 w_k1_KO_s102 + 1.61914 w_k1_MAT_s102
                                 + 1.10334 w_k1_MSFT_s102
                                 + 1.17641 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.839286 w_k1_ADM_s102
                                 + 0.897459 w_k1_AXP_s102
                                 + 1.20353 w_k1_DIS_s102
                                 + 1.00226 w_k1_FDX_s102
                                 + 1.05651 w_k1_IBM_s102
                                 + 1.10291 w_k1_INTC_s102
                                 + 1.15617 w_k1_KO_s102
                                 + 0.960212 w_k1_MAT_s102
                                 + 1.13793 w_k1_MSFT_s102
                                 + 0.945347 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.91088 w_k1_ADM_s102
                                 + 0.748335 w_k1_AXP_s102
                                 + 0.851909 w_k1_DIS_s102
                                 + 0.945967 w_k1_FDX_s102
                                 + 0.692369 w_k1_IBM_s102
                                 + 1.18745 w_k1_INTC_s102
                                 + 0.830971 w_k1_KO_s102
                                 + 0.690722 w_k1_MAT_s102
                                 + 1.04341 w_k1_MSFT_s102
                                 + 0.904915 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.12335 w_k1_ADM_s102
                                 + 1.21028 w_k1_AXP_s102 + 0.9744 w_k1_DIS_s102
                                 + 1.14152 w_k1_FDX_s102
                                 + 0.83257 w_k1_IBM_s102
                                 + 1.14448 w_k1_INTC_s102
                                 + 0.886845 w_k1_KO_s102
                                 + 0.618023 w_k1_MAT_s102
                                 + 1.02357 w_k1_MSFT_s102
                                 + 0.858569 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.948976 w_k1_ADM_s103
                                 + 0.83419 w_k1_AXP_s103
                                 + 0.979116 w_k1_DIS_s103
                                 + 0.915301 w_k1_FDX_s103
                                 + 0.785681 w_k1_IBM_s103
                                 + 0.830261 w_k1_INTC_s103
                                 + 0.845277 w_k1_KO_s103
                                 + 0.749912 w_k1_MAT_s103
                                 + 0.846571 w_k1_MSFT_s103
                                 + 0.881333 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.33297 w_k1_ADM_s103
                                 + 1.31275 w_k1_AXP_s103
                                 + 1.42241 w_k1_DIS_s103
                                 + 1.27828 w_k1_FDX_s103
                                 + 1.01718 w_k1_IBM_s103
                                 + 1.39414 w_k1_INTC_s103
                                 + 1.10325 w_k1_KO_s103
                                 + 0.795016 w_k1_MAT_s103
                                 + 1.11827 w_k1_MSFT_s103
                                 + 1.09331 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.996982 w_k1_ADM_s103
                                 + 1.18175 w_k1_AXP_s103
                                 + 1.27532 w_k1_DIS_s103
                                 + 1.15583 w_k1_FDX_s103
                                 + 1.65752 w_k1_IBM_s103
                                 + 1.24979 w_k1_INTC_s103
                                 + 1.30464 w_k1_KO_s103 + 1.40328 w_k1_MAT_s103
                                 + 1.09504 w_k1_MSFT_s103
                                 + 1.19933 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.13071 w_k1_ADM_s103
                                 + 0.815858 w_k1_AXP_s103 + 1.068 w_k1_DIS_s103
                                 + 1.07738 w_k1_FDX_s103
                                 + 0.666029 w_k1_IBM_s103
                                 + 0.974709 w_k1_INTC_s103
                                 + 0.829867 w_k1_KO_s103
                                 + 0.625578 w_k1_MAT_s103
                                 + 1.02733 w_k1_MSFT_s103
                                 + 0.948674 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.989449 w_k1_ADM_s103
                                 + 0.894281 w_k1_AXP_s103
                                 + 0.768635 w_k1_DIS_s103
                                 + 0.832182 w_k1_FDX_s103
                                 + 0.899775 w_k1_IBM_s103
                                 + 0.796397 w_k1_INTC_s103
                                 + 0.906615 w_k1_KO_s103
                                 + 0.90151 w_k1_MAT_s103
                                 + 0.746823 w_k1_MSFT_s103
                                 + 0.941121 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.03512 w_k1_ADM_s103
                                 + 0.93495 w_k1_AXP_s103
                                 + 0.832638 w_k1_DIS_s103
                                 + 0.848152 w_k1_FDX_s103
                                 + 1.21797 w_k1_IBM_s103
                                 + 0.836423 w_k1_INTC_s103
                                 + 0.932727 w_k1_KO_s103 + 1.3001 w_k1_MAT_s103
                                 + 0.886153 w_k1_MSFT_s103
                                 + 0.945932 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.990662 w_k1_ADM_s103
                                 + 0.966118 w_k1_AXP_s103
                                 + 1.30503 w_k1_DIS_s103
                                 + 1.11528 w_k1_FDX_s103
                                 + 1.07668 w_k1_IBM_s103
                                 + 1.04123 w_k1_INTC_s103
                                 + 1.12812 w_k1_KO_s103
                                 + 0.937684 w_k1_MAT_s103
                                 + 1.03562 w_k1_MSFT_s103
                                 + 1.10577 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.03174 w_k1_ADM_s103
                                 + 1.31322 w_k1_AXP_s103
                                 + 1.22471 w_k1_DIS_s103
                                 + 1.15638 w_k1_FDX_s103
                                 + 1.23492 w_k1_IBM_s103
                                 + 1.11517 w_k1_INTC_s103
                                 + 1.09485 w_k1_KO_s103 + 1.45759 w_k1_MAT_s103
                                 + 1.0461 w_k1_MSFT_s103 + 1.10257 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.819465 w_k1_ADM_s104
                                 + 0.763293 w_k1_AXP_s104
                                 + 0.648451 w_k1_DIS_s104
                                 + 0.685337 w_k1_FDX_s104
                                 + 0.982827 w_k1_IBM_s104
                                 + 0.70183 w_k1_INTC_s104
                                 + 0.785156 w_k1_KO_s104
                                 + 0.768618 w_k1_MAT_s104
                                 + 0.714746 w_k1_MSFT_s104
                                 + 0.913082 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.0525 w_k1_ADM_s104
                                 + 1.07278 w_k1_AXP_s104
                                 + 0.824729 w_k1_DIS_s104
                                 + 0.966852 w_k1_FDX_s104
                                 + 1.12829 w_k1_IBM_s104
                                 + 1.12101 w_k1_INTC_s104
                                 + 0.981541 w_k1_KO_s104
                                 + 1.06245 w_k1_MAT_s104
                                 + 1.05009 w_k1_MSFT_s104
                                 + 1.08576 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.923266 w_k1_ADM_s104
                                 + 0.856724 w_k1_AXP_s104
                                 + 0.724555 w_k1_DIS_s104
                                 + 0.815221 w_k1_FDX_s104
                                 + 0.715575 w_k1_IBM_s104
                                 + 0.834013 w_k1_INTC_s104
                                 + 0.687983 w_k1_KO_s104
                                 + 0.796087 w_k1_MAT_s104
                                 + 0.928182 w_k1_MSFT_s104
                                 + 0.894219 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.21692 w_k1_ADM_s104
                                 + 1.18578 w_k1_AXP_s104
                                 + 1.22378 w_k1_DIS_s104
                                 + 1.05467 w_k1_FDX_s104
                                 + 1.32266 w_k1_IBM_s104
                                 + 1.24293 w_k1_INTC_s104
                                 + 1.11795 w_k1_KO_s104
                                 + 0.945598 w_k1_MAT_s104
                                 + 1.22899 w_k1_MSFT_s104
                                 + 1.02846 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.12413 w_k1_ADM_s104
                                 + 1.17978 w_k1_AXP_s104
                                 + 1.35882 w_k1_DIS_s104
                                 + 1.05136 w_k1_FDX_s104
                                 + 1.72579 w_k1_IBM_s104
                                 + 1.06838 w_k1_INTC_s104
                                 + 1.21165 w_k1_KO_s104 + 1.40588 w_k1_MAT_s104
                                 + 1.06913 w_k1_MSFT_s104
                                 + 1.19382 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.22061 w_k1_ADM_s104
                                 + 1.18479 w_k1_AXP_s104
                                 + 1.44393 w_k1_DIS_s104
                                 + 1.31052 w_k1_FDX_s104
                                 + 1.03663 w_k1_IBM_s104
                                 + 1.15333 w_k1_INTC_s104
                                 + 1.03979 w_k1_KO_s104 + 1.10633 w_k1_MAT_s104
                                 + 1.17562 w_k1_MSFT_s104
                                 + 1.19375 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.0481 w_k1_ADM_s104
                                 + 0.908151 w_k1_AXP_s104
                                 + 1.01782 w_k1_DIS_s104
                                 + 0.984638 w_k1_FDX_s104
                                 + 0.84975 w_k1_IBM_s104
                                 + 1.05086 w_k1_INTC_s104
                                 + 0.919453 w_k1_KO_s104
                                 + 0.852525 w_k1_MAT_s104
                                 + 1.07028 w_k1_MSFT_s104
                                 + 0.962889 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.925583 w_k1_ADM_s104
                                 + 0.89351 w_k1_AXP_s104
                                 + 0.83962 w_k1_DIS_s104
                                 + 1.01598 w_k1_FDX_s104
                                 + 0.558701 w_k1_IBM_s104
                                 + 0.929544 w_k1_INTC_s104
                                 + 0.656214 w_k1_KO_s104
                                 + 0.457262 w_k1_MAT_s104
                                 + 0.876976 w_k1_MSFT_s104
                                 + 0.839231 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.85995 w_k1_ADM_s105
                                 + 0.842884 w_k1_AXP_s105
                                 + 0.855853 w_k1_DIS_s105
                                 + 0.920897 w_k1_FDX_s105
                                 + 0.872839 w_k1_IBM_s105
                                 + 1.01719 w_k1_INTC_s105
                                 + 1.00022 w_k1_KO_s105
                                 + 0.888215 w_k1_MAT_s105
                                 + 1.02088 w_k1_MSFT_s105
                                 + 0.924948 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.04861 w_k1_ADM_s105
                                 + 1.06544 w_k1_AXP_s105
                                 + 1.20838 w_k1_DIS_s105
                                 + 1.20433 w_k1_FDX_s105
                                 + 0.599976 w_k1_IBM_s105
                                 + 1.10403 w_k1_INTC_s105
                                 + 0.813172 w_k1_KO_s105
                                 + 0.983421 w_k1_MAT_s105
                                 + 1.1809 w_k1_MSFT_s105 + 0.94253 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.759103 w_k1_ADM_s105
                                 + 0.611386 w_k1_AXP_s105
                                 + 0.657576 w_k1_DIS_s105
                                 + 0.787577 w_k1_FDX_s105
                                 + 0.560657 w_k1_IBM_s105
                                 + 0.792917 w_k1_INTC_s105
                                 + 0.626414 w_k1_KO_s105
                                 + 0.638577 w_k1_MAT_s105
                                 + 0.774999 w_k1_MSFT_s105
                                 + 0.838377 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.826632 w_k1_ADM_s105
                                 + 0.90282 w_k1_AXP_s105 + 1.0439 w_k1_DIS_s105
                                 + 0.961938 w_k1_FDX_s105
                                 + 1.5108 w_k1_IBM_s105
                                 + 1.10484 w_k1_INTC_s105
                                 + 1.23493 w_k1_KO_s105 + 1.98858 w_k1_MAT_s105
                                 + 1.04405 w_k1_MSFT_s105
                                 + 1.04035 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.674859 w_k1_ADM_s105
                                 + 0.632451 w_k1_AXP_s105
                                 + 0.597592 w_k1_DIS_s105
                                 + 0.674233 w_k1_FDX_s105
                                 + 0.990164 w_k1_IBM_s105
                                 + 0.838015 w_k1_INTC_s105
                                 + 0.895864 w_k1_KO_s105
                                 + 0.925617 w_k1_MAT_s105
                                 + 0.759493 w_k1_MSFT_s105
                                 + 0.87472 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.713602 w_k1_ADM_s105
                                 + 1.00305 w_k1_AXP_s105
                                 + 0.819806 w_k1_DIS_s105
                                 + 0.85302 w_k1_FDX_s105
                                 + 1.45434 w_k1_IBM_s105
                                 + 0.910667 w_k1_INTC_s105
                                 + 1.08969 w_k1_KO_s105 + 1.6661 w_k1_MAT_s105
                                 + 0.801097 w_k1_MSFT_s105
                                 + 1.03428 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.00881 w_k1_ADM_s105
                                 + 1.03964 w_k1_AXP_s105 + 1.3524 w_k1_DIS_s105
                                 + 1.13821 w_k1_FDX_s105
                                 + 1.27371 w_k1_IBM_s105
                                 + 1.07687 w_k1_INTC_s105 + 1.3009 w_k1_KO_s105
                                 + 1.93286 w_k1_MAT_s105
                                 + 1.24744 w_k1_MSFT_s105
                                 + 1.21505 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.25881 w_k1_ADM_s105
                                 + 1.52673 w_k1_AXP_s105
                                 + 1.33078 w_k1_DIS_s105
                                 + 1.31328 w_k1_FDX_s105
                                 + 1.23603 w_k1_IBM_s105
                                 + 1.15158 w_k1_INTC_s105
                                 + 1.23962 w_k1_KO_s105 + 1.63322 w_k1_MAT_s105
                                 + 1.0937 w_k1_MSFT_s105 + 1.21301 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.15163 w_k1_ADM_s106
                                 + 0.958294 w_k1_AXP_s106
                                 + 0.860501 w_k1_DIS_s106
                                 + 0.948332 w_k1_FDX_s106
                                 + 0.893208 w_k1_IBM_s106
                                 + 1.03668 w_k1_INTC_s106
                                 + 0.932522 w_k1_KO_s106
                                 + 0.745599 w_k1_MAT_s106
                                 + 1.02386 w_k1_MSFT_s106
                                 + 0.989314 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.0301 w_k1_ADM_s106
                                 + 0.963474 w_k1_AXP_s106
                                 + 1.2345 w_k1_DIS_s106 + 1.14872 w_k1_FDX_s106
                                 + 0.904133 w_k1_IBM_s106
                                 + 1.06713 w_k1_INTC_s106
                                 + 1.07974 w_k1_KO_s106 + 0.90738 w_k1_MAT_s106
                                 + 1.19827 w_k1_MSFT_s106
                                 + 0.964266 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.825581 w_k1_ADM_s106
                                 + 1.30271 w_k1_AXP_s106
                                 + 1.23385 w_k1_DIS_s106
                                 + 1.05011 w_k1_FDX_s106
                                 + 1.66558 w_k1_IBM_s106
                                 + 1.01694 w_k1_INTC_s106
                                 + 1.24492 w_k1_KO_s106 + 2.12766 w_k1_MAT_s106
                                 + 1.05445 w_k1_MSFT_s106
                                 + 1.25382 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.920519 w_k1_ADM_s106
                                 + 0.957095 w_k1_AXP_s106
                                 + 0.892621 w_k1_DIS_s106
                                 + 0.88515 w_k1_FDX_s106
                                 + 1.27138 w_k1_IBM_s106
                                 + 1.01701 w_k1_INTC_s106
                                 + 0.977845 w_k1_KO_s106
                                 + 1.24491 w_k1_MAT_s106
                                 + 0.874607 w_k1_MSFT_s106
                                 + 0.922109 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.1674 w_k1_ADM_s106
                                 + 1.27561 w_k1_AXP_s106 + 1.287 w_k1_DIS_s106
                                 + 1.21965 w_k1_FDX_s106
                                 + 1.00665 w_k1_IBM_s106
                                 + 0.986113 w_k1_INTC_s106
                                 + 1.00012 w_k1_KO_s106 + 1.43577 w_k1_MAT_s106
                                 + 1.14642 w_k1_MSFT_s106 + 1.1231 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.704943 w_k1_ADM_s106
                                 + 0.739028 w_k1_AXP_s106
                                 + 0.732292 w_k1_DIS_s106
                                 + 0.844382 w_k1_FDX_s106
                                 + 0.819185 w_k1_IBM_s106
                                 + 0.980016 w_k1_INTC_s106
                                 + 0.794818 w_k1_KO_s106
                                 + 0.577844 w_k1_MAT_s106
                                 + 0.845095 w_k1_MSFT_s106
                                 + 0.829445 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.1092 w_k1_ADM_s106
                                 + 1.38178 w_k1_AXP_s106
                                 + 1.08089 w_k1_DIS_s106 + 1.1036 w_k1_FDX_s106
                                 + 1.29629 w_k1_IBM_s106
                                 + 0.938024 w_k1_INTC_s106
                                 + 1.09439 w_k1_KO_s106 + 1.90974 w_k1_MAT_s106
                                 + 1.02493 w_k1_MSFT_s106
                                 + 1.20708 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.944555 w_k1_ADM_s106
                                 + 0.819076 w_k1_AXP_s106
                                 + 1.22933 w_k1_DIS_s106 + 1.0259 w_k1_FDX_s106
                                 + 1.24763 w_k1_IBM_s106
                                 + 1.09754 w_k1_INTC_s106
                                 + 1.21021 w_k1_KO_s106
                                 + 0.999362 w_k1_MAT_s106
                                 + 1.13138 w_k1_MSFT_s106
                                 + 1.08656 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.19626 w_k1_ADM_s107
                                 + 1.25356 w_k1_AXP_s107
                                 + 1.02918 w_k1_DIS_s107
                                 + 1.18458 w_k1_FDX_s107
                                 + 1.03327 w_k1_IBM_s107
                                 + 1.03547 w_k1_INTC_s107
                                 + 0.892025 w_k1_KO_s107
                                 + 1.05644 w_k1_MAT_s107
                                 + 1.00625 w_k1_MSFT_s107
                                 + 0.999777 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.680958 w_k1_ADM_s107
                                 + 0.743891 w_k1_AXP_s107
                                 + 0.774179 w_k1_DIS_s107
                                 + 0.764275 w_k1_FDX_s107
                                 + 0.982838 w_k1_IBM_s107
                                 + 0.90542 w_k1_INTC_s107
                                 + 0.816641 w_k1_KO_s107
                                 + 0.88091 w_k1_MAT_s107
                                 + 0.827859 w_k1_MSFT_s107
                                 + 0.852236 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.810319 w_k1_ADM_s107
                                 + 0.978407 w_k1_AXP_s107
                                 + 1.43584 w_k1_DIS_s107
                                 + 1.03922 w_k1_FDX_s107
                                 + 1.22239 w_k1_IBM_s107
                                 + 0.976134 w_k1_INTC_s107
                                 + 0.975871 w_k1_KO_s107
                                 + 1.05848 w_k1_MAT_s107
                                 + 1.03938 w_k1_MSFT_s107 + 1.0821 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.9784 w_k1_ADM_s107
                                 + 1.3673 w_k1_AXP_s107 + 1.05237 w_k1_DIS_s107
                                 + 1.07909 w_k1_FDX_s107 + 1.4057 w_k1_IBM_s107
                                 + 1.28326 w_k1_INTC_s107
                                 + 1.06082 w_k1_KO_s107 + 1.42446 w_k1_MAT_s107
                                 + 1.08798 w_k1_MSFT_s107
                                 + 1.12282 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.787998 w_k1_ADM_s107
                                 + 0.885269 w_k1_AXP_s107
                                 + 0.8963 w_k1_DIS_s107 + 1.08632 w_k1_FDX_s107
                                 + 0.523985 w_k1_IBM_s107
                                 + 1.00208 w_k1_INTC_s107
                                 + 0.614455 w_k1_KO_s107
                                 + 0.51087 w_k1_MAT_s107
                                 + 0.925732 w_k1_MSFT_s107
                                 + 0.938675 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.07965 w_k1_ADM_s107
                                 + 1.54197 w_k1_AXP_s107
                                 + 1.64047 w_k1_DIS_s107
                                 + 1.32848 w_k1_FDX_s107 + 1.5216 w_k1_IBM_s107
                                 + 1.32381 w_k1_INTC_s107
                                 + 1.26458 w_k1_KO_s107 + 1.70071 w_k1_MAT_s107
                                 + 1.32084 w_k1_MSFT_s107
                                 + 1.32114 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.81817 w_k1_ADM_s107
                                 + 0.872536 w_k1_AXP_s107
                                 + 0.811874 w_k1_DIS_s107
                                 + 0.986991 w_k1_FDX_s107
                                 + 0.895266 w_k1_IBM_s107
                                 + 1.04566 w_k1_INTC_s107
                                 + 0.81205 w_k1_KO_s107 + 0.60781 w_k1_MAT_s107
                                 + 0.957855 w_k1_MSFT_s107
                                 + 1.03159 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.13136 w_k1_ADM_s107
                                 + 1.02293 w_k1_AXP_s107
                                 + 1.30818 w_k1_DIS_s107
                                 + 1.22982 w_k1_FDX_s107
                                 + 0.967032 w_k1_IBM_s107
                                 + 1.07555 w_k1_INTC_s107
                                 + 0.973046 w_k1_KO_s107
                                 + 0.996831 w_k1_MAT_s107
                                 + 1.21669 w_k1_MSFT_s107
                                 + 1.05251 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.769361 w_k1_ADM_s110
                                 + 0.795448 w_k1_AXP_s110
                                 + 0.891221 w_k1_DIS_s110
                                 + 0.782516 w_k1_FDX_s110
                                 + 1.4831 w_k1_IBM_s110
                                 + 0.883579 w_k1_INTC_s110
                                 + 1.02235 w_k1_KO_s110 + 1.47572 w_k1_MAT_s110
                                 + 0.85874 w_k1_MSFT_s110
                                 + 0.808459 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.689595 w_k1_ADM_s110
                                 + 0.692439 w_k1_AXP_s110
                                 + 0.829504 w_k1_DIS_s110
                                 + 0.718047 w_k1_FDX_s110
                                 + 1.07225 w_k1_IBM_s110
                                 + 0.892171 w_k1_INTC_s110
                                 + 0.930135 w_k1_KO_s110
                                 + 0.94038 w_k1_MAT_s110
                                 + 0.836016 w_k1_MSFT_s110
                                 + 0.886331 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.862355 w_k1_ADM_s110
                                 + 0.685983 w_k1_AXP_s110
                                 + 0.760642 w_k1_DIS_s110
                                 + 0.886138 w_k1_FDX_s110
                                 + 0.596709 w_k1_IBM_s110
                                 + 0.764944 w_k1_INTC_s110
                                 + 0.754297 w_k1_KO_s110
                                 + 0.642554 w_k1_MAT_s110
                                 + 0.827256 w_k1_MSFT_s110
                                 + 0.904434 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.05711 w_k1_ADM_s110
                                 + 1.18229 w_k1_AXP_s110
                                 + 1.72613 w_k1_DIS_s110
                                 + 1.30129 w_k1_FDX_s110
                                 + 1.28422 w_k1_IBM_s110
                                 + 1.25761 w_k1_INTC_s110
                                 + 1.47272 w_k1_KO_s110 + 1.66104 w_k1_MAT_s110
                                 + 1.29003 w_k1_MSFT_s110
                                 + 1.17305 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.960064 w_k1_ADM_s110
                                 + 0.852281 w_k1_AXP_s110
                                 + 0.987644 w_k1_DIS_s110
                                 + 0.951909 w_k1_FDX_s110
                                 + 0.810619 w_k1_IBM_s110
                                 + 0.770758 w_k1_INTC_s110
                                 + 0.828628 w_k1_KO_s110
                                 + 1.04158 w_k1_MAT_s110
                                 + 0.834275 w_k1_MSFT_s110
                                 + 0.905652 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.98931 w_k1_ADM_s110
                                 + 1.01644 w_k1_AXP_s110
                                 + 1.07535 w_k1_DIS_s110 + 1.0594 w_k1_FDX_s110
                                 + 1.12649 w_k1_IBM_s110
                                 + 1.10405 w_k1_INTC_s110 + 1.1217 w_k1_KO_s110
                                 + 1.25806 w_k1_MAT_s110
                                 + 1.11502 w_k1_MSFT_s110
                                 + 1.06306 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.837463 w_k1_ADM_s110
                                 + 0.792945 w_k1_AXP_s110
                                 + 0.690767 w_k1_DIS_s110
                                 + 0.882443 w_k1_FDX_s110
                                 + 0.94589 w_k1_IBM_s110
                                 + 0.920131 w_k1_INTC_s110
                                 + 0.789455 w_k1_KO_s110
                                 + 0.991666 w_k1_MAT_s110
                                 + 0.829085 w_k1_MSFT_s110
                                 + 0.87473 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.15427 w_k1_ADM_s110
                                 + 1.0011 w_k1_AXP_s110 + 1.2091 w_k1_DIS_s110
                                 + 1.31846 w_k1_FDX_s110
                                 + 0.788496 w_k1_IBM_s110
                                 + 1.17061 w_k1_INTC_s110
                                 + 0.981963 w_k1_KO_s110
                                 + 1.04955 w_k1_MAT_s110
                                 + 1.17132 w_k1_MSFT_s110
                                 + 1.07615 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.37583 w_k1_ADM_s111
                                 + 1.3578 w_k1_AXP_s111 + 1.68479 w_k1_DIS_s111
                                 + 1.39961 w_k1_FDX_s111 + 1.4321 w_k1_IBM_s111
                                 + 1.20134 w_k1_INTC_s111 + 1.4607 w_k1_KO_s111
                                 + 1.4121 w_k1_MAT_s111
                                 + 1.27944 w_k1_MSFT_s111
                                 + 1.21961 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.10753 w_k1_ADM_s111
                                 + 0.952896 w_k1_AXP_s111
                                 + 1.14726 w_k1_DIS_s111
                                 + 1.09295 w_k1_FDX_s111
                                 + 1.15611 w_k1_IBM_s111
                                 + 1.01174 w_k1_INTC_s111
                                 + 1.14326 w_k1_KO_s111 + 1.09484 w_k1_MAT_s111
                                 + 1.05146 w_k1_MSFT_s111
                                 + 1.10151 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.664184 w_k1_ADM_s111
                                 + 0.555353 w_k1_AXP_s111
                                 + 0.559951 w_k1_DIS_s111
                                 + 0.735778 w_k1_FDX_s111
                                 + 0.634256 w_k1_IBM_s111
                                 + 0.970641 w_k1_INTC_s111
                                 + 0.709412 w_k1_KO_s111
                                 + 0.52922 w_k1_MAT_s111
                                 + 0.808559 w_k1_MSFT_s111
                                 + 0.8695 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.18612 w_k1_ADM_s111
                                 + 1.17657 w_k1_AXP_s111
                                 + 1.35786 w_k1_DIS_s111 + 1.1305 w_k1_FDX_s111
                                 + 1.57963 w_k1_IBM_s111
                                 + 1.28852 w_k1_INTC_s111
                                 + 1.31369 w_k1_KO_s111 + 1.5629 w_k1_MAT_s111
                                 + 1.23201 w_k1_MSFT_s111
                                 + 1.17397 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.951878 w_k1_ADM_s111
                                 + 0.749212 w_k1_AXP_s111
                                 + 0.918777 w_k1_DIS_s111
                                 + 0.991585 w_k1_FDX_s111
                                 + 0.614668 w_k1_IBM_s111
                                 + 0.918211 w_k1_INTC_s111
                                 + 0.766262 w_k1_KO_s111
                                 + 0.631112 w_k1_MAT_s111
                                 + 0.913711 w_k1_MSFT_s111
                                 + 0.886466 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.947553 w_k1_ADM_s111
                                 + 0.706275 w_k1_AXP_s111
                                 + 0.753624 w_k1_DIS_s111
                                 + 0.778616 w_k1_FDX_s111
                                 + 0.954219 w_k1_IBM_s111
                                 + 0.76959 w_k1_INTC_s111
                                 + 0.756745 w_k1_KO_s111
                                 + 0.716492 w_k1_MAT_s111
                                 + 0.779076 w_k1_MSFT_s111
                                 + 0.874261 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.03002 w_k1_ADM_s111
                                 + 0.810638 w_k1_AXP_s111
                                 + 1.12989 w_k1_DIS_s111
                                 + 1.16277 w_k1_FDX_s111
                                 + 0.926605 w_k1_IBM_s111
                                 + 0.999264 w_k1_INTC_s111
                                 + 0.87695 w_k1_KO_s111
                                 + 0.687975 w_k1_MAT_s111
                                 + 0.94812 w_k1_MSFT_s111
                                 + 0.993216 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.00843 w_k1_ADM_s111
                                 + 0.785823 w_k1_AXP_s111
                                 + 0.971259 w_k1_DIS_s111
                                 + 0.931672 w_k1_FDX_s111
                                 + 1.38437 w_k1_IBM_s111
                                 + 1.08627 w_k1_INTC_s111
                                 + 1.18944 w_k1_KO_s111 + 1.02954 w_k1_MAT_s111
                                 + 0.88028 w_k1_MSFT_s111
                                 + 0.935737 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.35653 w_k1_ADM_s112
                                 + 0.994219 w_k1_AXP_s112
                                 + 1.20284 w_k1_DIS_s112
                                 + 1.19579 w_k1_FDX_s112
                                 + 0.806487 w_k1_IBM_s112
                                 + 1.02156 w_k1_INTC_s112
                                 + 0.87998 w_k1_KO_s112
                                 + 0.833835 w_k1_MAT_s112
                                 + 1.2495 w_k1_MSFT_s112 + 1.12065 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.965223 w_k1_ADM_s112
                                 + 0.838175 w_k1_AXP_s112
                                 + 1.08394 w_k1_DIS_s112
                                 + 1.00131 w_k1_FDX_s112
                                 + 1.15235 w_k1_IBM_s112
                                 + 0.957481 w_k1_INTC_s112
                                 + 1.07137 w_k1_KO_s112 + 1.15468 w_k1_MAT_s112
                                 + 1.10054 w_k1_MSFT_s112
                                 + 1.05876 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.888412 w_k1_ADM_s112
                                 + 0.689317 w_k1_AXP_s112
                                 + 0.752477 w_k1_DIS_s112
                                 + 0.781085 w_k1_FDX_s112
                                 + 0.978499 w_k1_IBM_s112
                                 + 0.901069 w_k1_INTC_s112
                                 + 0.914046 w_k1_KO_s112
                                 + 0.886122 w_k1_MAT_s112
                                 + 0.843532 w_k1_MSFT_s112
                                 + 0.896194 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.913527 w_k1_ADM_s112
                                 + 0.963456 w_k1_AXP_s112
                                 + 0.768038 w_k1_DIS_s112
                                 + 0.8517 w_k1_FDX_s112
                                 + 0.850656 w_k1_IBM_s112
                                 + 0.750314 w_k1_INTC_s112
                                 + 0.833678 w_k1_KO_s112
                                 + 0.987571 w_k1_MAT_s112
                                 + 0.888038 w_k1_MSFT_s112
                                 + 0.98306 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.938034 w_k1_ADM_s112
                                 + 0.743886 w_k1_AXP_s112
                                 + 0.74712 w_k1_DIS_s112
                                 + 0.905839 w_k1_FDX_s112
                                 + 0.635575 w_k1_IBM_s112
                                 + 0.923441 w_k1_INTC_s112
                                 + 0.754704 w_k1_KO_s112
                                 + 0.559662 w_k1_MAT_s112
                                 + 0.822697 w_k1_MSFT_s112
                                 + 0.883498 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.993118 w_k1_ADM_s112
                                 + 1.31266 w_k1_AXP_s112
                                 + 1.03948 w_k1_DIS_s112
                                 + 1.01363 w_k1_FDX_s112
                                 + 1.41019 w_k1_IBM_s112
                                 + 0.859548 w_k1_INTC_s112
                                 + 0.979857 w_k1_KO_s112
                                 + 1.89811 w_k1_MAT_s112
                                 + 0.922043 w_k1_MSFT_s112
                                 + 1.08513 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.883079 w_k1_ADM_s112
                                 + 0.576461 w_k1_AXP_s112
                                 + 1.21761 w_k1_DIS_s112
                                 + 0.901635 w_k1_FDX_s112
                                 + 1.00753 w_k1_IBM_s112
                                 + 0.830707 w_k1_INTC_s112
                                 + 1.0921 w_k1_KO_s112 + 0.956601 w_k1_MAT_s112
                                 + 1.06903 w_k1_MSFT_s112
                                 + 1.00934 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.56386 w_k1_ADM_s112
                                 + 1.26749 w_k1_AXP_s112
                                 + 1.61864 w_k1_DIS_s112
                                 + 1.32341 w_k1_FDX_s112
                                 + 1.26871 w_k1_IBM_s112
                                 + 1.08115 w_k1_INTC_s112
                                 + 1.34385 w_k1_KO_s112 + 1.4868 w_k1_MAT_s112
                                 + 1.43406 w_k1_MSFT_s112
                                 + 1.23832 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.98921 w_k1_ADM_s113
                                 + 1.05943 w_k1_AXP_s113
                                 + 0.734078 w_k1_DIS_s113
                                 + 0.85637 w_k1_FDX_s113
                                 + 1.02184 w_k1_IBM_s113
                                 + 0.885431 w_k1_INTC_s113
                                 + 0.789236 w_k1_KO_s113
                                 + 1.06053 w_k1_MAT_s113
                                 + 0.907938 w_k1_MSFT_s113
                                 + 0.854518 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.933787 w_k1_ADM_s113
                                 + 0.998452 w_k1_AXP_s113
                                 + 1.00328 w_k1_DIS_s113
                                 + 0.934096 w_k1_FDX_s113
                                 + 1.28271 w_k1_IBM_s113
                                 + 1.11347 w_k1_INTC_s113
                                 + 1.07041 w_k1_KO_s113 + 1.3443 w_k1_MAT_s113
                                 + 0.990888 w_k1_MSFT_s113
                                 + 0.89478 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.28244 w_k1_ADM_s113
                                 + 1.45911 w_k1_AXP_s113 + 1.562 w_k1_DIS_s113
                                 + 1.2898 w_k1_FDX_s113 + 1.15869 w_k1_IBM_s113
                                 + 0.868945 w_k1_INTC_s113
                                 + 1.09386 w_k1_KO_s113 + 1.15171 w_k1_MAT_s113
                                 + 1.05265 w_k1_MSFT_s113
                                 + 1.13398 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.787589 w_k1_ADM_s113
                                 + 0.69505 w_k1_AXP_s113
                                 + 0.658352 w_k1_DIS_s113
                                 + 0.801569 w_k1_FDX_s113
                                 + 0.723255 w_k1_IBM_s113
                                 + 0.848994 w_k1_INTC_s113
                                 + 0.728268 w_k1_KO_s113
                                 + 0.67815 w_k1_MAT_s113
                                 + 0.860716 w_k1_MSFT_s113
                                 + 0.819547 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.969118 w_k1_ADM_s113
                                 + 1.20915 w_k1_AXP_s113
                                 + 1.09241 w_k1_DIS_s113
                                 + 1.17438 w_k1_FDX_s113
                                 + 0.793165 w_k1_IBM_s113
                                 + 0.984876 w_k1_INTC_s113
                                 + 0.834562 w_k1_KO_s113
                                 + 0.934173 w_k1_MAT_s113
                                 + 1.03971 w_k1_MSFT_s113
                                 + 0.943493 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.03266 w_k1_ADM_s113
                                 + 1.37478 w_k1_AXP_s113
                                 + 1.20851 w_k1_DIS_s113
                                 + 1.01855 w_k1_FDX_s113
                                 + 1.57661 w_k1_IBM_s113
                                 + 1.09621 w_k1_INTC_s113
                                 + 1.19676 w_k1_KO_s113 + 1.43126 w_k1_MAT_s113
                                 + 1.03681 w_k1_MSFT_s113
                                 + 1.06837 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.690151 w_k1_ADM_s113
                                 + 0.654668 w_k1_AXP_s113
                                 + 0.70788 w_k1_DIS_s113
                                 + 0.693624 w_k1_FDX_s113
                                 + 1.23394 w_k1_IBM_s113
                                 + 0.7469 w_k1_INTC_s113
                                 + 0.853014 w_k1_KO_s113
                                 + 0.952631 w_k1_MAT_s113
                                 + 0.778838 w_k1_MSFT_s113
                                 + 0.874367 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.08024 w_k1_ADM_s113
                                 + 1.04883 w_k1_AXP_s113
                                 + 1.19551 w_k1_DIS_s113
                                 + 1.07304 w_k1_FDX_s113
                                 + 0.897023 w_k1_IBM_s113
                                 + 0.877905 w_k1_INTC_s113
                                 + 0.921929 w_k1_KO_s113
                                 + 0.741028 w_k1_MAT_s113
                                 + 1.05693 w_k1_MSFT_s113 + 1.02 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.24997 w_k1_ADM_s114
                                 + 1.2938 w_k1_AXP_s114 + 1.35426 w_k1_DIS_s114
                                 + 1.40256 w_k1_FDX_s114
                                 + 0.838629 w_k1_IBM_s114
                                 + 1.10509 w_k1_INTC_s114
                                 + 1.02728 w_k1_KO_s114
                                 + 0.808447 w_k1_MAT_s114
                                 + 1.18479 w_k1_MSFT_s114
                                 + 1.19305 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.15372 w_k1_ADM_s114
                                 + 1.40874 w_k1_AXP_s114
                                 + 1.41337 w_k1_DIS_s114
                                 + 1.13762 w_k1_FDX_s114
                                 + 1.14693 w_k1_IBM_s114
                                 + 1.0111 w_k1_INTC_s114 + 1.26263 w_k1_KO_s114
                                 + 1.21283 w_k1_MAT_s114
                                 + 1.18476 w_k1_MSFT_s114
                                 + 1.21833 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.790014 w_k1_ADM_s114
                                 + 0.885831 w_k1_AXP_s114
                                 + 0.607649 w_k1_DIS_s114
                                 + 0.680032 w_k1_FDX_s114
                                 + 1.12226 w_k1_IBM_s114
                                 + 0.834019 w_k1_INTC_s114
                                 + 0.882448 w_k1_KO_s114
                                 + 1.07302 w_k1_MAT_s114
                                 + 0.766253 w_k1_MSFT_s114
                                 + 0.920273 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.853564 w_k1_ADM_s114
                                 + 1.13261 w_k1_AXP_s114
                                 + 0.82787 w_k1_DIS_s114
                                 + 0.927633 w_k1_FDX_s114
                                 + 1.09422 w_k1_IBM_s114
                                 + 1.05289 w_k1_INTC_s114
                                 + 0.890289 w_k1_KO_s114
                                 + 0.910868 w_k1_MAT_s114
                                 + 0.82985 w_k1_MSFT_s114
                                 + 0.971359 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.0551 w_k1_ADM_s114
                                 + 0.915731 w_k1_AXP_s114
                                 + 1.05272 w_k1_DIS_s114
                                 + 0.965493 w_k1_FDX_s114
                                 + 0.870372 w_k1_IBM_s114
                                 + 0.908391 w_k1_INTC_s114
                                 + 0.959714 w_k1_KO_s114
                                 + 0.72188 w_k1_MAT_s114
                                 + 1.04655 w_k1_MSFT_s114 + 1.1412 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.660841 w_k1_ADM_s114
                                 + 0.687492 w_k1_AXP_s114
                                 + 0.69327 w_k1_DIS_s114
                                 + 0.706531 w_k1_FDX_s114
                                 + 0.830088 w_k1_IBM_s114
                                 + 0.785768 w_k1_INTC_s114
                                 + 0.753588 w_k1_KO_s114 + 0.7869 w_k1_MAT_s114
                                 + 0.755447 w_k1_MSFT_s114
                                 + 0.86656 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.10764 w_k1_ADM_s114
                                 + 1.10329 w_k1_AXP_s114
                                 + 1.13788 w_k1_DIS_s114 + 1.021 w_k1_FDX_s114
                                 + 1.44229 w_k1_IBM_s114
                                 + 0.977332 w_k1_INTC_s114
                                 + 1.32572 w_k1_KO_s114 + 1.79516 w_k1_MAT_s114
                                 + 1.12588 w_k1_MSFT_s114
                                 + 1.13994 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.910811 w_k1_ADM_s114
                                 + 0.938982 w_k1_AXP_s114
                                 + 0.800301 w_k1_DIS_s114
                                 + 0.932525 w_k1_FDX_s114
                                 + 0.770616 w_k1_IBM_s114
                                 + 0.897876 w_k1_INTC_s114
                                 + 0.853904 w_k1_KO_s114
                                 + 0.775199 w_k1_MAT_s114
                                 + 0.899703 w_k1_MSFT_s114
                                 + 0.997093 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.918533 w_k1_ADM_s115
                                 + 1.21308 w_k1_AXP_s115
                                 + 1.32296 w_k1_DIS_s115
                                 + 1.09601 w_k1_FDX_s115
                                 + 1.22878 w_k1_IBM_s115
                                 + 1.34884 w_k1_INTC_s115
                                 + 1.32599 w_k1_KO_s115
                                 + 0.940581 w_k1_MAT_s115
                                 + 1.20265 w_k1_MSFT_s115
                                 + 1.02855 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.979587 w_k1_ADM_s115
                                 + 0.997426 w_k1_AXP_s115
                                 + 0.971403 w_k1_DIS_s115
                                 + 1.05056 w_k1_FDX_s115
                                 + 1.01825 w_k1_IBM_s115
                                 + 0.940003 w_k1_INTC_s115
                                 + 0.990147 w_k1_KO_s115
                                 + 1.16469 w_k1_MAT_s115
                                 + 1.08791 w_k1_MSFT_s115
                                 + 0.87273 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.12863 w_k1_ADM_s115
                                 + 1.70036 w_k1_AXP_s115
                                 + 1.82192 w_k1_DIS_s115
                                 + 1.45357 w_k1_FDX_s115
                                 + 1.29695 w_k1_IBM_s115
                                 + 1.35228 w_k1_INTC_s115
                                 + 1.37436 w_k1_KO_s115 + 1.2482 w_k1_MAT_s115
                                 + 1.40331 w_k1_MSFT_s115
                                 + 1.12605 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.00573 w_k1_ADM_s115
                                 + 0.905893 w_k1_AXP_s115
                                 + 0.775477 w_k1_DIS_s115
                                 + 0.950467 w_k1_FDX_s115
                                 + 0.80395 w_k1_IBM_s115
                                 + 1.10597 w_k1_INTC_s115
                                 + 0.846866 w_k1_KO_s115
                                 + 0.812888 w_k1_MAT_s115
                                 + 0.946813 w_k1_MSFT_s115
                                 + 0.963198 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.970608 w_k1_ADM_s115
                                 + 0.902188 w_k1_AXP_s115
                                 + 1.14252 w_k1_DIS_s115
                                 + 1.09866 w_k1_FDX_s115
                                 + 0.668662 w_k1_IBM_s115
                                 + 1.13049 w_k1_INTC_s115
                                 + 0.910521 w_k1_KO_s115
                                 + 0.41035 w_k1_MAT_s115
                                 + 1.09992 w_k1_MSFT_s115
                                 + 0.866555 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.12153 w_k1_ADM_s115
                                 + 1.21328 w_k1_AXP_s115
                                 + 1.30782 w_k1_DIS_s115
                                 + 1.12957 w_k1_FDX_s115
                                 + 1.38152 w_k1_IBM_s115
                                 + 0.9797 w_k1_INTC_s115 + 1.20918 w_k1_KO_s115
                                 + 1.19239 w_k1_MAT_s115
                                 + 1.00433 w_k1_MSFT_s115
                                 + 1.08853 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.872938 w_k1_ADM_s115
                                 + 0.940196 w_k1_AXP_s115
                                 + 0.844613 w_k1_DIS_s115
                                 + 0.82669 w_k1_FDX_s115
                                 + 1.16722 w_k1_IBM_s115
                                 + 0.842679 w_k1_INTC_s115
                                 + 0.916494 w_k1_KO_s115
                                 + 1.18621 w_k1_MAT_s115
                                 + 0.834196 w_k1_MSFT_s115
                                 + 0.822942 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.760303 w_k1_ADM_s115
                                 + 0.682594 w_k1_AXP_s115
                                 + 0.645803 w_k1_DIS_s115
                                 + 0.795602 w_k1_FDX_s115
                                 + 0.711181 w_k1_IBM_s115
                                 + 0.685379 w_k1_INTC_s115
                                 + 0.722586 w_k1_KO_s115
                                 + 0.61376 w_k1_MAT_s115
                                 + 0.790023 w_k1_MSFT_s115
                                 + 0.763721 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.30605 w_k1_ADM_s116
                                 + 1.33253 w_k1_AXP_s116
                                 + 1.59258 w_k1_DIS_s116
                                 + 1.38556 w_k1_FDX_s116
                                 + 0.910158 w_k1_IBM_s116
                                 + 1.38446 w_k1_INTC_s116
                                 + 1.31238 w_k1_KO_s116
                                 + 0.867494 w_k1_MAT_s116
                                 + 1.3133 w_k1_MSFT_s116 + 1.29849 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.27612 w_k1_ADM_s116
                                 + 0.999472 w_k1_AXP_s116
                                 + 0.948494 w_k1_DIS_s116
                                 + 0.906404 w_k1_FDX_s116
                                 + 1.72276 w_k1_IBM_s116
                                 + 0.92223 w_k1_INTC_s116
                                 + 1.18799 w_k1_KO_s116 + 1.42623 w_k1_MAT_s116
                                 + 0.933497 w_k1_MSFT_s116
                                 + 1.10678 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.829387 w_k1_ADM_s116
                                 + 1.39129 w_k1_AXP_s116 + 1.6618 w_k1_DIS_s116
                                 + 1.36396 w_k1_FDX_s116
                                 + 1.38084 w_k1_IBM_s116
                                 + 1.37785 w_k1_INTC_s116
                                 + 1.53986 w_k1_KO_s116 + 1.29692 w_k1_MAT_s116
                                 + 1.32214 w_k1_MSFT_s116
                                 + 1.27393 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.12044 w_k1_ADM_s116
                                 + 1.09822 w_k1_AXP_s116
                                 + 1.15641 w_k1_DIS_s116
                                 + 1.20635 w_k1_FDX_s116
                                 + 0.988616 w_k1_IBM_s116
                                 + 1.14466 w_k1_INTC_s116
                                 + 1.14324 w_k1_KO_s116 + 1.00373 w_k1_MAT_s116
                                 + 1.06393 w_k1_MSFT_s116
                                 + 1.15388 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.20676 w_k1_ADM_s116
                                 + 1.19069 w_k1_AXP_s116
                                 + 1.03968 w_k1_DIS_s116
                                 + 1.13929 w_k1_FDX_s116
                                 + 0.752903 w_k1_IBM_s116
                                 + 1.22269 w_k1_INTC_s116
                                 + 1.00685 w_k1_KO_s116
                                 + 0.865531 w_k1_MAT_s116
                                 + 1.01724 w_k1_MSFT_s116 + 1.1815 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.761998 w_k1_ADM_s116
                                 + 0.791831 w_k1_AXP_s116
                                 + 0.905324 w_k1_DIS_s116
                                 + 0.839665 w_k1_FDX_s116
                                 + 1.23005 w_k1_IBM_s116
                                 + 0.891532 w_k1_INTC_s116
                                 + 0.977376 w_k1_KO_s116
                                 + 0.961807 w_k1_MAT_s116
                                 + 0.836582 w_k1_MSFT_s116
                                 + 0.982341 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.816306 w_k1_ADM_s116
                                 + 0.634326 w_k1_AXP_s116
                                 + 0.770243 w_k1_DIS_s116
                                 + 0.83828 w_k1_FDX_s116
                                 + 0.801625 w_k1_IBM_s116
                                 + 1.10617 w_k1_INTC_s116
                                 + 0.903285 w_k1_KO_s116
                                 + 0.644539 w_k1_MAT_s116
                                 + 0.967182 w_k1_MSFT_s116
                                 + 0.894783 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.917999 w_k1_ADM_s116
                                 + 1.25474 w_k1_AXP_s116
                                 + 1.11816 w_k1_DIS_s116
                                 + 1.11485 w_k1_FDX_s116
                                 + 1.05405 w_k1_IBM_s116
                                 + 1.17364 w_k1_INTC_s116
                                 + 1.22025 w_k1_KO_s116 + 1.2652 w_k1_MAT_s116
                                 + 1.05098 w_k1_MSFT_s116 + 1.1136 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.90039 w_k1_ADM_s117
                                 + 1.30154 w_k1_AXP_s117
                                 + 1.08573 w_k1_DIS_s117
                                 + 1.11578 w_k1_FDX_s117
                                 + 1.02807 w_k1_IBM_s117
                                 + 1.07606 w_k1_INTC_s117
                                 + 1.10179 w_k1_KO_s117 + 1.28465 w_k1_MAT_s117
                                 + 1.08056 w_k1_MSFT_s117
                                 + 1.13823 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.936781 w_k1_ADM_s117
                                 + 0.962555 w_k1_AXP_s117
                                 + 1.22242 w_k1_DIS_s117
                                 + 1.09231 w_k1_FDX_s117
                                 + 0.759351 w_k1_IBM_s117
                                 + 1.039 w_k1_INTC_s117 + 1.11848 w_k1_KO_s117
                                 + 0.944451 w_k1_MAT_s117
                                 + 1.16456 w_k1_MSFT_s117
                                 + 1.03532 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.895489 w_k1_ADM_s117
                                 + 0.862653 w_k1_AXP_s117
                                 + 0.703464 w_k1_DIS_s117
                                 + 0.790533 w_k1_FDX_s117
                                 + 0.981201 w_k1_IBM_s117
                                 + 0.812923 w_k1_INTC_s117
                                 + 0.911155 w_k1_KO_s117
                                 + 0.934116 w_k1_MAT_s117
                                 + 0.856358 w_k1_MSFT_s117
                                 + 0.987969 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.952106 w_k1_ADM_s117
                                 + 1.06048 w_k1_AXP_s117
                                 + 1.42079 w_k1_DIS_s117
                                 + 1.10646 w_k1_FDX_s117
                                 + 1.16626 w_k1_IBM_s117
                                 + 1.13319 w_k1_INTC_s117
                                 + 1.20647 w_k1_KO_s117 + 1.05138 w_k1_MAT_s117
                                 + 1.22492 w_k1_MSFT_s117
                                 + 1.29741 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.29355 w_k1_ADM_s117
                                 + 1.50463 w_k1_AXP_s117
                                 + 1.81068 w_k1_DIS_s117
                                 + 1.38905 w_k1_FDX_s117
                                 + 0.978964 w_k1_IBM_s117
                                 + 1.27041 w_k1_INTC_s117
                                 + 1.26973 w_k1_KO_s117 + 1.43023 w_k1_MAT_s117
                                 + 1.3606 w_k1_MSFT_s117 + 1.37378 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.968993 w_k1_ADM_s117
                                 + 1.0092 w_k1_AXP_s117
                                 + 0.939987 w_k1_DIS_s117
                                 + 0.969923 w_k1_FDX_s117
                                 + 1.27004 w_k1_IBM_s117
                                 + 1.01316 w_k1_INTC_s117
                                 + 0.901113 w_k1_KO_s117
                                 + 0.893521 w_k1_MAT_s117
                                 + 0.914954 w_k1_MSFT_s117
                                 + 1.03786 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.815452 w_k1_ADM_s117
                                 + 0.881441 w_k1_AXP_s117
                                 + 0.73054 w_k1_DIS_s117
                                 + 0.956648 w_k1_FDX_s117
                                 + 0.680833 w_k1_IBM_s117
                                 + 0.829387 w_k1_INTC_s117
                                 + 0.737687 w_k1_KO_s117
                                 + 0.853461 w_k1_MAT_s117
                                 + 0.953999 w_k1_MSFT_s117
                                 + 0.908131 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.885318 w_k1_ADM_s117
                                 + 1.19129 w_k1_AXP_s117
                                 + 1.27407 w_k1_DIS_s117
                                 + 1.04653 w_k1_FDX_s117
                                 + 1.60955 w_k1_IBM_s117
                                 + 1.19182 w_k1_INTC_s117
                                 + 1.44295 w_k1_KO_s117 + 2.01743 w_k1_MAT_s117
                                 + 1.27069 w_k1_MSFT_s117
                                 + 1.20286 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.011840436193073 w_k1_ADM_s000
                                 + 0.0125362800467509 w_k1_ADM_s001
                                 + 0.0222934045775525 w_k1_ADM_s002
                                 + 0.0113935295403355 w_k1_ADM_s003
                                 + 0.0315782468333999 w_k1_ADM_s004
                                 + 0.0152913829322154 w_k1_ADM_s005
                                 + 0.0104617250748091 w_k1_ADM_s006
                                 + 0.0518552172808826 w_k1_ADM_s007
                                 + 0.0215457177438012 w_k1_ADM_s010
                                 + 0.0322761839486683 w_k1_ADM_s011
                                 + 0.0261874975638528 w_k1_ADM_s012
                                 + 0.0131517535821029 w_k1_ADM_s013
                                 + 0.0274144478299997 w_k1_ADM_s014
                                 + 0.0229613592171726 w_k1_ADM_s015
                                 + 0.0146371909621451 w_k1_ADM_s016
                                 + 0.0257033332021716 w_k1_ADM_s017
                                 + 0.0286724333239499 w_k1_ADM_s100
                                 + 0.0275866982416363 w_k1_ADM_s101
                                 + 0.0631360244384812 w_k1_ADM_s102
                                 + 0.0504810122010146 w_k1_ADM_s103
                                 + 0.0339014820797742 w_k1_ADM_s104
                                 + 0.0836371872962954 w_k1_ADM_s105
                                 + 0.0226745371076374 w_k1_ADM_s106
                                 + 0.0303610708318315 w_k1_ADM_s107
                                 + 0.0610769619654649 w_k1_ADM_s110
                                 + 0.0231337127247631 w_k1_ADM_s111
                                 + 0.0539519126065093 w_k1_ADM_s112
                                 + 0.0460819630340029 w_k1_ADM_s113
                                 + 0.0243266271645459 w_k1_ADM_s114
                                 + 0.0467814456722399 w_k1_ADM_s115
                                 + 0.0148017828993072 w_k1_ADM_s116
                                 + 0.0361226249332382 w_k1_ADM_s117
                                 + 0.0128875568956297 w_k1_AXP_s000
                                 + 0.0137980657076346 w_k1_AXP_s001
                                 + 0.0215174932158107 w_k1_AXP_s002
                                 + 0.0135561607559653 w_k1_AXP_s003
                                 + 0.0239207086010895 w_k1_AXP_s004
                                 + 0.0147799769637103 w_k1_AXP_s005
                                 + 0.0109001856180595 w_k1_AXP_s006
                                 + 0.0556079488794565 w_k1_AXP_s007
                                 + 0.0231624517787313 w_k1_AXP_s010
                                 + 0.0329530340805321 w_k1_AXP_s011
                                 + 0.0248515900268582 w_k1_AXP_s012
                                 + 0.016016481980877 w_k1_AXP_s013
                                 + 0.0303582069071041 w_k1_AXP_s014
                                 + 0.0176382511256886 w_k1_AXP_s015
                                 + 0.0178456470500281 w_k1_AXP_s016
                                 + 0.0262891419490671 w_k1_AXP_s017
                                 + 0.0360457609279629 w_k1_AXP_s100
                                 + 0.0222476329869802 w_k1_AXP_s101
                                 + 0.0730659917617961 w_k1_AXP_s102
                                 + 0.0478029351077768 w_k1_AXP_s103
                                 + 0.0322675877090598 w_k1_AXP_s104
                                 + 0.0889058545217754 w_k1_AXP_s105
                                 + 0.023256891950901 w_k1_AXP_s106
                                 + 0.0340047465777308 w_k1_AXP_s107
                                 + 0.0592715281474465 w_k1_AXP_s110
                                 + 0.0193096072794433 w_k1_AXP_s111
                                 + 0.0455697857923517 w_k1_AXP_s112
                                 + 0.0493477301678449 w_k1_AXP_s113
                                 + 0.0267741769150707 w_k1_AXP_s114
                                 + 0.0491110556817541 w_k1_AXP_s115
                                 + 0.0149359115551464 w_k1_AXP_s116
                                 + 0.0410812637113491 w_k1_AXP_s117
                                 + 0.0134099504671995 w_k1_DIS_s000
                                 + 0.0130626848170552 w_k1_DIS_s001
                                 + 0.0224529631395796 w_k1_DIS_s002
                                 + 0.0124844786468066 w_k1_DIS_s003
                                 + 0.0267369848009508 w_k1_DIS_s004
                                 + 0.0153160112352228 w_k1_DIS_s005
                                 + 0.0123361495043253 w_k1_DIS_s006
                                 + 0.0513278997188122 w_k1_DIS_s007
                                 + 0.0232635724078363 w_k1_DIS_s010
                                 + 0.0354544278508886 w_k1_DIS_s011
                                 + 0.0272040035009854 w_k1_DIS_s012
                                 + 0.0164753446714734 w_k1_DIS_s013
                                 + 0.0295004970280659 w_k1_DIS_s014
                                 + 0.0183578488945361 w_k1_DIS_s015
                                 + 0.0174572428891258 w_k1_DIS_s016
                                 + 0.026026481344568 w_k1_DIS_s017
                                 + 0.0321458978288837 w_k1_DIS_s100
                                 + 0.0243775126028941 w_k1_DIS_s101
                                 + 0.0678785729663254 w_k1_DIS_s102
                                 + 0.0501849206621014 w_k1_DIS_s103
                                 + 0.0329056276675825 w_k1_DIS_s104
                                 + 0.0941577827368562 w_k1_DIS_s105
                                 + 0.024021919429204 w_k1_DIS_s106
                                 + 0.0351873013808303 w_k1_DIS_s107
                                 + 0.0701603978576122 w_k1_DIS_s110
                                 + 0.0235534263901193 w_k1_DIS_s111
                                 + 0.0523531118087653 w_k1_DIS_s112
                                 + 0.0485862822929997 w_k1_DIS_s113
                                 + 0.0252094427043942 w_k1_DIS_s114
                                 + 0.050858741930402 w_k1_DIS_s115
                                 + 0.0162684884552096 w_k1_DIS_s116
                                 + 0.0403685826307159 w_k1_DIS_s117
                                 + 0.0121434641131584 w_k1_FDX_s000
                                 + 0.0123910066151358 w_k1_FDX_s001
                                 + 0.0220750099532048 w_k1_FDX_s002
                                 + 0.0130132282156816 w_k1_FDX_s003
                                 + 0.0244341136745871 w_k1_FDX_s004
                                 + 0.0143692922626553 w_k1_FDX_s005
                                 + 0.0110520022025487 w_k1_FDX_s006
                                 + 0.0511281473094642 w_k1_FDX_s007
                                 + 0.0198079879683289 w_k1_FDX_s010
                                 + 0.0355367828994928 w_k1_FDX_s011
                                 + 0.0248529992822901 w_k1_FDX_s012
                                 + 0.0159514538041875 w_k1_FDX_s013
                                 + 0.0303100908315327 w_k1_FDX_s014
                                 + 0.017046603671823 w_k1_FDX_s015
                                 + 0.0165783202047543 w_k1_FDX_s016
                                 + 0.0240640402935044 w_k1_FDX_s017
                                 + 0.0322543713509608 w_k1_FDX_s100
                                 + 0.0218672301806205 w_k1_FDX_s101
                                 + 0.0668678806840777 w_k1_FDX_s102
                                 + 0.0483242189801832 w_k1_FDX_s103
                                 + 0.0319960128008766 w_k1_FDX_s104
                                 + 0.0916257250375509 w_k1_FDX_s105
                                 + 0.0238167287559174 w_k1_FDX_s106
                                 + 0.035417811111048 w_k1_FDX_s107
                                 + 0.0664932144227941 w_k1_FDX_s110
                                 + 0.0232225930848315 w_k1_FDX_s111
                                 + 0.0500420513578957 w_k1_FDX_s112
                                 + 0.0463352026688583 w_k1_FDX_s113
                                 + 0.0243097614814787 w_k1_FDX_s114
                                 + 0.0495086863183147 w_k1_FDX_s115
                                 + 0.0158134604785715 w_k1_FDX_s116
                                 + 0.0389174422611839 w_k1_FDX_s117
                                 + 0.0125845124790792 w_k1_IBM_s000
                                 + 0.01321540017076 w_k1_IBM_s001
                                 + 0.0216607663553812 w_k1_IBM_s002
                                 + 0.0127503673625844 w_k1_IBM_s003
                                 + 0.0251437319978342 w_k1_IBM_s004
                                 + 0.0148094522671919 w_k1_IBM_s005
                                 + 0.0112247418098869 w_k1_IBM_s006
                                 + 0.0488643398340011 w_k1_IBM_s007
                                 + 0.0224230470485837 w_k1_IBM_s010
                                 + 0.0339786376631371 w_k1_IBM_s011
                                 + 0.0242419105593038 w_k1_IBM_s012
                                 + 0.016267396804148 w_k1_IBM_s013
                                 + 0.0303647818320008 w_k1_IBM_s014
                                 + 0.0183040824419487 w_k1_IBM_s015
                                 + 0.0162288552002424 w_k1_IBM_s016
                                 + 0.0260413371840833 w_k1_IBM_s017
                                 + 0.0318190283244605 w_k1_IBM_s100
                                 + 0.0226596338932578 w_k1_IBM_s101
                                 + 0.0741359794567427 w_k1_IBM_s102
                                 + 0.0501146248444505 w_k1_IBM_s103
                                 + 0.0327517335324306 w_k1_IBM_s104
                                 + 0.0972958450452564 w_k1_IBM_s105
                                 + 0.024755848395288 w_k1_IBM_s106
                                 + 0.0346440173323664 w_k1_IBM_s107
                                 + 0.0650451562356164 w_k1_IBM_s110
                                 + 0.0234393530081629 w_k1_IBM_s111
                                 + 0.0500458004687138 w_k1_IBM_s112
                                 + 0.0477375393616476 w_k1_IBM_s113
                                 + 0.025765451328939 w_k1_IBM_s114
                                 + 0.0496042567306041 w_k1_IBM_s115
                                 + 0.0157195487497995 w_k1_IBM_s116
                                 + 0.0409504735592579 w_k1_IBM_s117
                                 + 0.015345058617347 w_k1_INTC_s000
                                 + 0.0120413294106475 w_k1_INTC_s001
                                 + 0.0248391412703831 w_k1_INTC_s002
                                 + 0.0120488652296653 w_k1_INTC_s003
                                 + 0.0226069858212528 w_k1_INTC_s004
                                 + 0.0135517148349151 w_k1_INTC_s005
                                 + 0.0100073746438799 w_k1_INTC_s006
                                 + 0.0512188913538824 w_k1_INTC_s007
                                 + 0.0177789372144721 w_k1_INTC_s010
                                 + 0.0304786650601731 w_k1_INTC_s011
                                 + 0.0265532382246791 w_k1_INTC_s012
                                 + 0.0188681871965446 w_k1_INTC_s013
                                 + 0.0348666035359612 w_k1_INTC_s014
                                 + 0.0159698340741151 w_k1_INTC_s015
                                 + 0.0183498398531225 w_k1_INTC_s016
                                 + 0.0268440144785342 w_k1_INTC_s017
                                 + 0.0355771619201413 w_k1_INTC_s100
                                 + 0.0223168106441364 w_k1_INTC_s101
                                 + 0.0759807341352621 w_k1_INTC_s102
                                 + 0.0469930645430542 w_k1_INTC_s103
                                 + 0.0332246570605231 w_k1_INTC_s104
                                 + 0.0922973775731696 w_k1_INTC_s105
                                 + 0.024264765555466 w_k1_INTC_s106
                                 + 0.0359522949742577 w_k1_INTC_s107
                                 + 0.0644290527952074 w_k1_INTC_s110
                                 + 0.0226408533512918 w_k1_INTC_s111
                                 + 0.0466425212266319 w_k1_INTC_s112
                                 + 0.0425741298251261 w_k1_INTC_s113
                                 + 0.0234113192279289 w_k1_INTC_s114
                                 + 0.0484002430881265 w_k1_INTC_s115
                                 + 0.0171365792467767 w_k1_INTC_s116
                                 + 0.0388508625739468 w_k1_INTC_s117
                                 + 0.0128826775007755 w_k1_KO_s000
                                 + 0.0130398245455031 w_k1_KO_s001
                                 + 0.0213003057082586 w_k1_KO_s002
                                 + 0.0125534083116137 w_k1_KO_s003
                                 + 0.0241271165527983 w_k1_KO_s004
                                 + 0.0139337248514323 w_k1_KO_s005
                                 + 0.0105857184005137 w_k1_KO_s006
                                 + 0.0559718697456812 w_k1_KO_s007
                                 + 0.0211164662061022 w_k1_KO_s010
                                 + 0.033723586887946 w_k1_KO_s011
                                 + 0.0260770601755107 w_k1_KO_s012
                                 + 0.0165809381136773 w_k1_KO_s013
                                 + 0.0320417216801957 w_k1_KO_s014
                                 + 0.0175353603388893 w_k1_KO_s015
                                 + 0.0164388569019918 w_k1_KO_s016
                                 + 0.0263085192475171 w_k1_KO_s017
                                 + 0.0300446603277893 w_k1_KO_s100
                                 + 0.0206214636739707 w_k1_KO_s101
                                 + 0.0736948602242318 w_k1_KO_s102
                                 + 0.0477162571842941 w_k1_KO_s103
                                 + 0.0300117419944217 w_k1_KO_s104
                                 + 0.0976683882457142 w_k1_KO_s105
                                 + 0.0236215469341242 w_k1_KO_s106
                                 + 0.0305186879154128 w_k1_KO_s107
                                 + 0.0660891955447157 w_k1_KO_s110
                                 + 0.0220476049795203 w_k1_KO_s111
                                 + 0.0492136198592982 w_k1_KO_s112
                                 + 0.0429534829592011 w_k1_KO_s113
                                 + 0.0254520191805237 w_k1_KO_s114
                                 + 0.0487328852864584 w_k1_KO_s115
                                 + 0.0166143776243256 w_k1_KO_s116
                                 + 0.0412381989310008 w_k1_KO_s117
                                 + 0.0117011774774304 w_k1_MAT_s000
                                 + 0.0153733331034132 w_k1_MAT_s001
                                 + 0.0184239617143442 w_k1_MAT_s002
                                 + 0.0133442530698631 w_k1_MAT_s003
                                 + 0.0242818171080782 w_k1_MAT_s004
                                 + 0.0128323505107906 w_k1_MAT_s005
                                 + 0.00997872535594728 w_k1_MAT_s006
                                 + 0.0474967289362494 w_k1_MAT_s007
                                 + 0.0227281891462193 w_k1_MAT_s010
                                 + 0.0351220237468744 w_k1_MAT_s011
                                 + 0.0236425075227528 w_k1_MAT_s012
                                 + 0.0154358713542301 w_k1_MAT_s013
                                 + 0.0339118413768434 w_k1_MAT_s014
                                 + 0.018188013681823 w_k1_MAT_s015
                                 + 0.0203168137419261 w_k1_MAT_s016
                                 + 0.0271675154858813 w_k1_MAT_s017
                                 + 0.0338631123116549 w_k1_MAT_s100
                                 + 0.0210208985286357 w_k1_MAT_s101
                                 + 0.0703191199176525 w_k1_MAT_s102
                                 + 0.0484401740641452 w_k1_MAT_s103
                                 + 0.0302618903664636 w_k1_MAT_s104
                                 + 0.124085864033245 w_k1_MAT_s105
                                 + 0.0249142730379214 w_k1_MAT_s106
                                 + 0.0317974316618217 w_k1_MAT_s107
                                 + 0.0752350235182042 w_k1_MAT_s110
                                 + 0.020061515749019 w_k1_MAT_s111
                                 + 0.0528066739370679 w_k1_MAT_s112
                                 + 0.0455914140002421 w_k1_MAT_s113
                                 + 0.0266527661262961 w_k1_MAT_s114
                                 + 0.0451710844768399 w_k1_MAT_s115
                                 + 0.0142694093944011 w_k1_MAT_s116
                                 + 0.0451162346789396 w_k1_MAT_s117
                                 + 0.0130019970487746 w_k1_MSFT_s000
                                 + 0.0129289134829799 w_k1_MSFT_s001
                                 + 0.0224756637508159 w_k1_MSFT_s002
                                 + 0.013748327114566 w_k1_MSFT_s003
                                 + 0.0223627751760324 w_k1_MSFT_s004
                                 + 0.0138861609592857 w_k1_MSFT_s005
                                 + 0.011184758595512 w_k1_MSFT_s006
                                 + 0.0496072461458081 w_k1_MSFT_s007
                                 + 0.0165621200783743 w_k1_MSFT_s010
                                 + 0.0338009785547473 w_k1_MSFT_s011
                                 + 0.0241171177517526 w_k1_MSFT_s012
                                 + 0.0149090666831722 w_k1_MSFT_s013
                                 + 0.033259024802266 w_k1_MSFT_s014
                                 + 0.0168091764997474 w_k1_MSFT_s015
                                 + 0.0158628441405488 w_k1_MSFT_s016
                                 + 0.0257472027004921 w_k1_MSFT_s017
                                 + 0.0317655465854586 w_k1_MSFT_s100
                                 + 0.0230274939546773 w_k1_MSFT_s101
                                 + 0.0688845239065174 w_k1_MSFT_s102
                                 + 0.0451955211453771 w_k1_MSFT_s103
                                 + 0.0334942605080362 w_k1_MSFT_s104
                                 + 0.0941551077420146 w_k1_MSFT_s105
                                 + 0.0242391331331275 w_k1_MSFT_s106
                                 + 0.0346308456836671 w_k1_MSFT_s107
                                 + 0.0651516715864511 w_k1_MSFT_s110
                                 + 0.0217673023512533 w_k1_MSFT_s111
                                 + 0.0524075901951501 w_k1_MSFT_s112
                                 + 0.0452633835585934 w_k1_MSFT_s113
                                 + 0.0244253806664671 w_k1_MSFT_s114
                                 + 0.0490790421198865 w_k1_MSFT_s115
                                 + 0.0156480307434254 w_k1_MSFT_s116
                                 + 0.0409427140500368 w_k1_MSFT_s117
                                 + 0.0118324961126432 w_k1_PG_s000
                                 + 0.0134682159912358 w_k1_PG_s001
                                 + 0.0211337616368513 w_k1_PG_s002
                                 + 0.0124434675596594 w_k1_PG_s003
                                 + 0.0223939892527659 w_k1_PG_s004
                                 + 0.0154044663595548 w_k1_PG_s005
                                 + 0.0104270459738042 w_k1_PG_s006
                                 + 0.0540536581446456 w_k1_PG_s007
                                 + 0.019907683083191 w_k1_PG_s010
                                 + 0.0342815772981711 w_k1_PG_s011
                                 + 0.0249308106500053 w_k1_PG_s012
                                 + 0.0168156824467241 w_k1_PG_s013
                                 + 0.0337912586083242 w_k1_PG_s014
                                 + 0.0176833029297507 w_k1_PG_s015
                                 + 0.0164792232397675 w_k1_PG_s016
                                 + 0.024980132125174 w_k1_PG_s017
                                 + 0.0321839046971696 w_k1_PG_s100
                                 + 0.0201157150375403 w_k1_PG_s101
                                 + 0.06491007322565 w_k1_PG_s102
                                 + 0.0485228879310691 w_k1_PG_s103
                                 + 0.0326967973436678 w_k1_PG_s104
                                 + 0.0938786413757176 w_k1_PG_s105
                                 + 0.0237451185226094 w_k1_PG_s106
                                 + 0.0349294194505886 w_k1_PG_s107
                                 + 0.0637110378418629 w_k1_PG_s110
                                 + 0.0224958558733249 w_k1_PG_s111
                                 + 0.0514429675095981 w_k1_PG_s112
                                 + 0.0442972736833792 w_k1_PG_s113
                                 + 0.0261663051837153 w_k1_PG_s114
                                 + 0.0447835712998968 w_k1_PG_s115
                                 + 0.0163570743138738 w_k1_PG_s116
                                 + 0.0423779835196379 w_k1_PG_s117 >= 1.1
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
