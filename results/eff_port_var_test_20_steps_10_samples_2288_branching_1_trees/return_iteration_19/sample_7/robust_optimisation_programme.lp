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
 w_k1_ADM_s0_return_constraint:  1.1637 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.924782 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.02676 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.892736 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.11098 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.899363 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.721652 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.919502 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.961602 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.19354 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.01492 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.738514 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.11087 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.31588 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.29143 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.12779 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.15925 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.769279 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.874159 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.22604 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.66668 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.095 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.685696 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.885189 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.793013 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.01444 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.15547 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.01613 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.13234 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.29434 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.48745 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.22285 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.88207 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.27498 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.947086 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.947107 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.821004 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.13715 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23753 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.890667 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.14285 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.911385 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.14829 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.775749 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.785189 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.930301 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.794371 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.20043 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.765746 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.568365 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.08116 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.55014 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.0318 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.01816 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.14454 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.712476 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.624697 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.64542 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.632783 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.889524 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.628243 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.895778 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.723953 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.0099 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.32509 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.03707 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.20327 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.76625 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.34577 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.18951 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.763602 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.19198 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.952274 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.960607 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.78227 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.34426 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.27406 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.828605 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.08443 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.839718 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.16367 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.871381 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.79554 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.916401 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.07209 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.41766 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.923961 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.616565 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.35996 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 2.00667 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.55166 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.00254 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.29766 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.719397 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.585222 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.75204 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.701942 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.957645 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.638285 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.965534 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.683284 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.03275 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.85404 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.04364 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.37339 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.71799 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.70009 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.05733 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.661067 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.21307 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.855873 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.83557 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.550382 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.38139 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17135 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.891938 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.02635 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.995263 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.04945 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.922723 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.764231 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.937959 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.929645 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.37604 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.05218 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.703103 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.1534 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.40631 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.30101 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.996047 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.1492 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.71535 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.749518 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.32458 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.698813 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.846427 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.741953 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.873694 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.713797 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.15518 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.5311 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.02307 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.28659 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.44895 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.56747 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.19252 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.768492 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.00352 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.987901 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.858752 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.726215 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.16296 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17623 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.934033 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.1501 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.661268 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.17788 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.697722 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.32652 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.942969 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.11105 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.79908 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.689766 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.706196 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.21546 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.63324 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.950027 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.963669 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.06576 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.01437 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.651244 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.56067 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.768279 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.23786 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.727516 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.39344 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.08077 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.637618 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.955781 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.895353 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.16011 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.56653 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.89937 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.88075 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.0108 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.77289 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.755858 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.22602 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.691106 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.36194 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.20154 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.936586 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.898658 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.963648 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.07421 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.948174 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.03469 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.00938 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.966344 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.3083 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.13161 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.858124 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.19958 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.28418 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.08727 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.965271 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.0495 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.849666 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.742662 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.2075 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.91771 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.783085 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.748422 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.01719 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.821913 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.01583 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.28628 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.947601 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.03013 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.30739 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.16932 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.08145 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.772486 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.07054 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.977481 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.804321 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.820152 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.07904 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18417 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.872026 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.13909 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.786346 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.19443 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.795083 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.92705 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.87802 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.0061 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.978573 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.803702 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.693383 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.14028 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.44825 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.21271 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.06485 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.06839 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.946669 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.693102 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.49026 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.749153 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.04234 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.703655 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.11105 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.847158 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.783041 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.15069 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.925415 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.09476 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.53799 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.27567 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.01959 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.84422 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.42268 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.824098 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.02287 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.754817 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.28789 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.13942 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.968429 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.34912 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.672109 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.17381 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.69008 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.09978 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.05595 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.899879 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.657541 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.608095 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.633007 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.22214 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.69035 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.45479 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.07609 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.38492 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.852121 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.730839 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 2.04807 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.652723 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.02506 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.688301 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.32752 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.918064 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.796133 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.08221 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.954814 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.20894 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.77827 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.00758 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.793043 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.876275 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.88618 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.797886 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.08979 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.536652 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.28547 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16316 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.881504 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.01193 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.955809 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.12068 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.937719 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.913968 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.94812 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.987888 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.27998 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.01129 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.748103 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.16174 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.36021 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.3536 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.976027 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.07225 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.762789 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.788764 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.29793 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.777847 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.849687 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.70948 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.942075 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.796921 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.0459 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.46152 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.962177 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.12424 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.38612 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.54292 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.15987 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.776037 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.08111 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.05661 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.918433 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.773701 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.23765 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06849 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.905168 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.09461 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.944881 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.10161 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.893492 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.964544 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.959454 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.939485 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.0444 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.905713 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.764961 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.07003 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.30925 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.13476 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.990625 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.03112 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.905252 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.782896 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.21058 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.852054 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.00465 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.796253 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.0104 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.87307 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.0463 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.23314 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.99341 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.13817 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.39506 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.27965 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.08969 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.935449 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.20492 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.891944 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.981305 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.805834 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.21831 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 1.02300336018435 z_k1_s0000
                                 - 0.687999261198623 z_k1_s0001
                                 - 0.306000533692553 z_k1_s0002
                                 - 0.687999261198623 z_k1_s0003
                                 - 0.573001085163636 z_k1_s0004
                                 - 0.68100273661028 z_k1_s0005
                                 - 0.280000545544906 z_k1_s0006
                                 - 0.3670006476084 z_k1_s0007
                                 - 0.418000583892435 z_k1_s0010
                                 - 1.40400136768093 z_k1_s0011
                                 - 1.12299907445989 z_k1_s0012
                                 - 0.421000753422758 z_k1_s0013
                                 - 0.595999419375738 z_k1_s0014
                                 - 0.846001732885945 z_k1_s0015
                                 - 0.643999769057565 z_k1_s0016
                                 - 0.455000902663913 z_k1_s0017
                                 - 0.095000037996021 z_k1_s0020
                                 - 0.192000207303518 z_k1_s0021
                                 - 0.0850000409385441 z_k1_s0022
                                 - 0.0750000438810672 z_k1_s0023
                                 - 0.159000151063854 z_k1_s0024
                                 - 0.213999805130025 z_k1_s0025
                                 - 0.275999391186438 z_k1_s0026
                                 - 0.22299934083182 z_k1_s0027
                                 - 0.511999393712136 z_k1_s0030
                                 - 0.208000042295502 z_k1_s0031
                                 - 0.278999958220427 z_k1_s0032
                                 - 0.387999540547478 z_k1_s0033
                                 - 0.165000093214209 z_k1_s0034
                                 - 0.224000023349006 z_k1_s0035
                                 - 0.285999949931335 z_k1_s0036
                                 - 0.823999024255468 z_k1_s0037
                                 - 0.439998642135491 z_k1_s0040
                                 - 0.183999367565945 z_k1_s0041
                                 - 0.585998078521699 z_k1_s0042
                                 - 0.228999398267928 z_k1_s0043
                                 - 0.734996237878557 z_k1_s0044
                                 - 0.631998631011045 z_k1_s0045
                                 - 0.426998254280504 z_k1_s0046
                                 - 0.319998915562467 z_k1_s0047
                                 - 0.45000036631873 z_k1_s0050
                                 - 0.653002202943092 z_k1_s0051
                                 - 0.724002834347663 z_k1_s0052
                                 - 0.301000287058121 z_k1_s0053
                                 - 1.20900228962813 z_k1_s0054
                                 - 0.748001637440926 z_k1_s0055
                                 - 0.587002712934174 z_k1_s0056
                                 - 0.890998678125286 z_k1_s0057
                                 - 1.04599914294065 z_k1_s0060
                                 - 0.266999626077914 z_k1_s0061
                                 - 0.604996237673363 z_k1_s0062
                                 - 1.03899979777619 z_k1_s0063
                                 - 0.928997470932874 z_k1_s0064
                                 - 0.744000583045126 z_k1_s0065
                                 - 0.362999398600508 z_k1_s0066
                                 - 0.52699880299371 z_k1_s0067
                                 - 0.43799869986625 z_k1_s0070
                                 - 0.513001340699544 z_k1_s0071
                                 - 0.202999913255368 z_k1_s0072
                                 - 0.558001146199501 z_k1_s0073
                                 - 0.271999911521972 z_k1_s0074
                                 - 0.192000059633158 z_k1_s0075
                                 - 0.182000004022055 z_k1_s0076
                                 - 0.607000736743896 z_k1_s0077
                                 - 0.6000012780424 z_k1_s0100
                                 - 0.165000515037106 z_k1_s0101
                                 - 0.263000963694685 z_k1_s0102
                                 - 0.290001817273762 z_k1_s0103
                                 - 0.192000496213806 z_k1_s0104
                                 - 0.236000110115608 z_k1_s0105
                                 - 0.124000438609238 z_k1_s0106
                                 - 0.31100054249665 z_k1_s0107
                                 - 0.394001547840057 z_k1_s0110
                                 - 0.406999602787413 z_k1_s0111
                                 - 0.240000103935369 z_k1_s0112
                                 - 0.280000209824672 z_k1_s0113
                                 - 0.248000337673408 z_k1_s0114
                                 - 0.245000316446699 z_k1_s0115
                                 - 0.387999379784849 z_k1_s0116
                                 - 0.455001005215651 z_k1_s0117
                                 - 0.217000002998778 z_k1_s0120
                                 - 0.101999794083631 z_k1_s0121
                                 - 0.256999103483876 z_k1_s0122
                                 - 0.171999499580488 z_k1_s0123
                                 - 0.124999781986747 z_k1_s0124
                                 - 0.148000039289429 z_k1_s0125
                                 - 0.260999687031299 z_k1_s0126
                                 - 0.0649999189610565 z_k1_s0127
                                 - 0.356998743845237 z_k1_s0130
                                 - 0.0869997118681134 z_k1_s0131
                                 - 0.0849998092294955 z_k1_s0132
                                 - 0.163000026123629 z_k1_s0133
                                 - 0.200999112854577 z_k1_s0134
                                 - 0.0859996936490038 z_k1_s0135
                                 - 0.238999537581537 z_k1_s0136
                                 - 0.119999644100722 z_k1_s0137
                                 - 0.262999535880275 z_k1_s0140
                                 - 0.147000413225826 z_k1_s0141
                                 - 0.107000039269151 z_k1_s0142
                                 - 0.185999805428629 z_k1_s0143
                                 - 0.0849999866829723 z_k1_s0144
                                 - 0.110000036130903 z_k1_s0145
                                 - 0.117000444608305 z_k1_s0146
                                 - 0.118999686516175 z_k1_s0147
                                 - 0.168000179561808 z_k1_s0150
                                 - 0.108000057832529 z_k1_s0151
                                 - 0.204000602039749 z_k1_s0152
                                 - 0.158000808874289 z_k1_s0153
                                 - 0.212000098589764 z_k1_s0154
                                 - 0.153999716597845 z_k1_s0155
                                 - 0.149000031254086 z_k1_s0156
                                 - 0.191000076144538 z_k1_s0157
                                 - 0.163000195792684 z_k1_s0160
                                 - 0.176999620431335 z_k1_s0161
                                 - 0.115999947005842 z_k1_s0162
                                 - 0.0750000055641208 z_k1_s0163
                                 - 0.0690000282189905 z_k1_s0164
                                 - 0.0870003727543706 z_k1_s0165
                                 - 0.0790000179608736 z_k1_s0166
                                 - 0.0590000384771075 z_k1_s0167
                                 - 0.179000070008647 z_k1_s0170
                                 - 0.0980001489729722 z_k1_s0171
                                 - 0.218999537507382 z_k1_s0172
                                 - 0.380000392242 z_k1_s0173
                                 - 0.167000048058893 z_k1_s0174
                                 - 0.39600090604203 z_k1_s0175
                                 - 0.200000971496354 z_k1_s0176
                                 - 0.178000174171246 z_k1_s0177
                                 - 0.702001022815125 z_k1_s1000
                                 - 0.743001061113405 z_k1_s1001
                                 - 1.39000551975436 z_k1_s1002
                                 - 1.22000040545965 z_k1_s1003
                                 - 1.23300362949918 z_k1_s1004
                                 - 0.383000633067365 z_k1_s1005
                                 - 0.631001011554791 z_k1_s1006
                                 - 1.22300288601044 z_k1_s1007
                                 - 0.558998345761715 z_k1_s1010
                                 - 0.41699864182408 z_k1_s1011
                                 - 0.352998749536778 z_k1_s1012
                                 - 0.785994777041081 z_k1_s1013
                                 - 1.17299550502388 z_k1_s1014
                                 - 0.884997511726084 z_k1_s1015
                                 - 1.22099314124677 z_k1_s1016
                                 - 0.693999307613883 z_k1_s1017
                                 - 0.598001424104974 z_k1_s1020
                                 - 0.192000375774936 z_k1_s1021
                                 - 0.618002419907259 z_k1_s1022
                                 - 0.170000478295944 z_k1_s1023
                                 - 0.780002899183463 z_k1_s1024
                                 - 0.569000179842247 z_k1_s1025
                                 - 0.24000076919455 z_k1_s1026
                                 - 0.826002394420501 z_k1_s1027
                                 - 0.575001204714469 z_k1_s1030
                                 - 0.847002911850082 z_k1_s1031
                                 - 0.675002262926514 z_k1_s1032
                                 - 0.416000400558179 z_k1_s1033
                                 - 0.556998867034192 z_k1_s1034
                                 - 0.395000275933746 z_k1_s1035
                                 - 1.06300168398465 z_k1_s1036
                                 - 0.352000113702854 z_k1_s1037
                                 - 0.262999139424031 z_k1_s1040
                                 - 0.30600058767141 z_k1_s1041
                                 - 0.265000174597928 z_k1_s1042
                                 - 0.161999923520318 z_k1_s1043
                                 - 0.256000718313243 z_k1_s1044
                                 - 0.11699986769804 z_k1_s1045
                                 - 0.162999920907482 z_k1_s1046
                                 - 0.202000512606566 z_k1_s1047
                                 - 1.24000072106934 z_k1_s1050
                                 - 1.41300364595081 z_k1_s1051
                                 - 0.986000944725128 z_k1_s1052
                                 - 0.627998254305675 z_k1_s1053
                                 - 0.372000822221844 z_k1_s1054
                                 - 0.389000587792536 z_k1_s1055
                                 - 0.647999047743623 z_k1_s1056
                                 - 0.383000955662194 z_k1_s1057
                                 - 0.295000011486746 z_k1_s1060
                                 - 0.822000126113632 z_k1_s1061
                                 - 0.584002528761033 z_k1_s1062
                                 - 0.348000154710212 z_k1_s1063
                                 - 1.25300126171765 z_k1_s1064
                                 - 0.260000062405299 z_k1_s1065
                                 - 1.12600306222309 z_k1_s1066
                                 - 0.453000516054862 z_k1_s1067
                                 - 0.360000971027436 z_k1_s1070
                                 - 0.550998199755878 z_k1_s1071
                                 - 0.643998056229965 z_k1_s1072
                                 - 0.710000600499663 z_k1_s1073
                                 - 0.235999795229807 z_k1_s1074
                                 - 0.212999864650165 z_k1_s1075
                                 - 0.279999703577782 z_k1_s1076
                                 - 0.160999829557225 z_k1_s1077
                                 - 0.130000341736113 z_k1_s1100
                                 - 0.182000367130325 z_k1_s1101
                                 - 0.20899957766987 z_k1_s1102
                                 - 0.194000016259012 z_k1_s1103
                                 - 0.201000800919484 z_k1_s1104
                                 - 0.238000708401673 z_k1_s1105
                                 - 0.407000837307959 z_k1_s1106
                                 - 0.294001420673797 z_k1_s1107
                                 - 0.34900101008319 z_k1_s1110
                                 - 0.446001548700641 z_k1_s1111
                                 - 0.193000827857293 z_k1_s1112
                                 - 0.202000979245984 z_k1_s1113
                                 - 0.166000792293176 z_k1_s1114
                                 - 0.177000814534149 z_k1_s1115
                                 - 0.205000890174895 z_k1_s1116
                                 - 0.355001669144928 z_k1_s1117
                                 - 0.33500058828424 z_k1_s1120
                                 - 0.601001240349201 z_k1_s1121
                                 - 0.249000287879557 z_k1_s1122
                                 - 0.145000428910029 z_k1_s1123
                                 - 0.383001476395619 z_k1_s1124
                                 - 0.16500052093908 z_k1_s1125
                                 - 0.299000517952185 z_k1_s1126
                                 - 0.206000614278398 z_k1_s1127
                                 - 0.0859999165956754 z_k1_s1130
                                 - 0.0920000507767733 z_k1_s1131
                                 - 0.0770000163240377 z_k1_s1132
                                 - 0.0530000213996626 z_k1_s1133
                                 - 0.102999795108771 z_k1_s1134
                                 - 0.0709998821429399 z_k1_s1135
                                 - 0.0500000145091155 z_k1_s1136
                                 - 0.0699999601127599 z_k1_s1137
                                 - 0.244000311031314 z_k1_s1140
                                 - 0.116999667115237 z_k1_s1141
                                 - 0.170999705291086 z_k1_s1142
                                 - 0.387999166036271 z_k1_s1143
                                 - 0.0929998579477441 z_k1_s1144
                                 - 0.378999367473181 z_k1_s1145
                                 - 0.27499984808146 z_k1_s1146
                                 - 0.410998420364667 z_k1_s1147
                                 - 0.196000267364288 z_k1_s1150
                                 - 0.384001139416372 z_k1_s1151
                                 - 0.300001138231792 z_k1_s1152
                                 - 0.215000154232114 z_k1_s1153
                                 - 0.464000384543777 z_k1_s1154
                                 - 0.390999816513746 z_k1_s1155
                                 - 0.441001329220386 z_k1_s1156
                                 - 0.255000305996353 z_k1_s1157
                                 - 0.297001522257669 z_k1_s1160
                                 - 0.438002890484011 z_k1_s1161
                                 - 0.387001611121896 z_k1_s1162
                                 - 0.123000613050065 z_k1_s1163
                                 - 0.456002908256856 z_k1_s1164
                                 - 0.340001974383595 z_k1_s1165
                                 - 0.248001064207461 z_k1_s1166
                                 - 0.169000822336988 z_k1_s1167
                                 - 0.348000247513687 z_k1_s1170
                                 - 0.151000470394437 z_k1_s1171
                                 - 0.444000801587572 z_k1_s1172
                                 - 0.376000213451443 z_k1_s1173
                                 - 0.138000268179966 z_k1_s1174
                                 - 0.411999834227768 z_k1_s1175
                                 - 0.374999963095931 z_k1_s1176
                                 - 0.104000208912454 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.24713 w_k1_ADM_s000
                                 + 1.16609 w_k1_AXP_s000 + 1.0383 w_k1_DIS_s000
                                 + 1.09424 w_k1_FDX_s000
                                 + 0.785972 w_k1_IBM_s000
                                 + 0.902776 w_k1_INTC_s000
                                 + 0.840986 w_k1_KO_s000
                                 + 0.791064 w_k1_MAT_s000
                                 + 0.887726 w_k1_MSFT_s000
                                 + 0.938136 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.17211 w_k1_ADM_s000
                                 + 1.3355 w_k1_AXP_s000 + 1.44727 w_k1_DIS_s000
                                 + 1.31825 w_k1_FDX_s000
                                 + 1.12696 w_k1_IBM_s000
                                 + 0.999947 w_k1_INTC_s000
                                 + 1.23903 w_k1_KO_s000 + 1.42572 w_k1_MAT_s000
                                 + 1.06802 w_k1_MSFT_s000
                                 + 1.28307 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.03922 w_k1_ADM_s000
                                 + 1.18215 w_k1_AXP_s000
                                 + 1.03906 w_k1_DIS_s000 + 1.0359 w_k1_FDX_s000
                                 + 1.735 w_k1_IBM_s000 + 1.02849 w_k1_INTC_s000
                                 + 1.40812 w_k1_KO_s000 + 1.99697 w_k1_MAT_s000
                                 + 0.872996 w_k1_MSFT_s000
                                 + 1.06032 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.06485 w_k1_ADM_s000
                                 + 0.946176 w_k1_AXP_s000
                                 + 1.04661 w_k1_DIS_s000
                                 + 0.966758 w_k1_FDX_s000
                                 + 1.19257 w_k1_IBM_s000
                                 + 0.862795 w_k1_INTC_s000
                                 + 1.19731 w_k1_KO_s000 + 1.16798 w_k1_MAT_s000
                                 + 0.892447 w_k1_MSFT_s000
                                 + 1.02461 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.14263 w_k1_ADM_s000
                                 + 0.779356 w_k1_AXP_s000
                                 + 0.900657 w_k1_DIS_s000
                                 + 0.898336 w_k1_FDX_s000
                                 + 0.900095 w_k1_IBM_s000
                                 + 0.838291 w_k1_INTC_s000
                                 + 1.03312 w_k1_KO_s000
                                 + 0.705903 w_k1_MAT_s000
                                 + 0.793788 w_k1_MSFT_s000
                                 + 0.974609 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.959692 w_k1_ADM_s000
                                 + 0.641758 w_k1_AXP_s000
                                 + 0.635442 w_k1_DIS_s000
                                 + 0.773097 w_k1_FDX_s000
                                 + 0.689208 w_k1_IBM_s000
                                 + 0.860006 w_k1_INTC_s000
                                 + 0.712658 w_k1_KO_s000
                                 + 0.645624 w_k1_MAT_s000
                                 + 0.766838 w_k1_MSFT_s000
                                 + 0.673907 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.884933 w_k1_ADM_s000
                                 + 0.747393 w_k1_AXP_s000
                                 + 0.585227 w_k1_DIS_s000
                                 + 0.757577 w_k1_FDX_s000
                                 + 1.11425 w_k1_IBM_s000
                                 + 0.673202 w_k1_INTC_s000
                                 + 0.845287 w_k1_KO_s000
                                 + 0.935924 w_k1_MAT_s000
                                 + 0.654998 w_k1_MSFT_s000
                                 + 0.720457 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.20692 w_k1_ADM_s000
                                 + 1.72675 w_k1_AXP_s000
                                 + 1.71384 w_k1_DIS_s000
                                 + 1.33012 w_k1_FDX_s000
                                 + 1.58609 w_k1_IBM_s000
                                 + 1.2884 w_k1_INTC_s000 + 1.63763 w_k1_KO_s000
                                 + 2.00452 w_k1_MAT_s000
                                 + 1.27944 w_k1_MSFT_s000
                                 + 1.21051 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.993988 w_k1_ADM_s001
                                 + 1.02711 w_k1_AXP_s001
                                 + 0.795845 w_k1_DIS_s001
                                 + 1.01032 w_k1_FDX_s001
                                 + 0.803997 w_k1_IBM_s001
                                 + 0.976195 w_k1_INTC_s001
                                 + 0.864315 w_k1_KO_s001 + 0.7665 w_k1_MAT_s001
                                 + 0.994904 w_k1_MSFT_s001
                                 + 0.875929 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.976673 w_k1_ADM_s001
                                 + 1.19099 w_k1_AXP_s001
                                 + 1.10064 w_k1_DIS_s001 + 1.1053 w_k1_FDX_s001
                                 + 0.974041 w_k1_IBM_s001
                                 + 1.25867 w_k1_INTC_s001
                                 + 1.03868 w_k1_KO_s001
                                 + 0.843545 w_k1_MAT_s001
                                 + 1.01795 w_k1_MSFT_s001
                                 + 1.03165 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.900217 w_k1_ADM_s001
                                 + 0.779494 w_k1_AXP_s001
                                 + 0.680562 w_k1_DIS_s001
                                 + 0.846965 w_k1_FDX_s001
                                 + 0.72892 w_k1_IBM_s001
                                 + 0.771597 w_k1_INTC_s001
                                 + 0.738616 w_k1_KO_s001
                                 + 0.652047 w_k1_MAT_s001
                                 + 0.785711 w_k1_MSFT_s001
                                 + 0.838702 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.32119 w_k1_ADM_s001
                                 + 1.32188 w_k1_AXP_s001
                                 + 0.971266 w_k1_DIS_s001
                                 + 0.979479 w_k1_FDX_s001
                                 + 1.41723 w_k1_IBM_s001
                                 + 0.830553 w_k1_INTC_s001
                                 + 1.11765 w_k1_KO_s001 + 1.4108 w_k1_MAT_s001
                                 + 0.885535 w_k1_MSFT_s001
                                 + 1.14147 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.844358 w_k1_ADM_s001
                                 + 0.855503 w_k1_AXP_s001
                                 + 0.646156 w_k1_DIS_s001
                                 + 0.838539 w_k1_FDX_s001
                                 + 1.04925 w_k1_IBM_s001
                                 + 0.67519 w_k1_INTC_s001
                                 + 0.774779 w_k1_KO_s001
                                 + 0.86854 w_k1_MAT_s001
                                 + 0.712449 w_k1_MSFT_s001
                                 + 0.901319 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.04919 w_k1_ADM_s001
                                 + 0.85519 w_k1_AXP_s001
                                 + 1.00431 w_k1_DIS_s001
                                 + 0.976483 w_k1_FDX_s001
                                 + 1.08557 w_k1_IBM_s001
                                 + 1.11111 w_k1_INTC_s001
                                 + 0.959593 w_k1_KO_s001
                                 + 0.735325 w_k1_MAT_s001
                                 + 0.963587 w_k1_MSFT_s001
                                 + 0.970063 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.19093 w_k1_ADM_s001
                                 + 1.42063 w_k1_AXP_s001
                                 + 1.61337 w_k1_DIS_s001
                                 + 1.40883 w_k1_FDX_s001
                                 + 1.10592 w_k1_IBM_s001
                                 + 1.2463 w_k1_INTC_s001 + 1.13111 w_k1_KO_s001
                                 + 1.01275 w_k1_MAT_s001
                                 + 1.3236 w_k1_MSFT_s001 + 1.19362 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.19198 w_k1_ADM_s001
                                 + 1.2759 w_k1_AXP_s001 + 1.38159 w_k1_DIS_s001
                                 + 1.1915 w_k1_FDX_s001 + 1.59196 w_k1_IBM_s001
                                 + 1.1636 w_k1_INTC_s001 + 1.38964 w_k1_KO_s001
                                 + 1.27085 w_k1_MAT_s001
                                 + 1.11733 w_k1_MSFT_s001
                                 + 1.27467 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.706289 w_k1_ADM_s002
                                 + 0.689895 w_k1_AXP_s002
                                 + 0.679147 w_k1_DIS_s002
                                 + 0.822233 w_k1_FDX_s002
                                 + 0.60963 w_k1_IBM_s002
                                 + 0.868396 w_k1_INTC_s002
                                 + 0.68354 w_k1_KO_s002
                                 + 0.541008 w_k1_MAT_s002
                                 + 0.804037 w_k1_MSFT_s002
                                 + 0.717161 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.10647 w_k1_ADM_s002
                                 + 1.01702 w_k1_AXP_s002
                                 + 1.30089 w_k1_DIS_s002
                                 + 1.07036 w_k1_FDX_s002
                                 + 1.33228 w_k1_IBM_s002
                                 + 1.0691 w_k1_INTC_s002 + 1.18511 w_k1_KO_s002
                                 + 1.04888 w_k1_MAT_s002
                                 + 1.07772 w_k1_MSFT_s002
                                 + 1.12459 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.896292 w_k1_ADM_s002
                                 + 0.79269 w_k1_AXP_s002
                                 + 1.01032 w_k1_DIS_s002
                                 + 0.968558 w_k1_FDX_s002
                                 + 0.73814 w_k1_IBM_s002
                                 + 0.862604 w_k1_INTC_s002
                                 + 0.942002 w_k1_KO_s002
                                 + 0.749291 w_k1_MAT_s002
                                 + 1.0057 w_k1_MSFT_s002
                                 + 0.991103 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.929722 w_k1_ADM_s002
                                 + 1.14241 w_k1_AXP_s002 + 1.5017 w_k1_DIS_s002
                                 + 1.14141 w_k1_FDX_s002
                                 + 1.58605 w_k1_IBM_s002
                                 + 1.2954 w_k1_INTC_s002 + 1.61274 w_k1_KO_s002
                                 + 1.47311 w_k1_MAT_s002
                                 + 1.17211 w_k1_MSFT_s002 + 1.243 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.613568 w_k1_ADM_s002
                                 + 0.673667 w_k1_AXP_s002
                                 + 0.720574 w_k1_DIS_s002
                                 + 0.829471 w_k1_FDX_s002
                                 + 0.814502 w_k1_IBM_s002
                                 + 0.997182 w_k1_INTC_s002
                                 + 0.788067 w_k1_KO_s002
                                 + 0.973357 w_k1_MAT_s002
                                 + 0.847923 w_k1_MSFT_s002
                                 + 0.872405 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.81623 w_k1_ADM_s002
                                 + 0.928481 w_k1_AXP_s002
                                 + 0.810799 w_k1_DIS_s002
                                 + 0.847825 w_k1_FDX_s002
                                 + 1.1066 w_k1_IBM_s002
                                 + 0.989279 w_k1_INTC_s002
                                 + 0.944607 w_k1_KO_s002
                                 + 1.21503 w_k1_MAT_s002
                                 + 0.923583 w_k1_MSFT_s002
                                 + 0.950255 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.89607 w_k1_ADM_s002
                                 + 1.21679 w_k1_AXP_s002
                                 + 1.17714 w_k1_DIS_s002
                                 + 1.19795 w_k1_FDX_s002
                                 + 0.963175 w_k1_IBM_s002
                                 + 1.20536 w_k1_INTC_s002
                                 + 0.990694 w_k1_KO_s002
                                 + 0.748355 w_k1_MAT_s002
                                 + 1.04761 w_k1_MSFT_s002
                                 + 1.04459 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.00504 w_k1_ADM_s002
                                 + 0.967168 w_k1_AXP_s002
                                 + 1.15975 w_k1_DIS_s002
                                 + 1.00726 w_k1_FDX_s002
                                 + 0.983566 w_k1_IBM_s002
                                 + 1.14787 w_k1_INTC_s002 + 1.1307 w_k1_KO_s002
                                 + 0.659898 w_k1_MAT_s002
                                 + 0.988176 w_k1_MSFT_s002
                                 + 0.98593 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.945128 w_k1_ADM_s003
                                 + 0.798573 w_k1_AXP_s003
                                 + 0.668992 w_k1_DIS_s003
                                 + 0.795416 w_k1_FDX_s003
                                 + 0.728932 w_k1_IBM_s003
                                 + 0.660985 w_k1_INTC_s003
                                 + 0.817746 w_k1_KO_s003
                                 + 0.891067 w_k1_MAT_s003
                                 + 0.714339 w_k1_MSFT_s003
                                 + 0.930103 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.06559 w_k1_ADM_s003
                                 + 0.869025 w_k1_AXP_s003
                                 + 0.95669 w_k1_DIS_s003
                                 + 0.932479 w_k1_FDX_s003
                                 + 0.75084 w_k1_IBM_s003
                                 + 0.804571 w_k1_INTC_s003
                                 + 0.82159 w_k1_KO_s003 + 1.16498 w_k1_MAT_s003
                                 + 0.960372 w_k1_MSFT_s003
                                 + 0.955579 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.17319 w_k1_ADM_s003
                                 + 1.34028 w_k1_AXP_s003 + 1.233 w_k1_DIS_s003
                                 + 1.30804 w_k1_FDX_s003
                                 + 1.02242 w_k1_IBM_s003
                                 + 1.40053 w_k1_INTC_s003
                                 + 1.27407 w_k1_KO_s003 + 1.6047 w_k1_MAT_s003
                                 + 1.20929 w_k1_MSFT_s003
                                 + 1.30611 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.22024 w_k1_ADM_s003
                                 + 1.12816 w_k1_AXP_s003 + 1.0869 w_k1_DIS_s003
                                 + 1.10569 w_k1_FDX_s003
                                 + 0.722565 w_k1_IBM_s003
                                 + 0.94129 w_k1_INTC_s003
                                 + 1.23928 w_k1_KO_s003 + 1.2008 w_k1_MAT_s003
                                 + 1.12321 w_k1_MSFT_s003
                                 + 1.13741 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.882101 w_k1_ADM_s003
                                 + 0.625909 w_k1_AXP_s003
                                 + 1.08789 w_k1_DIS_s003
                                 + 0.926098 w_k1_FDX_s003
                                 + 0.955166 w_k1_IBM_s003
                                 + 0.898164 w_k1_INTC_s003
                                 + 1.07723 w_k1_KO_s003
                                 + 0.913686 w_k1_MAT_s003
                                 + 1.01221 w_k1_MSFT_s003
                                 + 0.978084 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.858522 w_k1_ADM_s003
                                 + 1.27798 w_k1_AXP_s003
                                 + 0.853842 w_k1_DIS_s003
                                 + 0.958332 w_k1_FDX_s003
                                 + 1.30359 w_k1_IBM_s003
                                 + 0.829865 w_k1_INTC_s003
                                 + 1.10486 w_k1_KO_s003 + 2.24128 w_k1_MAT_s003
                                 + 0.863009 w_k1_MSFT_s003
                                 + 1.11579 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.949194 w_k1_ADM_s003
                                 + 0.967374 w_k1_AXP_s003
                                 + 0.760471 w_k1_DIS_s003
                                 + 0.823818 w_k1_FDX_s003
                                 + 1.13533 w_k1_IBM_s003
                                 + 0.96682 w_k1_INTC_s003
                                 + 0.964652 w_k1_KO_s003
                                 + 1.43522 w_k1_MAT_s003
                                 + 0.770863 w_k1_MSFT_s003
                                 + 1.04661 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.37403 w_k1_ADM_s003
                                 + 1.30301 w_k1_AXP_s003
                                 + 1.41138 w_k1_DIS_s003
                                 + 1.20389 w_k1_FDX_s003 + 1.3083 w_k1_IBM_s003
                                 + 1.26826 w_k1_INTC_s003
                                 + 1.37704 w_k1_KO_s003 + 1.25866 w_k1_MAT_s003
                                 + 1.2279 w_k1_MSFT_s003 + 1.14033 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.99102 w_k1_ADM_s004
                                 + 1.30144 w_k1_AXP_s004 + 1.1629 w_k1_DIS_s004
                                 + 1.02015 w_k1_FDX_s004
                                 + 1.32906 w_k1_IBM_s004
                                 + 0.958487 w_k1_INTC_s004
                                 + 1.14356 w_k1_KO_s004 + 1.20192 w_k1_MAT_s004
                                 + 1.03077 w_k1_MSFT_s004
                                 + 0.964928 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.766644 w_k1_ADM_s004
                                 + 0.887931 w_k1_AXP_s004
                                 + 1.31065 w_k1_DIS_s004
                                 + 0.918661 w_k1_FDX_s004
                                 + 1.25663 w_k1_IBM_s004
                                 + 0.965222 w_k1_INTC_s004
                                 + 1.28543 w_k1_KO_s004 + 1.46397 w_k1_MAT_s004
                                 + 0.989574 w_k1_MSFT_s004
                                 + 1.06682 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.04798 w_k1_ADM_s004
                                 + 1.25241 w_k1_AXP_s004
                                 + 1.21234 w_k1_DIS_s004
                                 + 1.12424 w_k1_FDX_s004
                                 + 1.00964 w_k1_IBM_s004
                                 + 0.940678 w_k1_INTC_s004
                                 + 1.18897 w_k1_KO_s004 + 1.05215 w_k1_MAT_s004
                                 + 1.18992 w_k1_MSFT_s004
                                 + 1.10847 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.920222 w_k1_ADM_s004
                                 + 1.15177 w_k1_AXP_s004
                                 + 1.71447 w_k1_DIS_s004
                                 + 1.13243 w_k1_FDX_s004
                                 + 1.21427 w_k1_IBM_s004
                                 + 0.986135 w_k1_INTC_s004
                                 + 1.43976 w_k1_KO_s004 + 1.18828 w_k1_MAT_s004
                                 + 1.259 w_k1_MSFT_s004 + 1.18221 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.02828 w_k1_ADM_s004
                                 + 0.978074 w_k1_AXP_s004
                                 + 1.00494 w_k1_DIS_s004
                                 + 0.974811 w_k1_FDX_s004
                                 + 0.74692 w_k1_IBM_s004
                                 + 0.966771 w_k1_INTC_s004
                                 + 0.751447 w_k1_KO_s004
                                 + 0.631066 w_k1_MAT_s004
                                 + 0.994639 w_k1_MSFT_s004
                                 + 0.982096 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.840152 w_k1_ADM_s004
                                 + 0.901111 w_k1_AXP_s004
                                 + 0.829764 w_k1_DIS_s004
                                 + 0.826907 w_k1_FDX_s004
                                 + 1.08771 w_k1_IBM_s004
                                 + 0.92622 w_k1_INTC_s004
                                 + 1.00227 w_k1_KO_s004
                                 + 0.889149 w_k1_MAT_s004
                                 + 0.950505 w_k1_MSFT_s004
                                 + 0.999661 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.827908 w_k1_ADM_s004
                                 + 0.816227 w_k1_AXP_s004
                                 + 0.662831 w_k1_DIS_s004
                                 + 0.760339 w_k1_FDX_s004
                                 + 0.720522 w_k1_IBM_s004
                                 + 0.852853 w_k1_INTC_s004
                                 + 0.747862 w_k1_KO_s004
                                 + 0.77964 w_k1_MAT_s004
                                 + 0.811241 w_k1_MSFT_s004
                                 + 0.915074 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.05417 w_k1_ADM_s004
                                 + 1.88232 w_k1_AXP_s004
                                 + 1.76739 w_k1_DIS_s004
                                 + 1.36003 w_k1_FDX_s004
                                 + 1.25902 w_k1_IBM_s004
                                 + 1.29165 w_k1_INTC_s004
                                 + 1.42384 w_k1_KO_s004 + 1.81448 w_k1_MAT_s004
                                 + 1.36982 w_k1_MSFT_s004
                                 + 1.21817 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.748985 w_k1_ADM_s005
                                 + 1.0162 w_k1_AXP_s005
                                 + 0.723811 w_k1_DIS_s005
                                 + 0.807352 w_k1_FDX_s005
                                 + 1.2844 w_k1_IBM_s005
                                 + 0.801601 w_k1_INTC_s005
                                 + 0.923401 w_k1_KO_s005
                                 + 1.33386 w_k1_MAT_s005
                                 + 0.789323 w_k1_MSFT_s005
                                 + 1.00211 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.26019 w_k1_ADM_s005
                                 + 0.980316 w_k1_AXP_s005
                                 + 0.965916 w_k1_DIS_s005
                                 + 1.20068 w_k1_FDX_s005
                                 + 0.676565 w_k1_IBM_s005
                                 + 0.951543 w_k1_INTC_s005
                                 + 0.826568 w_k1_KO_s005
                                 + 0.469708 w_k1_MAT_s005
                                 + 1.04681 w_k1_MSFT_s005
                                 + 0.969797 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.673488 w_k1_ADM_s005
                                 + 0.693447 w_k1_AXP_s005
                                 + 0.661936 w_k1_DIS_s005
                                 + 0.749139 w_k1_FDX_s005
                                 + 0.92489 w_k1_IBM_s005
                                 + 0.82093 w_k1_INTC_s005
                                 + 0.77325 w_k1_KO_s005
                                 + 0.694682 w_k1_MAT_s005
                                 + 0.814852 w_k1_MSFT_s005
                                 + 0.864932 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.74762 w_k1_ADM_s005
                                 + 0.773321 w_k1_AXP_s005
                                 + 0.525219 w_k1_DIS_s005
                                 + 0.798141 w_k1_FDX_s005
                                 + 0.560688 w_k1_IBM_s005
                                 + 0.766083 w_k1_INTC_s005
                                 + 0.561202 w_k1_KO_s005
                                 + 0.500152 w_k1_MAT_s005
                                 + 0.695909 w_k1_MSFT_s005
                                 + 0.790764 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.97406 w_k1_ADM_s005
                                 + 1.30509 w_k1_AXP_s005
                                 + 1.10369 w_k1_DIS_s005
                                 + 1.09331 w_k1_FDX_s005 + 1.1799 w_k1_IBM_s005
                                 + 1.03176 w_k1_INTC_s005
                                 + 1.10598 w_k1_KO_s005 + 1.25567 w_k1_MAT_s005
                                 + 1.08025 w_k1_MSFT_s005
                                 + 0.994164 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.11963 w_k1_ADM_s005
                                 + 1.08954 w_k1_AXP_s005
                                 + 0.96033 w_k1_DIS_s005
                                 + 0.957966 w_k1_FDX_s005
                                 + 1.48371 w_k1_IBM_s005
                                 + 1.14166 w_k1_INTC_s005
                                 + 1.17989 w_k1_KO_s005 + 1.21302 w_k1_MAT_s005
                                 + 0.974893 w_k1_MSFT_s005
                                 + 1.02772 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.3102 w_k1_ADM_s005
                                 + 1.63813 w_k1_AXP_s005 + 1.5163 w_k1_DIS_s005
                                 + 1.42088 w_k1_FDX_s005
                                 + 1.18539 w_k1_IBM_s005
                                 + 1.32463 w_k1_INTC_s005
                                 + 1.24655 w_k1_KO_s005 + 1.03927 w_k1_MAT_s005
                                 + 1.36809 w_k1_MSFT_s005
                                 + 1.14606 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.04124 w_k1_ADM_s005
                                 + 1.05926 w_k1_AXP_s005 + 1.0418 w_k1_DIS_s005
                                 + 1.06567 w_k1_FDX_s005
                                 + 0.831313 w_k1_IBM_s005
                                 + 1.00511 w_k1_INTC_s005
                                 + 0.963814 w_k1_KO_s005
                                 + 0.803042 w_k1_MAT_s005
                                 + 1.00707 w_k1_MSFT_s005 + 1.0278 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.03856 w_k1_ADM_s006
                                 + 0.931909 w_k1_AXP_s006
                                 + 1.03177 w_k1_DIS_s006
                                 + 0.93503 w_k1_FDX_s006
                                 + 0.874737 w_k1_IBM_s006
                                 + 1.09509 w_k1_INTC_s006
                                 + 1.06369 w_k1_KO_s006
                                 + 0.679013 w_k1_MAT_s006
                                 + 1.14027 w_k1_MSFT_s006
                                 + 1.08805 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.48273 w_k1_ADM_s006
                                 + 1.37819 w_k1_AXP_s006
                                 + 1.81611 w_k1_DIS_s006
                                 + 1.37164 w_k1_FDX_s006
                                 + 1.30152 w_k1_IBM_s006
                                 + 1.37894 w_k1_INTC_s006
                                 + 1.65352 w_k1_KO_s006 + 1.07144 w_k1_MAT_s006
                                 + 1.49688 w_k1_MSFT_s006
                                 + 1.49715 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.36284 w_k1_ADM_s006
                                 + 1.22329 w_k1_AXP_s006
                                 + 1.17545 w_k1_DIS_s006
                                 + 1.27471 w_k1_FDX_s006
                                 + 0.794717 w_k1_IBM_s006
                                 + 1.28936 w_k1_INTC_s006
                                 + 1.00499 w_k1_KO_s006
                                 + 0.684571 w_k1_MAT_s006
                                 + 1.16675 w_k1_MSFT_s006
                                 + 1.10791 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.839933 w_k1_ADM_s006
                                 + 0.991201 w_k1_AXP_s006
                                 + 0.696846 w_k1_DIS_s006
                                 + 0.892133 w_k1_FDX_s006
                                 + 0.861718 w_k1_IBM_s006
                                 + 0.797627 w_k1_INTC_s006
                                 + 0.815792 w_k1_KO_s006
                                 + 0.855948 w_k1_MAT_s006
                                 + 0.817232 w_k1_MSFT_s006
                                 + 0.920902 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.935868 w_k1_ADM_s006
                                 + 1.27405 w_k1_AXP_s006
                                 + 0.903525 w_k1_DIS_s006
                                 + 0.977902 w_k1_FDX_s006
                                 + 1.19684 w_k1_IBM_s006
                                 + 1.13225 w_k1_INTC_s006
                                 + 1.11157 w_k1_KO_s006 + 1.26235 w_k1_MAT_s006
                                 + 0.990228 w_k1_MSFT_s006
                                 + 1.0058 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.14806 w_k1_ADM_s006
                                 + 1.2543 w_k1_AXP_s006 + 1.38448 w_k1_DIS_s006
                                 + 1.23673 w_k1_FDX_s006
                                 + 1.14515 w_k1_IBM_s006
                                 + 1.39954 w_k1_INTC_s006
                                 + 1.15181 w_k1_KO_s006
                                 + 0.871005 w_k1_MAT_s006
                                 + 1.18381 w_k1_MSFT_s006
                                 + 1.19634 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.706515 w_k1_ADM_s006
                                 + 0.799907 w_k1_AXP_s006
                                 + 0.647354 w_k1_DIS_s006
                                 + 0.669851 w_k1_FDX_s006
                                 + 1.12557 w_k1_IBM_s006
                                 + 0.86542 w_k1_INTC_s006
                                 + 0.831577 w_k1_KO_s006
                                 + 0.892092 w_k1_MAT_s006
                                 + 0.724906 w_k1_MSFT_s006
                                 + 0.868752 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.31333 w_k1_ADM_s006
                                 + 1.55478 w_k1_AXP_s006
                                 + 1.29443 w_k1_DIS_s006
                                 + 1.11713 w_k1_FDX_s006
                                 + 1.63611 w_k1_IBM_s006
                                 + 0.97505 w_k1_INTC_s006
                                 + 1.45239 w_k1_KO_s006 + 1.31381 w_k1_MAT_s006
                                 + 0.987667 w_k1_MSFT_s006
                                 + 1.26995 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.10285 w_k1_ADM_s007
                                 + 1.14067 w_k1_AXP_s007
                                 + 1.22803 w_k1_DIS_s007 + 1.0246 w_k1_FDX_s007
                                 + 1.68128 w_k1_IBM_s007
                                 + 1.04064 w_k1_INTC_s007
                                 + 1.24067 w_k1_KO_s007 + 1.75363 w_k1_MAT_s007
                                 + 0.934741 w_k1_MSFT_s007
                                 + 1.22209 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.06638 w_k1_ADM_s007
                                 + 1.08537 w_k1_AXP_s007 + 0.9413 w_k1_DIS_s007
                                 + 1.06777 w_k1_FDX_s007
                                 + 0.839246 w_k1_IBM_s007
                                 + 1.30541 w_k1_INTC_s007
                                 + 0.885693 w_k1_KO_s007
                                 + 0.726172 w_k1_MAT_s007
                                 + 0.984377 w_k1_MSFT_s007
                                 + 0.961089 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.920518 w_k1_ADM_s007
                                 + 0.75654 w_k1_AXP_s007
                                 + 1.00429 w_k1_DIS_s007
                                 + 0.999749 w_k1_FDX_s007
                                 + 0.737618 w_k1_IBM_s007
                                 + 0.971139 w_k1_INTC_s007
                                 + 0.931032 w_k1_KO_s007
                                 + 0.705123 w_k1_MAT_s007
                                 + 1.05229 w_k1_MSFT_s007
                                 + 0.933851 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.973203 w_k1_ADM_s007
                                 + 0.970362 w_k1_AXP_s007
                                 + 0.810516 w_k1_DIS_s007
                                 + 0.827613 w_k1_FDX_s007
                                 + 1.28191 w_k1_IBM_s007
                                 + 1.19943 w_k1_INTC_s007
                                 + 1.10965 w_k1_KO_s007 + 1.25545 w_k1_MAT_s007
                                 + 0.797474 w_k1_MSFT_s007
                                 + 0.969121 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.70042 w_k1_ADM_s007
                                 + 0.568132 w_k1_AXP_s007
                                 + 0.575962 w_k1_DIS_s007
                                 + 0.683915 w_k1_FDX_s007
                                 + 0.984969 w_k1_IBM_s007
                                 + 0.991272 w_k1_INTC_s007
                                 + 0.769877 w_k1_KO_s007
                                 + 0.704739 w_k1_MAT_s007
                                 + 0.727442 w_k1_MSFT_s007
                                 + 0.710792 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.31155 w_k1_ADM_s007
                                 + 1.03826 w_k1_AXP_s007
                                 + 0.961382 w_k1_DIS_s007
                                 + 1.30488 w_k1_FDX_s007
                                 + 0.565451 w_k1_IBM_s007
                                 + 1.25847 w_k1_INTC_s007
                                 + 0.897573 w_k1_KO_s007
                                 + 0.569829 w_k1_MAT_s007
                                 + 1.09505 w_k1_MSFT_s007
                                 + 0.942495 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.64616 w_k1_ADM_s007
                                 + 0.591043 w_k1_AXP_s007
                                 + 0.711423 w_k1_DIS_s007
                                 + 0.796498 w_k1_FDX_s007
                                 + 0.766259 w_k1_IBM_s007
                                 + 1.08052 w_k1_INTC_s007
                                 + 0.894829 w_k1_KO_s007
                                 + 0.740495 w_k1_MAT_s007
                                 + 0.898212 w_k1_MSFT_s007
                                 + 0.840655 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.23685 w_k1_ADM_s007
                                 + 1.29041 w_k1_AXP_s007
                                 + 1.55746 w_k1_DIS_s007 + 1.4508 w_k1_FDX_s007
                                 + 0.945386 w_k1_IBM_s007
                                 + 1.10632 w_k1_INTC_s007
                                 + 1.23656 w_k1_KO_s007 + 1.04121 w_k1_MAT_s007
                                 + 1.40949 w_k1_MSFT_s007
                                 + 1.24871 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.17794 w_k1_ADM_s010
                                 + 1.19318 w_k1_AXP_s010
                                 + 1.33103 w_k1_DIS_s010
                                 + 1.15772 w_k1_FDX_s010
                                 + 1.10801 w_k1_IBM_s010
                                 + 1.32823 w_k1_INTC_s010 + 1.3143 w_k1_KO_s010
                                 + 0.583981 w_k1_MAT_s010
                                 + 1.13669 w_k1_MSFT_s010
                                 + 1.15586 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.921322 w_k1_ADM_s010
                                 + 0.948834 w_k1_AXP_s010
                                 + 0.739556 w_k1_DIS_s010
                                 + 0.936053 w_k1_FDX_s010
                                 + 0.558571 w_k1_IBM_s010
                                 + 1.04516 w_k1_INTC_s010
                                 + 0.677542 w_k1_KO_s010
                                 + 0.441069 w_k1_MAT_s010
                                 + 0.798365 w_k1_MSFT_s010
                                 + 0.778574 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.02987 w_k1_ADM_s010
                                 + 0.913188 w_k1_AXP_s010
                                 + 0.97679 w_k1_DIS_s010
                                 + 0.905621 w_k1_FDX_s010
                                 + 0.92701 w_k1_IBM_s010
                                 + 0.806407 w_k1_INTC_s010
                                 + 0.981378 w_k1_KO_s010
                                 + 0.759641 w_k1_MAT_s010
                                 + 0.80994 w_k1_MSFT_s010
                                 + 1.05061 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.10121 w_k1_ADM_s010
                                 + 1.24575 w_k1_AXP_s010
                                 + 0.896687 w_k1_DIS_s010
                                 + 1.05165 w_k1_FDX_s010
                                 + 0.949721 w_k1_IBM_s010
                                 + 1.00125 w_k1_INTC_s010
                                 + 0.862782 w_k1_KO_s010
                                 + 0.858298 w_k1_MAT_s010
                                 + 0.984572 w_k1_MSFT_s010
                                 + 0.984518 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.882328 w_k1_ADM_s010
                                 + 0.685549 w_k1_AXP_s010
                                 + 0.688456 w_k1_DIS_s010
                                 + 0.715357 w_k1_FDX_s010
                                 + 1.16144 w_k1_IBM_s010
                                 + 0.751743 w_k1_INTC_s010
                                 + 0.829477 w_k1_KO_s010
                                 + 0.565166 w_k1_MAT_s010
                                 + 0.712923 w_k1_MSFT_s010
                                 + 0.827545 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.05157 w_k1_ADM_s010
                                 + 1.15769 w_k1_AXP_s010
                                 + 1.09763 w_k1_DIS_s010
                                 + 0.975774 w_k1_FDX_s010
                                 + 1.63186 w_k1_IBM_s010
                                 + 0.991227 w_k1_INTC_s010
                                 + 1.29008 w_k1_KO_s010 + 1.21317 w_k1_MAT_s010
                                 + 0.961809 w_k1_MSFT_s010
                                 + 1.06213 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.53348 w_k1_ADM_s010
                                 + 2.04484 w_k1_AXP_s010
                                 + 2.07849 w_k1_DIS_s010
                                 + 1.71046 w_k1_FDX_s010
                                 + 1.18545 w_k1_IBM_s010
                                 + 1.86819 w_k1_INTC_s010
                                 + 1.56168 w_k1_KO_s010
                                 + 0.911786 w_k1_MAT_s010
                                 + 1.40924 w_k1_MSFT_s010
                                 + 1.42795 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.731676 w_k1_ADM_s010
                                 + 0.677267 w_k1_AXP_s010
                                 + 0.648693 w_k1_DIS_s010
                                 + 0.73421 w_k1_FDX_s010
                                 + 0.729389 w_k1_IBM_s010
                                 + 0.821413 w_k1_INTC_s010
                                 + 0.740284 w_k1_KO_s010
                                 + 0.471831 w_k1_MAT_s010
                                 + 0.740087 w_k1_MSFT_s010
                                 + 0.845139 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.935812 w_k1_ADM_s011
                                 + 0.806086 w_k1_AXP_s011
                                 + 0.973995 w_k1_DIS_s011
                                 + 0.860715 w_k1_FDX_s011
                                 + 1.16231 w_k1_IBM_s011
                                 + 0.984506 w_k1_INTC_s011
                                 + 0.992068 w_k1_KO_s011
                                 + 0.851915 w_k1_MAT_s011
                                 + 0.904305 w_k1_MSFT_s011
                                 + 1.04383 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.909717 w_k1_ADM_s011
                                 + 0.998854 w_k1_AXP_s011
                                 + 1.12666 w_k1_DIS_s011
                                 + 1.11925 w_k1_FDX_s011 + 1.0663 w_k1_IBM_s011
                                 + 1.03153 w_k1_INTC_s011 + 0.9852 w_k1_KO_s011
                                 + 1.33045 w_k1_MAT_s011
                                 + 1.15273 w_k1_MSFT_s011
                                 + 1.01879 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.97542 w_k1_ADM_s011
                                 + 0.942648 w_k1_AXP_s011
                                 + 0.818468 w_k1_DIS_s011
                                 + 1.0727 w_k1_FDX_s011
                                 + 0.636274 w_k1_IBM_s011
                                 + 0.999617 w_k1_INTC_s011
                                 + 0.748536 w_k1_KO_s011
                                 + 0.512801 w_k1_MAT_s011
                                 + 1.04738 w_k1_MSFT_s011
                                 + 0.975975 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.909926 w_k1_ADM_s011
                                 + 0.818982 w_k1_AXP_s011
                                 + 0.887822 w_k1_DIS_s011
                                 + 1.03943 w_k1_FDX_s011
                                 + 0.806765 w_k1_IBM_s011
                                 + 0.990603 w_k1_INTC_s011
                                 + 0.815903 w_k1_KO_s011
                                 + 0.813985 w_k1_MAT_s011
                                 + 0.996137 w_k1_MSFT_s011
                                 + 0.90044 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.05741 w_k1_ADM_s011
                                 + 1.13052 w_k1_AXP_s011
                                 + 0.934569 w_k1_DIS_s011
                                 + 0.968253 w_k1_FDX_s011
                                 + 1.19085 w_k1_IBM_s011
                                 + 0.831499 w_k1_INTC_s011
                                 + 0.970063 w_k1_KO_s011
                                 + 1.86597 w_k1_MAT_s011
                                 + 0.907441 w_k1_MSFT_s011
                                 + 1.10297 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.27171 w_k1_ADM_s011
                                 + 1.28993 w_k1_AXP_s011
                                 + 1.01674 w_k1_DIS_s011
                                 + 1.34706 w_k1_FDX_s011 + 1.0589 w_k1_IBM_s011
                                 + 1.01055 w_k1_INTC_s011
                                 + 0.936983 w_k1_KO_s011
                                 + 1.16591 w_k1_MAT_s011
                                 + 1.24266 w_k1_MSFT_s011
                                 + 1.02307 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.760033 w_k1_ADM_s011
                                 + 0.664503 w_k1_AXP_s011
                                 + 0.679677 w_k1_DIS_s011
                                 + 0.766038 w_k1_FDX_s011
                                 + 0.772366 w_k1_IBM_s011
                                 + 0.776583 w_k1_INTC_s011
                                 + 0.629129 w_k1_KO_s011
                                 + 0.481422 w_k1_MAT_s011
                                 + 0.826618 w_k1_MSFT_s011
                                 + 0.824916 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.28879 w_k1_ADM_s011
                                 + 1.5606 w_k1_AXP_s011 + 1.41805 w_k1_DIS_s011
                                 + 1.30661 w_k1_FDX_s011
                                 + 1.45007 w_k1_IBM_s011
                                 + 1.17852 w_k1_INTC_s011
                                 + 1.38532 w_k1_KO_s011 + 1.56638 w_k1_MAT_s011
                                 + 1.18684 w_k1_MSFT_s011
                                 + 1.41738 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.23218 w_k1_ADM_s012
                                 + 1.48405 w_k1_AXP_s012
                                 + 1.58746 w_k1_DIS_s012
                                 + 1.34559 w_k1_FDX_s012
                                 + 1.17971 w_k1_IBM_s012
                                 + 1.24859 w_k1_INTC_s012
                                 + 1.29174 w_k1_KO_s012 + 1.15535 w_k1_MAT_s012
                                 + 1.15524 w_k1_MSFT_s012
                                 + 1.29085 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.912149 w_k1_ADM_s012
                                 + 0.631743 w_k1_AXP_s012
                                 + 0.604207 w_k1_DIS_s012
                                 + 0.693691 w_k1_FDX_s012
                                 + 0.652001 w_k1_IBM_s012
                                 + 0.713517 w_k1_INTC_s012
                                 + 0.729123 w_k1_KO_s012
                                 + 0.591058 w_k1_MAT_s012
                                 + 0.745802 w_k1_MSFT_s012
                                 + 0.842355 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.956987 w_k1_ADM_s012
                                 + 0.884735 w_k1_AXP_s012
                                 + 0.851912 w_k1_DIS_s012
                                 + 0.812904 w_k1_FDX_s012
                                 + 0.915898 w_k1_IBM_s012
                                 + 0.710125 w_k1_INTC_s012
                                 + 0.936075 w_k1_KO_s012
                                 + 1.09305 w_k1_MAT_s012
                                 + 0.853684 w_k1_MSFT_s012
                                 + 1.01119 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.81804 w_k1_ADM_s012
                                 + 1.02531 w_k1_AXP_s012
                                 + 1.07102 w_k1_DIS_s012
                                 + 0.868208 w_k1_FDX_s012
                                 + 1.37507 w_k1_IBM_s012
                                 + 0.827393 w_k1_INTC_s012
                                 + 1.18037 w_k1_KO_s012 + 1.11247 w_k1_MAT_s012
                                 + 0.91127 w_k1_MSFT_s012
                                 + 1.00095 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.814283 w_k1_ADM_s012
                                 + 0.871065 w_k1_AXP_s012
                                 + 0.778292 w_k1_DIS_s012
                                 + 0.929968 w_k1_FDX_s012
                                 + 0.706037 w_k1_IBM_s012
                                 + 1.05782 w_k1_INTC_s012
                                 + 0.778491 w_k1_KO_s012
                                 + 0.680609 w_k1_MAT_s012
                                 + 1.03343 w_k1_MSFT_s012
                                 + 0.945739 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.922578 w_k1_ADM_s012
                                 + 1.03988 w_k1_AXP_s012 + 1.1319 w_k1_DIS_s012
                                 + 1.06876 w_k1_FDX_s012
                                 + 0.965055 w_k1_IBM_s012
                                 + 1.02792 w_k1_INTC_s012
                                 + 1.04896 w_k1_KO_s012 + 1.1391 w_k1_MAT_s012
                                 + 1.07064 w_k1_MSFT_s012
                                 + 1.02688 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.25928 w_k1_ADM_s012
                                 + 1.58134 w_k1_AXP_s012
                                 + 1.10262 w_k1_DIS_s012
                                 + 1.15971 w_k1_FDX_s012
                                 + 1.00836 w_k1_IBM_s012
                                 + 0.942318 w_k1_INTC_s012
                                 + 0.877427 w_k1_KO_s012
                                 + 0.958795 w_k1_MAT_s012
                                 + 1.02762 w_k1_MSFT_s012
                                 + 1.13635 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.04967 w_k1_ADM_s012
                                 + 1.21699 w_k1_AXP_s012
                                 + 0.796385 w_k1_DIS_s012
                                 + 0.924255 w_k1_FDX_s012
                                 + 0.816571 w_k1_IBM_s012
                                 + 0.883325 w_k1_INTC_s012
                                 + 0.798015 w_k1_KO_s012
                                 + 0.740273 w_k1_MAT_s012
                                 + 0.833011 w_k1_MSFT_s012
                                 + 0.950841 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.40762 w_k1_ADM_s013
                                 + 1.02603 w_k1_AXP_s013
                                 + 1.41144 w_k1_DIS_s013
                                 + 1.21542 w_k1_FDX_s013
                                 + 0.993779 w_k1_IBM_s013
                                 + 0.921732 w_k1_INTC_s013
                                 + 1.07333 w_k1_KO_s013 + 1.2057 w_k1_MAT_s013
                                 + 1.18766 w_k1_MSFT_s013
                                 + 1.09086 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.884734 w_k1_ADM_s013
                                 + 0.679652 w_k1_AXP_s013
                                 + 0.777908 w_k1_DIS_s013
                                 + 0.765641 w_k1_FDX_s013
                                 + 1.33369 w_k1_IBM_s013
                                 + 0.8013 w_k1_INTC_s013
                                 + 0.888432 w_k1_KO_s013
                                 + 1.01488 w_k1_MAT_s013
                                 + 0.811271 w_k1_MSFT_s013
                                 + 0.860292 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.82982 w_k1_ADM_s013
                                 + 1.22625 w_k1_AXP_s013
                                 + 1.02913 w_k1_DIS_s013
                                 + 1.04117 w_k1_FDX_s013
                                 + 1.42219 w_k1_IBM_s013
                                 + 0.947517 w_k1_INTC_s013
                                 + 0.997264 w_k1_KO_s013
                                 + 1.67236 w_k1_MAT_s013
                                 + 0.949379 w_k1_MSFT_s013
                                 + 1.03521 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.881487 w_k1_ADM_s013
                                 + 0.752711 w_k1_AXP_s013 + 0.663 w_k1_DIS_s013
                                 + 0.772129 w_k1_FDX_s013
                                 + 0.819785 w_k1_IBM_s013
                                 + 0.631014 w_k1_INTC_s013
                                 + 0.599608 w_k1_KO_s013
                                 + 0.973275 w_k1_MAT_s013
                                 + 0.777785 w_k1_MSFT_s013
                                 + 0.769592 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.00641 w_k1_ADM_s013
                                 + 0.919412 w_k1_AXP_s013
                                 + 0.87863 w_k1_DIS_s013
                                 + 0.915542 w_k1_FDX_s013
                                 + 1.0876 w_k1_IBM_s013
                                 + 0.73043 w_k1_INTC_s013
                                 + 0.799271 w_k1_KO_s013
                                 + 1.37757 w_k1_MAT_s013
                                 + 0.91408 w_k1_MSFT_s013
                                 + 0.860131 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.986443 w_k1_ADM_s013
                                 + 1.30538 w_k1_AXP_s013
                                 + 1.32434 w_k1_DIS_s013
                                 + 1.15233 w_k1_FDX_s013
                                 + 1.68315 w_k1_IBM_s013
                                 + 1.11244 w_k1_INTC_s013
                                 + 1.25875 w_k1_KO_s013 + 2.70071 w_k1_MAT_s013
                                 + 1.17641 w_k1_MSFT_s013
                                 + 1.05858 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.17927 w_k1_ADM_s013
                                 + 0.789186 w_k1_AXP_s013
                                 + 0.871927 w_k1_DIS_s013
                                 + 0.976606 w_k1_FDX_s013
                                 + 0.752606 w_k1_IBM_s013
                                 + 0.875257 w_k1_INTC_s013
                                 + 0.788714 w_k1_KO_s013
                                 + 0.642571 w_k1_MAT_s013
                                 + 0.95288 w_k1_MSFT_s013
                                 + 0.86252 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.04183 w_k1_ADM_s013
                                 + 0.802331 w_k1_AXP_s013
                                 + 1.26603 w_k1_DIS_s013 + 1.1265 w_k1_FDX_s013
                                 + 1.05195 w_k1_IBM_s013
                                 + 1.00941 w_k1_INTC_s013
                                 + 0.989805 w_k1_KO_s013
                                 + 0.766927 w_k1_MAT_s013
                                 + 1.27109 w_k1_MSFT_s013
                                 + 0.998264 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.915604 w_k1_ADM_s014
                                 + 1.06997 w_k1_AXP_s014
                                 + 0.796568 w_k1_DIS_s014
                                 + 0.860483 w_k1_FDX_s014
                                 + 0.903947 w_k1_IBM_s014
                                 + 0.962524 w_k1_INTC_s014
                                 + 0.844071 w_k1_KO_s014
                                 + 0.630467 w_k1_MAT_s014
                                 + 0.874173 w_k1_MSFT_s014
                                 + 0.91327 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.06376 w_k1_ADM_s014
                                 + 0.948541 w_k1_AXP_s014
                                 + 0.987405 w_k1_DIS_s014
                                 + 0.892827 w_k1_FDX_s014
                                 + 1.30211 w_k1_IBM_s014
                                 + 0.943942 w_k1_INTC_s014
                                 + 1.27891 w_k1_KO_s014 + 1.01975 w_k1_MAT_s014
                                 + 0.940849 w_k1_MSFT_s014
                                 + 1.08166 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.28984 w_k1_ADM_s014
                                 + 1.42165 w_k1_AXP_s014
                                 + 1.41748 w_k1_DIS_s014
                                 + 1.32331 w_k1_FDX_s014
                                 + 0.845149 w_k1_IBM_s014
                                 + 1.20887 w_k1_INTC_s014
                                 + 0.840217 w_k1_KO_s014
                                 + 0.63339 w_k1_MAT_s014
                                 + 0.964241 w_k1_MSFT_s014
                                 + 1.0639 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.10129 w_k1_ADM_s014
                                 + 1.16146 w_k1_AXP_s014
                                 + 1.47127 w_k1_DIS_s014
                                 + 1.27409 w_k1_FDX_s014 + 1.1968 w_k1_IBM_s014
                                 + 1.2468 w_k1_INTC_s014 + 1.11726 w_k1_KO_s014
                                 + 1.02876 w_k1_MAT_s014
                                 + 1.08842 w_k1_MSFT_s014
                                 + 1.13165 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.63259 w_k1_ADM_s014
                                 + 0.665438 w_k1_AXP_s014
                                 + 0.612285 w_k1_DIS_s014
                                 + 0.758549 w_k1_FDX_s014
                                 + 0.692231 w_k1_IBM_s014
                                 + 0.862504 w_k1_INTC_s014
                                 + 0.731467 w_k1_KO_s014
                                 + 0.550155 w_k1_MAT_s014
                                 + 0.766306 w_k1_MSFT_s014
                                 + 0.770195 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.03717 w_k1_ADM_s014
                                 + 0.946815 w_k1_AXP_s014
                                 + 1.08455 w_k1_DIS_s014
                                 + 1.06724 w_k1_FDX_s014
                                 + 0.673004 w_k1_IBM_s014
                                 + 0.896911 w_k1_INTC_s014
                                 + 0.833032 w_k1_KO_s014
                                 + 0.594251 w_k1_MAT_s014
                                 + 0.993296 w_k1_MSFT_s014
                                 + 0.917449 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.744963 w_k1_ADM_s014
                                 + 0.689829 w_k1_AXP_s014
                                 + 0.647497 w_k1_DIS_s014
                                 + 0.659318 w_k1_FDX_s014
                                 + 1.03042 w_k1_IBM_s014
                                 + 0.877766 w_k1_INTC_s014
                                 + 0.759612 w_k1_KO_s014
                                 + 0.638714 w_k1_MAT_s014
                                 + 0.675648 w_k1_MSFT_s014
                                 + 0.82444 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.22982 w_k1_ADM_s014
                                 + 1.31985 w_k1_AXP_s014
                                 + 1.48374 w_k1_DIS_s014 + 1.2011 w_k1_FDX_s014
                                 + 1.51538 w_k1_IBM_s014
                                 + 1.07231 w_k1_INTC_s014
                                 + 1.26965 w_k1_KO_s014 + 1.42287 w_k1_MAT_s014
                                 + 1.14619 w_k1_MSFT_s014
                                 + 1.16071 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 2.1764 w_k1_ADM_s015
                                 + 1.42201 w_k1_AXP_s015 + 1.8762 w_k1_DIS_s015
                                 + 1.57458 w_k1_FDX_s015 + 1.1247 w_k1_IBM_s015
                                 + 1.07902 w_k1_INTC_s015
                                 + 1.38664 w_k1_KO_s015 + 1.33841 w_k1_MAT_s015
                                 + 1.22654 w_k1_MSFT_s015
                                 + 1.38271 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.24007 w_k1_ADM_s015
                                 + 1.09964 w_k1_AXP_s015
                                 + 1.33735 w_k1_DIS_s015
                                 + 1.30235 w_k1_FDX_s015
                                 + 0.867788 w_k1_IBM_s015
                                 + 1.17064 w_k1_INTC_s015
                                 + 1.07987 w_k1_KO_s015
                                 + 0.860896 w_k1_MAT_s015
                                 + 1.20656 w_k1_MSFT_s015 + 1.2924 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.942916 w_k1_ADM_s015
                                 + 0.659565 w_k1_AXP_s015
                                 + 0.642357 w_k1_DIS_s015
                                 + 0.794978 w_k1_FDX_s015
                                 + 0.770171 w_k1_IBM_s015
                                 + 0.868749 w_k1_INTC_s015
                                 + 0.787011 w_k1_KO_s015
                                 + 0.511481 w_k1_MAT_s015
                                 + 0.816962 w_k1_MSFT_s015
                                 + 0.93774 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.2536 w_k1_ADM_s015
                                 + 1.08763 w_k1_AXP_s015
                                 + 1.00235 w_k1_DIS_s015
                                 + 1.10787 w_k1_FDX_s015
                                 + 0.978426 w_k1_IBM_s015
                                 + 0.829885 w_k1_INTC_s015
                                 + 1.03504 w_k1_KO_s015 + 1.18259 w_k1_MAT_s015
                                 + 0.869546 w_k1_MSFT_s015
                                 + 1.13598 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.0469 w_k1_ADM_s015
                                 + 0.872084 w_k1_AXP_s015
                                 + 1.00433 w_k1_DIS_s015
                                 + 0.954153 w_k1_FDX_s015
                                 + 1.12759 w_k1_IBM_s015
                                 + 1.04155 w_k1_INTC_s015
                                 + 1.12656 w_k1_KO_s015 + 1.00939 w_k1_MAT_s015
                                 + 0.908114 w_k1_MSFT_s015
                                 + 1.1255 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.06371 w_k1_ADM_s015
                                 + 1.0926 w_k1_AXP_s015
                                 + 0.920251 w_k1_DIS_s015
                                 + 1.04925 w_k1_FDX_s015
                                 + 0.721169 w_k1_IBM_s015
                                 + 1.02913 w_k1_INTC_s015
                                 + 0.803651 w_k1_KO_s015
                                 + 0.698131 w_k1_MAT_s015
                                 + 0.937416 w_k1_MSFT_s015
                                 + 0.975918 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.26641 w_k1_ADM_s015
                                 + 1.23307 w_k1_AXP_s015
                                 + 1.09871 w_k1_DIS_s015
                                 + 1.14456 w_k1_FDX_s015
                                 + 1.50886 w_k1_IBM_s015
                                 + 1.28834 w_k1_INTC_s015 + 1.248 w_k1_KO_s015
                                 + 1.18056 w_k1_MAT_s015
                                 + 1.21838 w_k1_MSFT_s015 + 1.1478 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.888519 w_k1_ADM_s015
                                 + 0.837417 w_k1_AXP_s015
                                 + 0.513332 w_k1_DIS_s015
                                 + 0.70802 w_k1_FDX_s015
                                 + 1.03749 w_k1_IBM_s015
                                 + 0.70269 w_k1_INTC_s015
                                 + 0.815669 w_k1_KO_s015
                                 + 1.01256 w_k1_MAT_s015
                                 + 0.639452 w_k1_MSFT_s015
                                 + 0.958218 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.969295 w_k1_ADM_s016
                                 + 0.769566 w_k1_AXP_s016
                                 + 0.829115 w_k1_DIS_s016
                                 + 0.823808 w_k1_FDX_s016
                                 + 0.995677 w_k1_IBM_s016
                                 + 0.643952 w_k1_INTC_s016
                                 + 0.894533 w_k1_KO_s016
                                 + 1.31736 w_k1_MAT_s016
                                 + 0.802023 w_k1_MSFT_s016
                                 + 0.956702 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.13294 w_k1_ADM_s016
                                 + 0.908357 w_k1_AXP_s016
                                 + 0.847146 w_k1_DIS_s016
                                 + 0.964571 w_k1_FDX_s016 + 1.229 w_k1_IBM_s016
                                 + 0.735893 w_k1_INTC_s016
                                 + 1.12046 w_k1_KO_s016 + 1.26411 w_k1_MAT_s016
                                 + 0.908177 w_k1_MSFT_s016
                                 + 0.990191 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.19605 w_k1_ADM_s016
                                 + 0.669491 w_k1_AXP_s016
                                 + 0.945019 w_k1_DIS_s016
                                 + 0.892409 w_k1_FDX_s016
                                 + 0.766579 w_k1_IBM_s016
                                 + 0.733548 w_k1_INTC_s016
                                 + 0.822842 w_k1_KO_s016 + 0.6621 w_k1_MAT_s016
                                 + 0.992024 w_k1_MSFT_s016
                                 + 0.823398 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.22586 w_k1_ADM_s016
                                 + 0.979069 w_k1_AXP_s016
                                 + 1.10135 w_k1_DIS_s016
                                 + 0.943585 w_k1_FDX_s016
                                 + 1.48481 w_k1_IBM_s016
                                 + 0.708555 w_k1_INTC_s016 + 1.339 w_k1_KO_s016
                                 + 2.01332 w_k1_MAT_s016
                                 + 1.12743 w_k1_MSFT_s016
                                 + 1.04819 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.46757 w_k1_ADM_s016
                                 + 1.09644 w_k1_AXP_s016
                                 + 1.89764 w_k1_DIS_s016
                                 + 1.31957 w_k1_FDX_s016
                                 + 1.17298 w_k1_IBM_s016
                                 + 1.19201 w_k1_INTC_s016
                                 + 1.60294 w_k1_KO_s016
                                 + 0.914016 w_k1_MAT_s016
                                 + 1.41293 w_k1_MSFT_s016
                                 + 1.17206 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.38844 w_k1_ADM_s016
                                 + 1.0308 w_k1_AXP_s016 + 1.25704 w_k1_DIS_s016
                                 + 1.21045 w_k1_FDX_s016
                                 + 0.860157 w_k1_IBM_s016
                                 + 0.827924 w_k1_INTC_s016
                                 + 1.08146 w_k1_KO_s016
                                 + 0.948748 w_k1_MAT_s016
                                 + 1.15405 w_k1_MSFT_s016
                                 + 1.02743 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.2246 w_k1_ADM_s016
                                 + 0.703121 w_k1_AXP_s016
                                 + 0.596751 w_k1_DIS_s016
                                 + 0.789467 w_k1_FDX_s016
                                 + 0.745796 w_k1_IBM_s016
                                 + 0.70274 w_k1_INTC_s016
                                 + 0.799567 w_k1_KO_s016
                                 + 0.818923 w_k1_MAT_s016
                                 + 0.803508 w_k1_MSFT_s016
                                 + 0.919666 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.37332 w_k1_ADM_s016
                                 + 1.44939 w_k1_AXP_s016
                                 + 1.42675 w_k1_DIS_s016
                                 + 1.38981 w_k1_FDX_s016
                                 + 1.08767 w_k1_IBM_s016
                                 + 1.24292 w_k1_INTC_s016
                                 + 1.13106 w_k1_KO_s016 + 1.3712 w_k1_MAT_s016
                                 + 1.38846 w_k1_MSFT_s016
                                 + 1.08256 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.39563 w_k1_ADM_s017
                                 + 1.38306 w_k1_AXP_s017
                                 + 1.46095 w_k1_DIS_s017
                                 + 1.41141 w_k1_FDX_s017
                                 + 1.21699 w_k1_IBM_s017
                                 + 1.30914 w_k1_INTC_s017
                                 + 1.27192 w_k1_KO_s017 + 1.11431 w_k1_MAT_s017
                                 + 1.28815 w_k1_MSFT_s017
                                 + 1.10345 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.800598 w_k1_ADM_s017
                                 + 1.30856 w_k1_AXP_s017
                                 + 0.855872 w_k1_DIS_s017
                                 + 1.01838 w_k1_FDX_s017
                                 + 0.903658 w_k1_IBM_s017
                                 + 1.06438 w_k1_INTC_s017
                                 + 0.839859 w_k1_KO_s017
                                 + 1.48395 w_k1_MAT_s017
                                 + 1.0122 w_k1_MSFT_s017 + 1.0084 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.992543 w_k1_ADM_s017
                                 + 1.32574 w_k1_AXP_s017 + 1.1375 w_k1_DIS_s017
                                 + 1.07216 w_k1_FDX_s017
                                 + 1.57166 w_k1_IBM_s017
                                 + 1.28031 w_k1_INTC_s017
                                 + 1.14463 w_k1_KO_s017 + 1.92179 w_k1_MAT_s017
                                 + 1.18742 w_k1_MSFT_s017
                                 + 1.09662 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.937959 w_k1_ADM_s017
                                 + 0.79875 w_k1_AXP_s017
                                 + 0.795567 w_k1_DIS_s017
                                 + 0.947758 w_k1_FDX_s017
                                 + 0.756416 w_k1_IBM_s017
                                 + 0.978843 w_k1_INTC_s017
                                 + 0.75592 w_k1_KO_s017 + 1.09685 w_k1_MAT_s017
                                 + 1.06336 w_k1_MSFT_s017
                                 + 0.826737 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.775721 w_k1_ADM_s017
                                 + 0.650473 w_k1_AXP_s017
                                 + 0.533615 w_k1_DIS_s017
                                 + 0.713237 w_k1_FDX_s017
                                 + 0.826943 w_k1_IBM_s017
                                 + 0.792665 w_k1_INTC_s017
                                 + 0.871066 w_k1_KO_s017
                                 + 0.948954 w_k1_MAT_s017
                                 + 0.816221 w_k1_MSFT_s017
                                 + 0.826538 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.898845 w_k1_ADM_s017
                                 + 1.07537 w_k1_AXP_s017
                                 + 1.13665 w_k1_DIS_s017
                                 + 1.11575 w_k1_FDX_s017
                                 + 1.09151 w_k1_IBM_s017
                                 + 1.35404 w_k1_INTC_s017
                                 + 1.17585 w_k1_KO_s017 + 1.43229 w_k1_MAT_s017
                                 + 1.2562 w_k1_MSFT_s017 + 0.90419 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.769609 w_k1_ADM_s017
                                 + 0.96486 w_k1_AXP_s017
                                 + 0.725054 w_k1_DIS_s017
                                 + 0.800368 w_k1_FDX_s017
                                 + 1.24535 w_k1_IBM_s017
                                 + 0.854719 w_k1_INTC_s017
                                 + 0.819137 w_k1_KO_s017
                                 + 1.25894 w_k1_MAT_s017
                                 + 0.801012 w_k1_MSFT_s017
                                 + 0.821486 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.83612 w_k1_ADM_s017
                                 + 1.00623 w_k1_AXP_s017
                                 + 0.966286 w_k1_DIS_s017
                                 + 1.11287 w_k1_FDX_s017
                                 + 0.687073 w_k1_IBM_s017
                                 + 0.951954 w_k1_INTC_s017
                                 + 0.999104 w_k1_KO_s017
                                 + 1.41771 w_k1_MAT_s017
                                 + 1.0405 w_k1_MSFT_s017 + 1.00424 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.08723 w_k1_ADM_s100
                                 + 1.18991 w_k1_AXP_s100 + 1.113 w_k1_DIS_s100
                                 + 1.04698 w_k1_FDX_s100
                                 + 1.40256 w_k1_IBM_s100
                                 + 1.19843 w_k1_INTC_s100
                                 + 1.25431 w_k1_KO_s100 + 1.31037 w_k1_MAT_s100
                                 + 1.07856 w_k1_MSFT_s100
                                 + 1.05658 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.966077 w_k1_ADM_s100
                                 + 0.642819 w_k1_AXP_s100
                                 + 0.475345 w_k1_DIS_s100
                                 + 0.662691 w_k1_FDX_s100
                                 + 0.862961 w_k1_IBM_s100
                                 + 0.775862 w_k1_INTC_s100
                                 + 0.662411 w_k1_KO_s100
                                 + 0.82155 w_k1_MAT_s100
                                 + 0.776795 w_k1_MSFT_s100
                                 + 0.779137 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.04992 w_k1_ADM_s100
                                 + 1.23908 w_k1_AXP_s100
                                 + 0.880143 w_k1_DIS_s100
                                 + 0.971685 w_k1_FDX_s100
                                 + 1.16686 w_k1_IBM_s100
                                 + 1.23627 w_k1_INTC_s100
                                 + 1.11068 w_k1_KO_s100 + 1.08839 w_k1_MAT_s100
                                 + 1.05948 w_k1_MSFT_s100
                                 + 1.00419 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.05018 w_k1_ADM_s100
                                 + 0.891508 w_k1_AXP_s100
                                 + 0.858139 w_k1_DIS_s100
                                 + 1.01011 w_k1_FDX_s100
                                 + 0.665888 w_k1_IBM_s100
                                 + 0.846944 w_k1_INTC_s100
                                 + 0.751583 w_k1_KO_s100
                                 + 0.480722 w_k1_MAT_s100
                                 + 0.939505 w_k1_MSFT_s100
                                 + 0.891362 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.48137 w_k1_ADM_s100
                                 + 1.40026 w_k1_AXP_s100
                                 + 1.35026 w_k1_DIS_s100
                                 + 1.30883 w_k1_FDX_s100
                                 + 0.876952 w_k1_IBM_s100
                                 + 1.17599 w_k1_INTC_s100
                                 + 1.16288 w_k1_KO_s100
                                 + 0.967378 w_k1_MAT_s100
                                 + 1.34425 w_k1_MSFT_s100
                                 + 1.23697 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.41789 w_k1_ADM_s100
                                 + 1.56196 w_k1_AXP_s100
                                 + 1.42423 w_k1_DIS_s100
                                 + 1.20701 w_k1_FDX_s100
                                 + 1.76825 w_k1_IBM_s100
                                 + 1.17268 w_k1_INTC_s100
                                 + 1.70114 w_k1_KO_s100 + 1.4544 w_k1_MAT_s100
                                 + 1.08085 w_k1_MSFT_s100
                                 + 1.26022 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.32645 w_k1_ADM_s100
                                 + 1.48319 w_k1_AXP_s100 + 1.6293 w_k1_DIS_s100
                                 + 1.31477 w_k1_FDX_s100
                                 + 1.29663 w_k1_IBM_s100
                                 + 1.31159 w_k1_INTC_s100
                                 + 1.34696 w_k1_KO_s100 + 1.13804 w_k1_MAT_s100
                                 + 1.47005 w_k1_MSFT_s100
                                 + 1.18719 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.888292 w_k1_ADM_s100
                                 + 0.805804 w_k1_AXP_s100
                                 + 0.816868 w_k1_DIS_s100
                                 + 0.901125 w_k1_FDX_s100
                                 + 0.925875 w_k1_IBM_s100
                                 + 1.09864 w_k1_INTC_s100
                                 + 0.844858 w_k1_KO_s100
                                 + 0.900911 w_k1_MAT_s100
                                 + 0.940566 w_k1_MSFT_s100
                                 + 0.921939 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.01197 w_k1_ADM_s101
                                 + 1.03097 w_k1_AXP_s101
                                 + 0.806629 w_k1_DIS_s101
                                 + 0.965266 w_k1_FDX_s101
                                 + 0.744954 w_k1_IBM_s101
                                 + 0.758583 w_k1_INTC_s101
                                 + 0.790522 w_k1_KO_s101
                                 + 0.718531 w_k1_MAT_s101
                                 + 0.854041 w_k1_MSFT_s101
                                 + 0.882225 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.865329 w_k1_ADM_s101
                                 + 1.45441 w_k1_AXP_s101
                                 + 1.01909 w_k1_DIS_s101
                                 + 1.25414 w_k1_FDX_s101
                                 + 0.772513 w_k1_IBM_s101
                                 + 0.991292 w_k1_INTC_s101
                                 + 0.765882 w_k1_KO_s101
                                 + 0.932153 w_k1_MAT_s101
                                 + 0.969267 w_k1_MSFT_s101
                                 + 0.982573 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.913602 w_k1_ADM_s101
                                 + 0.761948 w_k1_AXP_s101
                                 + 0.680418 w_k1_DIS_s101
                                 + 0.807388 w_k1_FDX_s101
                                 + 0.603455 w_k1_IBM_s101
                                 + 0.742118 w_k1_INTC_s101
                                 + 0.675178 w_k1_KO_s101
                                 + 0.571965 w_k1_MAT_s101
                                 + 0.803623 w_k1_MSFT_s101
                                 + 0.820425 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.829578 w_k1_ADM_s101
                                 + 0.794931 w_k1_AXP_s101
                                 + 0.737056 w_k1_DIS_s101
                                 + 0.808675 w_k1_FDX_s101
                                 + 0.96575 w_k1_IBM_s101
                                 + 0.860866 w_k1_INTC_s101
                                 + 0.860238 w_k1_KO_s101
                                 + 0.888557 w_k1_MAT_s101
                                 + 0.837142 w_k1_MSFT_s101
                                 + 0.963177 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.02511 w_k1_ADM_s101
                                 + 1.16151 w_k1_AXP_s101 + 1.1666 w_k1_DIS_s101
                                 + 1.10813 w_k1_FDX_s101
                                 + 0.962756 w_k1_IBM_s101
                                 + 1.07095 w_k1_INTC_s101
                                 + 1.05836 w_k1_KO_s101 + 1.05457 w_k1_MAT_s101
                                 + 1.0192 w_k1_MSFT_s101 + 1.07067 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.03366 w_k1_ADM_s101
                                 + 1.29547 w_k1_AXP_s101
                                 + 1.51112 w_k1_DIS_s101
                                 + 1.31643 w_k1_FDX_s101
                                 + 0.847695 w_k1_IBM_s101
                                 + 1.05522 w_k1_INTC_s101
                                 + 0.976245 w_k1_KO_s101
                                 + 0.940011 w_k1_MAT_s101
                                 + 1.13992 w_k1_MSFT_s101
                                 + 1.14453 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.964864 w_k1_ADM_s101
                                 + 1.29658 w_k1_AXP_s101
                                 + 1.36306 w_k1_DIS_s101
                                 + 1.14904 w_k1_FDX_s101
                                 + 1.51727 w_k1_IBM_s101
                                 + 0.984914 w_k1_INTC_s101
                                 + 1.23892 w_k1_KO_s101 + 1.29042 w_k1_MAT_s101
                                 + 0.982069 w_k1_MSFT_s101
                                 + 1.14413 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.821917 w_k1_ADM_s101
                                 + 0.969749 w_k1_AXP_s101
                                 + 0.97748 w_k1_DIS_s101
                                 + 0.937391 w_k1_FDX_s101
                                 + 1.18694 w_k1_IBM_s101
                                 + 0.854142 w_k1_INTC_s101
                                 + 0.963702 w_k1_KO_s101
                                 + 0.890174 w_k1_MAT_s101
                                 + 0.918233 w_k1_MSFT_s101
                                 + 0.992788 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.906185 w_k1_ADM_s102
                                 + 1.23008 w_k1_AXP_s102
                                 + 0.978676 w_k1_DIS_s102
                                 + 1.04991 w_k1_FDX_s102
                                 + 1.16376 w_k1_IBM_s102
                                 + 1.15547 w_k1_INTC_s102
                                 + 1.05493 w_k1_KO_s102 + 1.11421 w_k1_MAT_s102
                                 + 0.975928 w_k1_MSFT_s102
                                 + 1.02358 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.772928 w_k1_ADM_s102
                                 + 0.621152 w_k1_AXP_s102
                                 + 0.478611 w_k1_DIS_s102
                                 + 0.634191 w_k1_FDX_s102
                                 + 0.825356 w_k1_IBM_s102
                                 + 0.656889 w_k1_INTC_s102
                                 + 0.694774 w_k1_KO_s102
                                 + 0.690481 w_k1_MAT_s102
                                 + 0.552528 w_k1_MSFT_s102
                                 + 0.765978 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.10261 w_k1_ADM_s102
                                 + 1.01248 w_k1_AXP_s102
                                 + 1.08821 w_k1_DIS_s102
                                 + 1.19244 w_k1_FDX_s102 + 0.8143 w_k1_IBM_s102
                                 + 0.851121 w_k1_INTC_s102
                                 + 0.948994 w_k1_KO_s102
                                 + 0.931782 w_k1_MAT_s102
                                 + 1.04455 w_k1_MSFT_s102
                                 + 0.968681 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.945205 w_k1_ADM_s102
                                 + 1.86059 w_k1_AXP_s102
                                 + 1.19624 w_k1_DIS_s102
                                 + 1.28442 w_k1_FDX_s102
                                 + 0.883879 w_k1_IBM_s102
                                 + 1.18155 w_k1_INTC_s102
                                 + 0.988862 w_k1_KO_s102
                                 + 1.15494 w_k1_MAT_s102
                                 + 1.03037 w_k1_MSFT_s102
                                 + 1.11453 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.925952 w_k1_ADM_s102
                                 + 0.988149 w_k1_AXP_s102
                                 + 0.8367 w_k1_DIS_s102
                                 + 0.983729 w_k1_FDX_s102
                                 + 1.00201 w_k1_IBM_s102
                                 + 0.857124 w_k1_INTC_s102
                                 + 0.914311 w_k1_KO_s102 + 1.1289 w_k1_MAT_s102
                                 + 0.894636 w_k1_MSFT_s102
                                 + 0.96819 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.910825 w_k1_ADM_s102
                                 + 0.746162 w_k1_AXP_s102
                                 + 0.989972 w_k1_DIS_s102
                                 + 1.03334 w_k1_FDX_s102
                                 + 0.852715 w_k1_IBM_s102
                                 + 0.883299 w_k1_INTC_s102
                                 + 1.00235 w_k1_KO_s102
                                 + 0.793477 w_k1_MAT_s102
                                 + 0.902909 w_k1_MSFT_s102
                                 + 0.960619 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.647503 w_k1_ADM_s102
                                 + 0.811564 w_k1_AXP_s102
                                 + 0.650085 w_k1_DIS_s102
                                 + 0.810271 w_k1_FDX_s102
                                 + 0.65228 w_k1_IBM_s102
                                 + 0.886583 w_k1_INTC_s102
                                 + 0.743943 w_k1_KO_s102
                                 + 0.62196 w_k1_MAT_s102
                                 + 0.753451 w_k1_MSFT_s102
                                 + 0.765567 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.01776 w_k1_ADM_s102
                                 + 1.20461 w_k1_AXP_s102
                                 + 1.32997 w_k1_DIS_s102
                                 + 1.11533 w_k1_FDX_s102
                                 + 1.44491 w_k1_IBM_s102
                                 + 1.09969 w_k1_INTC_s102
                                 + 1.20054 w_k1_KO_s102 + 1.26719 w_k1_MAT_s102
                                 + 1.00366 w_k1_MSFT_s102
                                 + 1.07646 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.924592 w_k1_ADM_s103
                                 + 0.889158 w_k1_AXP_s103
                                 + 1.05415 w_k1_DIS_s103 + 1.0175 w_k1_FDX_s103
                                 + 0.783531 w_k1_IBM_s103
                                 + 0.872218 w_k1_INTC_s103
                                 + 0.964318 w_k1_KO_s103
                                 + 0.73826 w_k1_MAT_s103
                                 + 1.00679 w_k1_MSFT_s103
                                 + 0.923565 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.04004 w_k1_ADM_s103
                                 + 1.19942 w_k1_AXP_s103
                                 + 1.15892 w_k1_DIS_s103
                                 + 1.11815 w_k1_FDX_s103
                                 + 1.05705 w_k1_IBM_s103
                                 + 1.25289 w_k1_INTC_s103
                                 + 1.12296 w_k1_KO_s103
                                 + 0.810969 w_k1_MAT_s103
                                 + 1.07833 w_k1_MSFT_s103
                                 + 1.00429 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.839803 w_k1_ADM_s103
                                 + 0.7099 w_k1_AXP_s103
                                 + 0.748699 w_k1_DIS_s103
                                 + 0.787987 w_k1_FDX_s103
                                 + 1.2115 w_k1_IBM_s103
                                 + 0.772338 w_k1_INTC_s103
                                 + 0.954499 w_k1_KO_s103
                                 + 0.761144 w_k1_MAT_s103
                                 + 0.854029 w_k1_MSFT_s103
                                 + 0.83144 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.13505 w_k1_ADM_s103
                                 + 1.00758 w_k1_AXP_s103
                                 + 1.10904 w_k1_DIS_s103
                                 + 1.02399 w_k1_FDX_s103
                                 + 1.32083 w_k1_IBM_s103
                                 + 1.01707 w_k1_INTC_s103
                                 + 1.31358 w_k1_KO_s103
                                 + 0.759556 w_k1_MAT_s103
                                 + 0.957606 w_k1_MSFT_s103
                                 + 0.967897 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.29878 w_k1_ADM_s103
                                 + 1.32619 w_k1_AXP_s103
                                 + 1.70895 w_k1_DIS_s103
                                 + 1.32394 w_k1_FDX_s103
                                 + 1.10231 w_k1_IBM_s103 + 1.237 w_k1_INTC_s103
                                 + 1.51698 w_k1_KO_s103 + 1.16002 w_k1_MAT_s103
                                 + 1.47148 w_k1_MSFT_s103
                                 + 1.12284 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.841433 w_k1_ADM_s103
                                 + 0.56545 w_k1_AXP_s103
                                 + 0.529289 w_k1_DIS_s103
                                 + 0.714849 w_k1_FDX_s103
                                 + 0.755081 w_k1_IBM_s103
                                 + 0.671413 w_k1_INTC_s103
                                 + 0.672662 w_k1_KO_s103
                                 + 0.474046 w_k1_MAT_s103
                                 + 0.683098 w_k1_MSFT_s103
                                 + 0.76917 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.953013 w_k1_ADM_s103
                                 + 0.941729 w_k1_AXP_s103
                                 + 0.771556 w_k1_DIS_s103
                                 + 0.904123 w_k1_FDX_s103
                                 + 0.809198 w_k1_IBM_s103
                                 + 0.868787 w_k1_INTC_s103
                                 + 0.837475 w_k1_KO_s103
                                 + 0.76135 w_k1_MAT_s103
                                 + 0.81843 w_k1_MSFT_s103
                                 + 0.832929 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.21505 w_k1_ADM_s103
                                 + 1.41305 w_k1_AXP_s103
                                 + 1.35737 w_k1_DIS_s103 + 1.2423 w_k1_FDX_s103
                                 + 1.42373 w_k1_IBM_s103
                                 + 1.07686 w_k1_INTC_s103
                                 + 1.48059 w_k1_KO_s103 + 1.52129 w_k1_MAT_s103
                                 + 1.12892 w_k1_MSFT_s103
                                 + 1.21263 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.918846 w_k1_ADM_s104
                                 + 0.758557 w_k1_AXP_s104
                                 + 0.721653 w_k1_DIS_s104
                                 + 0.907871 w_k1_FDX_s104
                                 + 0.809446 w_k1_IBM_s104
                                 + 1.00741 w_k1_INTC_s104
                                 + 0.829631 w_k1_KO_s104
                                 + 0.60906 w_k1_MAT_s104
                                 + 0.836085 w_k1_MSFT_s104
                                 + 0.936533 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.29141 w_k1_ADM_s104
                                 + 1.10614 w_k1_AXP_s104
                                 + 1.13846 w_k1_DIS_s104
                                 + 1.09747 w_k1_FDX_s104 + 1.2015 w_k1_IBM_s104
                                 + 1.06923 w_k1_INTC_s104
                                 + 1.25011 w_k1_KO_s104 + 0.86078 w_k1_MAT_s104
                                 + 1.02286 w_k1_MSFT_s104
                                 + 1.10377 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.977363 w_k1_ADM_s104
                                 + 0.81431 w_k1_AXP_s104
                                 + 0.965144 w_k1_DIS_s104
                                 + 0.896139 w_k1_FDX_s104
                                 + 1.3198 w_k1_IBM_s104
                                 + 0.981992 w_k1_INTC_s104
                                 + 1.25441 w_k1_KO_s104
                                 + 0.943609 w_k1_MAT_s104
                                 + 0.905926 w_k1_MSFT_s104
                                 + 0.8855 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.00178 w_k1_ADM_s104
                                 + 0.807595 w_k1_AXP_s104
                                 + 0.68244 w_k1_DIS_s104
                                 + 0.735851 w_k1_FDX_s104
                                 + 1.01467 w_k1_IBM_s104
                                 + 0.829229 w_k1_INTC_s104
                                 + 0.856897 w_k1_KO_s104
                                 + 0.745802 w_k1_MAT_s104
                                 + 0.771526 w_k1_MSFT_s104
                                 + 0.90427 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.13014 w_k1_ADM_s104
                                 + 0.959117 w_k1_AXP_s104
                                 + 0.920542 w_k1_DIS_s104
                                 + 1.09063 w_k1_FDX_s104
                                 + 0.673937 w_k1_IBM_s104
                                 + 1.08415 w_k1_INTC_s104
                                 + 0.909678 w_k1_KO_s104
                                 + 0.739667 w_k1_MAT_s104
                                 + 0.984759 w_k1_MSFT_s104
                                 + 0.853809 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.71451 w_k1_ADM_s104
                                 + 0.593967 w_k1_AXP_s104
                                 + 0.601804 w_k1_DIS_s104
                                 + 0.725207 w_k1_FDX_s104
                                 + 0.642499 w_k1_IBM_s104
                                 + 0.771712 w_k1_INTC_s104
                                 + 0.66783 w_k1_KO_s104 + 0.47552 w_k1_MAT_s104
                                 + 0.769743 w_k1_MSFT_s104
                                 + 0.74682 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.35492 w_k1_ADM_s104
                                 + 1.49125 w_k1_AXP_s104
                                 + 1.67217 w_k1_DIS_s104
                                 + 1.30172 w_k1_FDX_s104
                                 + 1.35757 w_k1_IBM_s104
                                 + 1.11151 w_k1_INTC_s104
                                 + 1.47752 w_k1_KO_s104
                                 + 0.980683 w_k1_MAT_s104
                                 + 0.995335 w_k1_MSFT_s104
                                 + 1.14884 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.46322 w_k1_ADM_s104
                                 + 1.2277 w_k1_AXP_s104 + 1.20042 w_k1_DIS_s104
                                 + 1.2889 w_k1_FDX_s104
                                 + 0.810912 w_k1_IBM_s104
                                 + 1.05867 w_k1_INTC_s104
                                 + 1.08959 w_k1_KO_s104
                                 + 0.837393 w_k1_MAT_s104
                                 + 1.04302 w_k1_MSFT_s104
                                 + 0.996557 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.893807 w_k1_ADM_s105
                                 + 0.986654 w_k1_AXP_s105
                                 + 0.804261 w_k1_DIS_s105
                                 + 0.916608 w_k1_FDX_s105
                                 + 0.963502 w_k1_IBM_s105
                                 + 1.00223 w_k1_INTC_s105
                                 + 0.90891 w_k1_KO_s105 + 1.16141 w_k1_MAT_s105
                                 + 0.886504 w_k1_MSFT_s105
                                 + 0.877386 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.818785 w_k1_ADM_s105
                                 + 0.832863 w_k1_AXP_s105
                                 + 0.896012 w_k1_DIS_s105
                                 + 0.988044 w_k1_FDX_s105
                                 + 0.794996 w_k1_IBM_s105
                                 + 0.940034 w_k1_INTC_s105
                                 + 0.876004 w_k1_KO_s105
                                 + 0.634616 w_k1_MAT_s105
                                 + 0.966664 w_k1_MSFT_s105
                                 + 0.951416 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.738984 w_k1_ADM_s105
                                 + 0.824585 w_k1_AXP_s105
                                 + 0.866233 w_k1_DIS_s105
                                 + 0.824027 w_k1_FDX_s105
                                 + 1.18969 w_k1_IBM_s105
                                 + 1.00419 w_k1_INTC_s105
                                 + 0.90788 w_k1_KO_s105 + 1.20869 w_k1_MAT_s105
                                 + 0.953366 w_k1_MSFT_s105
                                 + 0.927521 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.867129 w_k1_ADM_s105
                                 + 0.993763 w_k1_AXP_s105
                                 + 1.07457 w_k1_DIS_s105
                                 + 1.01773 w_k1_FDX_s105 + 1.2181 w_k1_IBM_s105
                                 + 0.967176 w_k1_INTC_s105
                                 + 1.10786 w_k1_KO_s105 + 1.25089 w_k1_MAT_s105
                                 + 1.01201 w_k1_MSFT_s105
                                 + 1.05275 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.897582 w_k1_ADM_s105
                                 + 0.731899 w_k1_AXP_s105
                                 + 0.907192 w_k1_DIS_s105
                                 + 1.02494 w_k1_FDX_s105
                                 + 0.595838 w_k1_IBM_s105
                                 + 0.992809 w_k1_INTC_s105
                                 + 0.827031 w_k1_KO_s105
                                 + 0.63951 w_k1_MAT_s105
                                 + 1.00542 w_k1_MSFT_s105
                                 + 0.969607 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.19966 w_k1_ADM_s105
                                 + 1.58076 w_k1_AXP_s105
                                 + 1.45011 w_k1_DIS_s105
                                 + 1.18284 w_k1_FDX_s105
                                 + 1.55502 w_k1_IBM_s105
                                 + 1.27668 w_k1_INTC_s105
                                 + 1.56389 w_k1_KO_s105 + 1.70657 w_k1_MAT_s105
                                 + 1.17684 w_k1_MSFT_s105
                                 + 1.25886 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.05981 w_k1_ADM_s105
                                 + 1.154 w_k1_AXP_s105 + 1.25816 w_k1_DIS_s105
                                 + 1.21724 w_k1_FDX_s105
                                 + 1.12691 w_k1_IBM_s105
                                 + 1.29316 w_k1_INTC_s105
                                 + 1.25798 w_k1_KO_s105 + 1.4854 w_k1_MAT_s105
                                 + 1.30024 w_k1_MSFT_s105 + 1.1255 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.14196 w_k1_ADM_s105
                                 + 1.47142 w_k1_AXP_s105
                                 + 1.31352 w_k1_DIS_s105
                                 + 1.36213 w_k1_FDX_s105
                                 + 1.01517 w_k1_IBM_s105
                                 + 1.26604 w_k1_INTC_s105
                                 + 1.17835 w_k1_KO_s105 + 1.05707 w_k1_MAT_s105
                                 + 1.32809 w_k1_MSFT_s105
                                 + 1.20109 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.29861 w_k1_ADM_s106
                                 + 1.19353 w_k1_AXP_s106
                                 + 0.988744 w_k1_DIS_s106
                                 + 1.10784 w_k1_FDX_s106
                                 + 1.07969 w_k1_IBM_s106
                                 + 1.01621 w_k1_INTC_s106
                                 + 1.13226 w_k1_KO_s106 + 1.00471 w_k1_MAT_s106
                                 + 1.09233 w_k1_MSFT_s106
                                 + 1.09806 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.35718 w_k1_ADM_s106
                                 + 1.45712 w_k1_AXP_s106
                                 + 1.42839 w_k1_DIS_s106
                                 + 1.15175 w_k1_FDX_s106
                                 + 1.48319 w_k1_IBM_s106
                                 + 1.31941 w_k1_INTC_s106
                                 + 1.51039 w_k1_KO_s106 + 1.39687 w_k1_MAT_s106
                                 + 1.24346 w_k1_MSFT_s106
                                 + 1.14027 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.801493 w_k1_ADM_s106
                                 + 0.726479 w_k1_AXP_s106
                                 + 0.568532 w_k1_DIS_s106
                                 + 0.752549 w_k1_FDX_s106
                                 + 0.687171 w_k1_IBM_s106
                                 + 0.909966 w_k1_INTC_s106
                                 + 0.847152 w_k1_KO_s106
                                 + 0.733769 w_k1_MAT_s106
                                 + 0.865692 w_k1_MSFT_s106
                                 + 0.782008 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.12739 w_k1_ADM_s106
                                 + 1.59579 w_k1_AXP_s106
                                 + 1.26866 w_k1_DIS_s106
                                 + 1.28731 w_k1_FDX_s106 + 0.7768 w_k1_IBM_s106
                                 + 1.18674 w_k1_INTC_s106
                                 + 1.04179 w_k1_KO_s106
                                 + 0.914431 w_k1_MAT_s106
                                 + 1.09686 w_k1_MSFT_s106
                                 + 1.06465 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.917247 w_k1_ADM_s106
                                 + 0.880079 w_k1_AXP_s106
                                 + 0.847954 w_k1_DIS_s106
                                 + 0.916391 w_k1_FDX_s106
                                 + 0.808665 w_k1_IBM_s106
                                 + 1.12016 w_k1_INTC_s106
                                 + 0.887915 w_k1_KO_s106
                                 + 0.731718 w_k1_MAT_s106
                                 + 1.01476 w_k1_MSFT_s106
                                 + 0.940819 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.02373 w_k1_ADM_s106
                                 + 0.964915 w_k1_AXP_s106
                                 + 0.886523 w_k1_DIS_s106
                                 + 1.07047 w_k1_FDX_s106
                                 + 0.579545 w_k1_IBM_s106
                                 + 0.970856 w_k1_INTC_s106
                                 + 0.786228 w_k1_KO_s106
                                 + 0.515971 w_k1_MAT_s106
                                 + 0.904701 w_k1_MSFT_s106
                                 + 0.944445 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.968674 w_k1_ADM_s106
                                 + 1.04071 w_k1_AXP_s106
                                 + 0.89579 w_k1_DIS_s106
                                 + 0.845806 w_k1_FDX_s106
                                 + 1.21013 w_k1_IBM_s106
                                 + 1.09499 w_k1_INTC_s106
                                 + 1.20029 w_k1_KO_s106 + 1.3103 w_k1_MAT_s106
                                 + 1.0707 w_k1_MSFT_s106 + 1.02625 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.14301 w_k1_ADM_s106
                                 + 1.23819 w_k1_AXP_s106
                                 + 1.23138 w_k1_DIS_s106 + 1.1469 w_k1_FDX_s106
                                 + 0.932767 w_k1_IBM_s106
                                 + 1.36297 w_k1_INTC_s106
                                 + 1.13836 w_k1_KO_s106 + 1.05752 w_k1_MAT_s106
                                 + 1.30613 w_k1_MSFT_s106
                                 + 1.04952 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.33209 w_k1_ADM_s107
                                 + 1.32229 w_k1_AXP_s107
                                 + 1.31205 w_k1_DIS_s107
                                 + 1.18832 w_k1_FDX_s107
                                 + 0.949118 w_k1_IBM_s107
                                 + 1.02619 w_k1_INTC_s107
                                 + 1.03294 w_k1_KO_s107 + 1.1407 w_k1_MAT_s107
                                 + 1.08622 w_k1_MSFT_s107
                                 + 0.974256 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.26797 w_k1_ADM_s107
                                 + 1.07078 w_k1_AXP_s107
                                 + 1.52426 w_k1_DIS_s107
                                 + 1.04713 w_k1_FDX_s107
                                 + 1.73482 w_k1_IBM_s107
                                 + 0.900647 w_k1_INTC_s107
                                 + 1.25604 w_k1_KO_s107 + 1.43506 w_k1_MAT_s107
                                 + 0.97405 w_k1_MSFT_s107 + 1.1199 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.79574 w_k1_ADM_s107
                                 + 0.808768 w_k1_AXP_s107
                                 + 0.910563 w_k1_DIS_s107
                                 + 0.814038 w_k1_FDX_s107
                                 + 1.25672 w_k1_IBM_s107
                                 + 0.813278 w_k1_INTC_s107
                                 + 1.14717 w_k1_KO_s107 + 1.3305 w_k1_MAT_s107
                                 + 0.949121 w_k1_MSFT_s107
                                 + 0.927602 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.29215 w_k1_ADM_s107
                                 + 0.918522 w_k1_AXP_s107
                                 + 1.01581 w_k1_DIS_s107 + 1.073 w_k1_FDX_s107
                                 + 0.785869 w_k1_IBM_s107
                                 + 0.859359 w_k1_INTC_s107
                                 + 0.878733 w_k1_KO_s107
                                 + 0.956137 w_k1_MAT_s107
                                 + 1.00052 w_k1_MSFT_s107
                                 + 0.933312 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.61599 w_k1_ADM_s107
                                 + 1.33267 w_k1_AXP_s107
                                 + 1.58892 w_k1_DIS_s107
                                 + 1.54952 w_k1_FDX_s107
                                 + 0.815597 w_k1_IBM_s107
                                 + 1.08367 w_k1_INTC_s107
                                 + 1.15241 w_k1_KO_s107 + 1.1765 w_k1_MAT_s107
                                 + 1.24783 w_k1_MSFT_s107
                                 + 1.25881 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.81013 w_k1_ADM_s107
                                 + 0.638984 w_k1_AXP_s107
                                 + 0.62006 w_k1_DIS_s107
                                 + 0.711574 w_k1_FDX_s107
                                 + 0.806904 w_k1_IBM_s107
                                 + 0.705161 w_k1_INTC_s107
                                 + 0.656484 w_k1_KO_s107
                                 + 0.597882 w_k1_MAT_s107
                                 + 0.701353 w_k1_MSFT_s107
                                 + 0.654004 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.818369 w_k1_ADM_s107
                                 + 0.613789 w_k1_AXP_s107
                                 + 0.843668 w_k1_DIS_s107
                                 + 0.818269 w_k1_FDX_s107
                                 + 0.928675 w_k1_IBM_s107
                                 + 0.838163 w_k1_INTC_s107
                                 + 0.917129 w_k1_KO_s107
                                 + 0.85265 w_k1_MAT_s107
                                 + 0.833083 w_k1_MSFT_s107
                                 + 0.914227 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.724396 w_k1_ADM_s107
                                 + 0.726008 w_k1_AXP_s107
                                 + 0.676791 w_k1_DIS_s107
                                 + 0.849973 w_k1_FDX_s107
                                 + 0.566083 w_k1_IBM_s107
                                 + 0.914377 w_k1_INTC_s107
                                 + 0.597556 w_k1_KO_s107
                                 + 0.538583 w_k1_MAT_s107
                                 + 0.767149 w_k1_MSFT_s107
                                 + 0.648398 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.829907 w_k1_ADM_s110
                                 + 0.689443 w_k1_AXP_s110
                                 + 0.65891 w_k1_DIS_s110
                                 + 0.723668 w_k1_FDX_s110
                                 + 1.07588 w_k1_IBM_s110
                                 + 0.821909 w_k1_INTC_s110
                                 + 0.924449 w_k1_KO_s110
                                 + 0.722441 w_k1_MAT_s110
                                 + 0.709338 w_k1_MSFT_s110
                                 + 0.826466 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.37144 w_k1_ADM_s110
                                 + 1.27852 w_k1_AXP_s110
                                 + 1.16845 w_k1_DIS_s110
                                 + 1.23445 w_k1_FDX_s110
                                 + 0.991731 w_k1_IBM_s110
                                 + 1.19843 w_k1_INTC_s110
                                 + 0.913187 w_k1_KO_s110
                                 + 0.59777 w_k1_MAT_s110
                                 + 1.0134 w_k1_MSFT_s110 + 1.11972 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.38746 w_k1_ADM_s110
                                 + 1.43339 w_k1_AXP_s110 + 1.6732 w_k1_DIS_s110
                                 + 1.29942 w_k1_FDX_s110
                                 + 1.00883 w_k1_IBM_s110
                                 + 1.19922 w_k1_INTC_s110
                                 + 1.30812 w_k1_KO_s110
                                 + 0.746852 w_k1_MAT_s110
                                 + 1.26529 w_k1_MSFT_s110
                                 + 1.21779 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.658744 w_k1_ADM_s110
                                 + 0.759305 w_k1_AXP_s110
                                 + 0.590489 w_k1_DIS_s110
                                 + 0.728721 w_k1_FDX_s110
                                 + 0.863456 w_k1_IBM_s110
                                 + 0.897551 w_k1_INTC_s110
                                 + 0.679625 w_k1_KO_s110
                                 + 0.760564 w_k1_MAT_s110
                                 + 0.684244 w_k1_MSFT_s110
                                 + 0.763239 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.10838 w_k1_ADM_s110
                                 + 1.4993 w_k1_AXP_s110 + 1.26752 w_k1_DIS_s110
                                 + 1.11762 w_k1_FDX_s110
                                 + 1.66125 w_k1_IBM_s110
                                 + 1.20932 w_k1_INTC_s110 + 1.1922 w_k1_KO_s110
                                 + 1.33288 w_k1_MAT_s110
                                 + 1.0566 w_k1_MSFT_s110 + 1.12156 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.944723 w_k1_ADM_s110
                                 + 1.17535 w_k1_AXP_s110
                                 + 1.11031 w_k1_DIS_s110
                                 + 1.02586 w_k1_FDX_s110
                                 + 1.11149 w_k1_IBM_s110
                                 + 1.23689 w_k1_INTC_s110
                                 + 1.18498 w_k1_KO_s110
                                 + 0.880715 w_k1_MAT_s110
                                 + 1.16278 w_k1_MSFT_s110
                                 + 0.984279 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.922725 w_k1_ADM_s110
                                 + 0.814602 w_k1_AXP_s110
                                 + 0.773659 w_k1_DIS_s110
                                 + 0.905702 w_k1_FDX_s110
                                 + 0.737869 w_k1_IBM_s110
                                 + 1.07434 w_k1_INTC_s110
                                 + 0.750495 w_k1_KO_s110
                                 + 0.558574 w_k1_MAT_s110
                                 + 0.909102 w_k1_MSFT_s110 + 0.854 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.958043 w_k1_ADM_s110
                                 + 1.07284 w_k1_AXP_s110
                                 + 0.892623 w_k1_DIS_s110
                                 + 0.888832 w_k1_FDX_s110
                                 + 1.41937 w_k1_IBM_s110
                                 + 1.01704 w_k1_INTC_s110
                                 + 1.15029 w_k1_KO_s110 + 1.54562 w_k1_MAT_s110
                                 + 0.877818 w_k1_MSFT_s110
                                 + 0.943475 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.982366 w_k1_ADM_s111
                                 + 0.944602 w_k1_AXP_s111
                                 + 1.16615 w_k1_DIS_s111
                                 + 1.02052 w_k1_FDX_s111
                                 + 0.755294 w_k1_IBM_s111
                                 + 0.967118 w_k1_INTC_s111
                                 + 0.948448 w_k1_KO_s111
                                 + 0.640506 w_k1_MAT_s111
                                 + 0.959555 w_k1_MSFT_s111
                                 + 0.956854 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.02034 w_k1_ADM_s111
                                 + 0.943105 w_k1_AXP_s111
                                 + 0.839483 w_k1_DIS_s111
                                 + 0.922738 w_k1_FDX_s111
                                 + 0.897534 w_k1_IBM_s111
                                 + 0.90844 w_k1_INTC_s111
                                 + 0.968869 w_k1_KO_s111
                                 + 1.02697 w_k1_MAT_s111
                                 + 0.876023 w_k1_MSFT_s111
                                 + 0.894276 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.859493 w_k1_ADM_s111
                                 + 0.848353 w_k1_AXP_s111
                                 + 0.928147 w_k1_DIS_s111
                                 + 0.779431 w_k1_FDX_s111
                                 + 1.09414 w_k1_IBM_s111
                                 + 0.900099 w_k1_INTC_s111
                                 + 1.04961 w_k1_KO_s111 + 1.31299 w_k1_MAT_s111
                                 + 0.851719 w_k1_MSFT_s111
                                 + 0.869419 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.869793 w_k1_ADM_s111
                                 + 0.931871 w_k1_AXP_s111
                                 + 1.25439 w_k1_DIS_s111
                                 + 0.99964 w_k1_FDX_s111
                                 + 1.46551 w_k1_IBM_s111
                                 + 1.0722 w_k1_INTC_s111 + 1.32161 w_k1_KO_s111
                                 + 1.45868 w_k1_MAT_s111
                                 + 1.15424 w_k1_MSFT_s111
                                 + 1.05841 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.0021 w_k1_ADM_s111
                                 + 1.25932 w_k1_AXP_s111
                                 + 1.83119 w_k1_DIS_s111
                                 + 1.22759 w_k1_FDX_s111
                                 + 1.47649 w_k1_IBM_s111
                                 + 1.24673 w_k1_INTC_s111
                                 + 1.50705 w_k1_KO_s111 + 1.47522 w_k1_MAT_s111
                                 + 1.39082 w_k1_MSFT_s111
                                 + 1.21637 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.647768 w_k1_ADM_s111
                                 + 0.557546 w_k1_AXP_s111
                                 + 0.731609 w_k1_DIS_s111
                                 + 0.768102 w_k1_FDX_s111
                                 + 0.76885 w_k1_IBM_s111
                                 + 0.745967 w_k1_INTC_s111
                                 + 0.887311 w_k1_KO_s111
                                 + 0.730146 w_k1_MAT_s111
                                 + 0.900299 w_k1_MSFT_s111
                                 + 0.84044 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.37745 w_k1_ADM_s111
                                 + 1.51169 w_k1_AXP_s111 + 1.4439 w_k1_DIS_s111
                                 + 1.35389 w_k1_FDX_s111
                                 + 0.905037 w_k1_IBM_s111
                                 + 1.11639 w_k1_INTC_s111
                                 + 1.06168 w_k1_KO_s111 + 1.25752 w_k1_MAT_s111
                                 + 1.24183 w_k1_MSFT_s111
                                 + 1.09116 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.09447 w_k1_ADM_s111
                                 + 1.37911 w_k1_AXP_s111
                                 + 1.18056 w_k1_DIS_s111
                                 + 1.11337 w_k1_FDX_s111
                                 + 1.14383 w_k1_IBM_s111
                                 + 1.21573 w_k1_INTC_s111
                                 + 1.17279 w_k1_KO_s111 + 1.23223 w_k1_MAT_s111
                                 + 1.02873 w_k1_MSFT_s111
                                 + 1.06503 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.07705 w_k1_ADM_s112
                                 + 1.15566 w_k1_AXP_s112
                                 + 1.27542 w_k1_DIS_s112
                                 + 1.16649 w_k1_FDX_s112
                                 + 1.20471 w_k1_IBM_s112
                                 + 1.00602 w_k1_INTC_s112
                                 + 1.16588 w_k1_KO_s112 + 1.06335 w_k1_MAT_s112
                                 + 1.1838 w_k1_MSFT_s112
                                 + 0.991883 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.01075 w_k1_ADM_s112
                                 + 1.05477 w_k1_AXP_s112
                                 + 1.06951 w_k1_DIS_s112
                                 + 1.02388 w_k1_FDX_s112
                                 + 0.888796 w_k1_IBM_s112
                                 + 1.05058 w_k1_INTC_s112
                                 + 0.975327 w_k1_KO_s112
                                 + 1.09815 w_k1_MAT_s112
                                 + 1.22646 w_k1_MSFT_s112
                                 + 0.90239 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.36574 w_k1_ADM_s112
                                 + 1.58086 w_k1_AXP_s112
                                 + 1.70062 w_k1_DIS_s112
                                 + 1.44686 w_k1_FDX_s112
                                 + 1.06753 w_k1_IBM_s112
                                 + 1.29212 w_k1_INTC_s112
                                 + 1.50714 w_k1_KO_s112 + 1.25031 w_k1_MAT_s112
                                 + 1.53486 w_k1_MSFT_s112
                                 + 1.32654 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.80299 w_k1_ADM_s112
                                 + 1.01187 w_k1_AXP_s112
                                 + 1.09121 w_k1_DIS_s112
                                 + 0.946082 w_k1_FDX_s112
                                 + 1.20524 w_k1_IBM_s112
                                 + 0.94647 w_k1_INTC_s112 + 1.1487 w_k1_KO_s112
                                 + 1.27748 w_k1_MAT_s112
                                 + 1.11467 w_k1_MSFT_s112
                                 + 0.938512 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.878936 w_k1_ADM_s112
                                 + 0.908579 w_k1_AXP_s112
                                 + 0.938996 w_k1_DIS_s112
                                 + 0.952737 w_k1_FDX_s112
                                 + 0.711209 w_k1_IBM_s112
                                 + 0.938749 w_k1_INTC_s112
                                 + 0.84966 w_k1_KO_s112
                                 + 0.606997 w_k1_MAT_s112
                                 + 0.905025 w_k1_MSFT_s112
                                 + 0.840543 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.663355 w_k1_ADM_s112
                                 + 1.08832 w_k1_AXP_s112
                                 + 1.15112 w_k1_DIS_s112
                                 + 0.979213 w_k1_FDX_s112 + 1.603 w_k1_IBM_s112
                                 + 0.924852 w_k1_INTC_s112
                                 + 1.27753 w_k1_KO_s112 + 1.4551 w_k1_MAT_s112
                                 + 1.04699 w_k1_MSFT_s112
                                 + 0.941604 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.78849 w_k1_ADM_s112
                                 + 0.922339 w_k1_AXP_s112
                                 + 0.671606 w_k1_DIS_s112
                                 + 0.793343 w_k1_FDX_s112
                                 + 1.15563 w_k1_IBM_s112
                                 + 0.837708 w_k1_INTC_s112
                                 + 0.99767 w_k1_KO_s112 + 1.14776 w_k1_MAT_s112
                                 + 0.813558 w_k1_MSFT_s112
                                 + 0.877984 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.831162 w_k1_ADM_s112
                                 + 0.587822 w_k1_AXP_s112
                                 + 0.63738 w_k1_DIS_s112
                                 + 0.652963 w_k1_FDX_s112
                                 + 0.920977 w_k1_IBM_s112
                                 + 0.743676 w_k1_INTC_s112
                                 + 0.762853 w_k1_KO_s112
                                 + 0.655348 w_k1_MAT_s112
                                 + 0.745407 w_k1_MSFT_s112
                                 + 0.763167 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.715392 w_k1_ADM_s113
                                 + 1.01435 w_k1_AXP_s113
                                 + 0.974796 w_k1_DIS_s113
                                 + 1.05903 w_k1_FDX_s113
                                 + 1.07637 w_k1_IBM_s113
                                 + 1.10935 w_k1_INTC_s113
                                 + 1.18254 w_k1_KO_s113 + 1.12261 w_k1_MAT_s113
                                 + 1.12104 w_k1_MSFT_s113
                                 + 1.02997 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.810883 w_k1_ADM_s113
                                 + 0.752765 w_k1_AXP_s113
                                 + 0.833716 w_k1_DIS_s113
                                 + 0.923864 w_k1_FDX_s113
                                 + 0.866464 w_k1_IBM_s113
                                 + 1.04945 w_k1_INTC_s113
                                 + 0.838864 w_k1_KO_s113
                                 + 0.38595 w_k1_MAT_s113
                                 + 0.953793 w_k1_MSFT_s113
                                 + 0.864725 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.58039 w_k1_ADM_s113
                                 + 0.689705 w_k1_AXP_s113
                                 + 0.644067 w_k1_DIS_s113
                                 + 0.709885 w_k1_FDX_s113
                                 + 0.846956 w_k1_IBM_s113
                                 + 0.775299 w_k1_INTC_s113
                                 + 0.817354 w_k1_KO_s113
                                 + 0.509446 w_k1_MAT_s113
                                 + 0.712124 w_k1_MSFT_s113
                                 + 0.648259 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.07605 w_k1_ADM_s113
                                 + 1.56827 w_k1_AXP_s113
                                 + 1.62908 w_k1_DIS_s113
                                 + 1.55123 w_k1_FDX_s113
                                 + 1.11476 w_k1_IBM_s113
                                 + 1.4173 w_k1_INTC_s113 + 1.40306 w_k1_KO_s113
                                 + 1.00625 w_k1_MAT_s113
                                 + 1.44788 w_k1_MSFT_s113
                                 + 1.15485 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.04007 w_k1_ADM_s113
                                 + 1.24165 w_k1_AXP_s113 + 1.2505 w_k1_DIS_s113
                                 + 1.21828 w_k1_FDX_s113
                                 + 1.60325 w_k1_IBM_s113
                                 + 1.05523 w_k1_INTC_s113
                                 + 1.29262 w_k1_KO_s113 + 1.48856 w_k1_MAT_s113
                                 + 1.07666 w_k1_MSFT_s113
                                 + 1.23876 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.591515 w_k1_ADM_s113
                                 + 0.867417 w_k1_AXP_s113
                                 + 0.686637 w_k1_DIS_s113
                                 + 0.838425 w_k1_FDX_s113
                                 + 1.03638 w_k1_IBM_s113
                                 + 0.652209 w_k1_INTC_s113
                                 + 0.81644 w_k1_KO_s113
                                 + 0.893039 w_k1_MAT_s113
                                 + 0.683643 w_k1_MSFT_s113
                                 + 0.902588 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.91414 w_k1_ADM_s113
                                 + 0.839673 w_k1_AXP_s113
                                 + 1.04873 w_k1_DIS_s113
                                 + 1.12668 w_k1_FDX_s113
                                 + 0.681296 w_k1_IBM_s113
                                 + 0.920686 w_k1_INTC_s113
                                 + 0.834223 w_k1_KO_s113
                                 + 0.491118 w_k1_MAT_s113
                                 + 1.05334 w_k1_MSFT_s113
                                 + 0.974036 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.79581 w_k1_ADM_s113
                                 + 1.29496 w_k1_AXP_s113 + 1.3068 w_k1_DIS_s113
                                 + 1.31113 w_k1_FDX_s113
                                 + 1.00755 w_k1_IBM_s113
                                 + 1.12192 w_k1_INTC_s113
                                 + 1.13363 w_k1_KO_s113
                                 + 0.908407 w_k1_MAT_s113
                                 + 1.23814 w_k1_MSFT_s113
                                 + 1.06531 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.0127 w_k1_ADM_s114
                                 + 0.841034 w_k1_AXP_s114
                                 + 0.902691 w_k1_DIS_s114
                                 + 1.03521 w_k1_FDX_s114
                                 + 0.790151 w_k1_IBM_s114
                                 + 1.01066 w_k1_INTC_s114
                                 + 0.993327 w_k1_KO_s114
                                 + 0.645456 w_k1_MAT_s114
                                 + 1.00023 w_k1_MSFT_s114
                                 + 0.979473 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.818461 w_k1_ADM_s114
                                 + 0.800007 w_k1_AXP_s114 + 1.124 w_k1_DIS_s114
                                 + 0.877881 w_k1_FDX_s114
                                 + 1.46366 w_k1_IBM_s114
                                 + 0.839134 w_k1_INTC_s114
                                 + 1.30382 w_k1_KO_s114 + 1.01448 w_k1_MAT_s114
                                 + 0.869498 w_k1_MSFT_s114
                                 + 1.04402 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.95056 w_k1_ADM_s114
                                 + 1.05763 w_k1_AXP_s114
                                 + 1.00613 w_k1_DIS_s114
                                 + 0.918965 w_k1_FDX_s114
                                 + 1.1243 w_k1_IBM_s114
                                 + 1.03966 w_k1_INTC_s114
                                 + 1.11431 w_k1_KO_s114
                                 + 0.825114 w_k1_MAT_s114
                                 + 0.884752 w_k1_MSFT_s114
                                 + 1.18928 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.07251 w_k1_ADM_s114
                                 + 0.99623 w_k1_AXP_s114
                                 + 1.41379 w_k1_DIS_s114
                                 + 1.20594 w_k1_FDX_s114
                                 + 1.07458 w_k1_IBM_s114
                                 + 1.1193 w_k1_INTC_s114 + 1.28946 w_k1_KO_s114
                                 + 1.04031 w_k1_MAT_s114
                                 + 1.21103 w_k1_MSFT_s114
                                 + 1.20454 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.10226 w_k1_ADM_s114
                                 + 1.14054 w_k1_AXP_s114
                                 + 0.895647 w_k1_DIS_s114
                                 + 1.00441 w_k1_FDX_s114
                                 + 1.50484 w_k1_IBM_s114
                                 + 0.910541 w_k1_INTC_s114
                                 + 1.14976 w_k1_KO_s114 + 1.09343 w_k1_MAT_s114
                                 + 0.768282 w_k1_MSFT_s114
                                 + 1.14001 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.857039 w_k1_ADM_s114
                                 + 0.714593 w_k1_AXP_s114
                                 + 0.651765 w_k1_DIS_s114
                                 + 0.769441 w_k1_FDX_s114
                                 + 0.855025 w_k1_IBM_s114
                                 + 0.631536 w_k1_INTC_s114
                                 + 0.814266 w_k1_KO_s114
                                 + 0.570197 w_k1_MAT_s114
                                 + 0.704341 w_k1_MSFT_s114
                                 + 0.890269 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.917725 w_k1_ADM_s114
                                 + 0.882587 w_k1_AXP_s114
                                 + 0.923654 w_k1_DIS_s114
                                 + 0.936914 w_k1_FDX_s114
                                 + 0.669291 w_k1_IBM_s114
                                 + 0.879012 w_k1_INTC_s114
                                 + 0.813431 w_k1_KO_s114
                                 + 0.574477 w_k1_MAT_s114
                                 + 0.878245 w_k1_MSFT_s114
                                 + 0.917293 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.32998 w_k1_ADM_s114
                                 + 1.3916 w_k1_AXP_s114 + 1.23691 w_k1_DIS_s114
                                 + 1.24818 w_k1_FDX_s114
                                 + 1.16977 w_k1_IBM_s114
                                 + 0.963434 w_k1_INTC_s114
                                 + 1.36543 w_k1_KO_s114 + 1.27837 w_k1_MAT_s114
                                 + 1.01438 w_k1_MSFT_s114
                                 + 1.24498 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.19975 w_k1_ADM_s115
                                 + 1.4732 w_k1_AXP_s115 + 1.77841 w_k1_DIS_s115
                                 + 1.30114 w_k1_FDX_s115
                                 + 1.26538 w_k1_IBM_s115
                                 + 1.36166 w_k1_INTC_s115
                                 + 1.38499 w_k1_KO_s115 + 1.54294 w_k1_MAT_s115
                                 + 1.38355 w_k1_MSFT_s115
                                 + 1.19332 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.733422 w_k1_ADM_s115
                                 + 0.824971 w_k1_AXP_s115
                                 + 0.92506 w_k1_DIS_s115
                                 + 0.949375 w_k1_FDX_s115
                                 + 1.08357 w_k1_IBM_s115
                                 + 1.06601 w_k1_INTC_s115
                                 + 1.03161 w_k1_KO_s115
                                 + 0.687229 w_k1_MAT_s115
                                 + 0.911684 w_k1_MSFT_s115
                                 + 1.01171 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.1136 w_k1_ADM_s115
                                 + 1.30566 w_k1_AXP_s115
                                 + 1.22452 w_k1_DIS_s115
                                 + 1.30152 w_k1_FDX_s115
                                 + 1.15376 w_k1_IBM_s115
                                 + 1.30188 w_k1_INTC_s115
                                 + 1.27784 w_k1_KO_s115 + 1.47212 w_k1_MAT_s115
                                 + 1.17833 w_k1_MSFT_s115
                                 + 1.16945 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.982033 w_k1_ADM_s115
                                 + 1.0578 w_k1_AXP_s115
                                 + 0.758159 w_k1_DIS_s115
                                 + 0.786647 w_k1_FDX_s115
                                 + 1.38538 w_k1_IBM_s115
                                 + 0.847031 w_k1_INTC_s115
                                 + 1.0904 w_k1_KO_s115 + 1.61805 w_k1_MAT_s115
                                 + 0.765123 w_k1_MSFT_s115
                                 + 0.936321 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.960322 w_k1_ADM_s115
                                 + 0.991971 w_k1_AXP_s115
                                 + 1.16481 w_k1_DIS_s115
                                 + 1.18319 w_k1_FDX_s115
                                 + 0.756767 w_k1_IBM_s115
                                 + 1.13764 w_k1_INTC_s115
                                 + 1.01271 w_k1_KO_s115
                                 + 0.906298 w_k1_MAT_s115
                                 + 1.18919 w_k1_MSFT_s115
                                 + 1.06873 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.0721 w_k1_ADM_s115
                                 + 1.14024 w_k1_AXP_s115
                                 + 0.996146 w_k1_DIS_s115
                                 + 1.14928 w_k1_FDX_s115
                                 + 0.895132 w_k1_IBM_s115
                                 + 1.12106 w_k1_INTC_s115 + 1.0205 w_k1_KO_s115
                                 + 1.19311 w_k1_MAT_s115
                                 + 0.964827 w_k1_MSFT_s115
                                 + 0.942072 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.83061 w_k1_ADM_s115
                                 + 0.845425 w_k1_AXP_s115
                                 + 0.728607 w_k1_DIS_s115
                                 + 0.8459 w_k1_FDX_s115
                                 + 0.750555 w_k1_IBM_s115
                                 + 0.905944 w_k1_INTC_s115
                                 + 0.802167 w_k1_KO_s115
                                 + 0.785011 w_k1_MAT_s115
                                 + 0.764732 w_k1_MSFT_s115
                                 + 0.76919 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.0889 w_k1_ADM_s115
                                 + 1.15335 w_k1_AXP_s115
                                 + 1.16255 w_k1_DIS_s115
                                 + 1.00325 w_k1_FDX_s115
                                 + 1.46471 w_k1_IBM_s115
                                 + 0.97699 w_k1_INTC_s115
                                 + 1.32604 w_k1_KO_s115 + 1.54631 w_k1_MAT_s115
                                 + 1.07623 w_k1_MSFT_s115
                                 + 1.18863 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.21349 w_k1_ADM_s116
                                 + 1.3761 w_k1_AXP_s116 + 1.54504 w_k1_DIS_s116
                                 + 1.522 w_k1_FDX_s116 + 1.0784 w_k1_IBM_s116
                                 + 1.08533 w_k1_INTC_s116
                                 + 1.16314 w_k1_KO_s116 + 1.25373 w_k1_MAT_s116
                                 + 1.42879 w_k1_MSFT_s116
                                 + 1.26684 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.999365 w_k1_ADM_s116
                                 + 0.690167 w_k1_AXP_s116
                                 + 0.777815 w_k1_DIS_s116
                                 + 0.869877 w_k1_FDX_s116
                                 + 0.85931 w_k1_IBM_s116
                                 + 0.875861 w_k1_INTC_s116
                                 + 0.852881 w_k1_KO_s116
                                 + 0.813387 w_k1_MAT_s116
                                 + 0.853974 w_k1_MSFT_s116
                                 + 0.952692 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.659778 w_k1_ADM_s116
                                 + 0.5934 w_k1_AXP_s116
                                 + 0.565149 w_k1_DIS_s116
                                 + 0.680146 w_k1_FDX_s116
                                 + 1.00862 w_k1_IBM_s116
                                 + 0.661692 w_k1_INTC_s116
                                 + 0.726194 w_k1_KO_s116
                                 + 1.10904 w_k1_MAT_s116
                                 + 0.681836 w_k1_MSFT_s116
                                 + 0.905693 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.00018 w_k1_ADM_s116
                                 + 0.9428 w_k1_AXP_s116 + 1.29602 w_k1_DIS_s116
                                 + 1.32018 w_k1_FDX_s116
                                 + 0.703798 w_k1_IBM_s116
                                 + 1.07088 w_k1_INTC_s116
                                 + 0.980428 w_k1_KO_s116
                                 + 0.582435 w_k1_MAT_s116
                                 + 1.14278 w_k1_MSFT_s116
                                 + 1.15645 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.829496 w_k1_ADM_s116
                                 + 0.797053 w_k1_AXP_s116
                                 + 1.22645 w_k1_DIS_s116
                                 + 1.07099 w_k1_FDX_s116
                                 + 1.29641 w_k1_IBM_s116
                                 + 0.961449 w_k1_INTC_s116
                                 + 1.11853 w_k1_KO_s116 + 1.45594 w_k1_MAT_s116
                                 + 1.1339 w_k1_MSFT_s116 + 1.08719 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.27341 w_k1_ADM_s116
                                 + 1.35341 w_k1_AXP_s116
                                 + 1.15436 w_k1_DIS_s116 + 1.0805 w_k1_FDX_s116
                                 + 1.23309 w_k1_IBM_s116
                                 + 1.02686 w_k1_INTC_s116
                                 + 1.18971 w_k1_KO_s116 + 1.39146 w_k1_MAT_s116
                                 + 1.01739 w_k1_MSFT_s116 + 1.1597 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.922071 w_k1_ADM_s116
                                 + 0.872639 w_k1_AXP_s116
                                 + 0.877169 w_k1_DIS_s116
                                 + 1.01011 w_k1_FDX_s116
                                 + 0.875496 w_k1_IBM_s116
                                 + 0.863526 w_k1_INTC_s116
                                 + 0.88818 w_k1_KO_s116
                                 + 0.997061 w_k1_MAT_s116
                                 + 0.856005 w_k1_MSFT_s116
                                 + 0.903344 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.02308 w_k1_ADM_s116
                                 + 0.863781 w_k1_AXP_s116
                                 + 0.910833 w_k1_DIS_s116
                                 + 1.09223 w_k1_FDX_s116
                                 + 0.723441 w_k1_IBM_s116
                                 + 0.817094 w_k1_INTC_s116
                                 + 0.887688 w_k1_KO_s116
                                 + 0.996734 w_k1_MAT_s116
                                 + 0.947497 w_k1_MSFT_s116
                                 + 1.06585 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.87342 w_k1_ADM_s117
                                 + 0.856878 w_k1_AXP_s117
                                 + 0.828562 w_k1_DIS_s117
                                 + 1.00575 w_k1_FDX_s117
                                 + 0.776761 w_k1_IBM_s117
                                 + 0.934463 w_k1_INTC_s117
                                 + 0.897217 w_k1_KO_s117
                                 + 0.764633 w_k1_MAT_s117
                                 + 0.883444 w_k1_MSFT_s117
                                 + 0.90942 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.888573 w_k1_ADM_s117
                                 + 0.746885 w_k1_AXP_s117
                                 + 0.995048 w_k1_DIS_s117
                                 + 0.938998 w_k1_FDX_s117
                                 + 0.734107 w_k1_IBM_s117
                                 + 0.840127 w_k1_INTC_s117
                                 + 0.91192 w_k1_KO_s117
                                 + 0.691877 w_k1_MAT_s117
                                 + 0.844894 w_k1_MSFT_s117
                                 + 0.940132 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.942536 w_k1_ADM_s117
                                 + 1.23486 w_k1_AXP_s117
                                 + 1.14389 w_k1_DIS_s117 + 1.0707 w_k1_FDX_s117
                                 + 1.22325 w_k1_IBM_s117
                                 + 1.00079 w_k1_INTC_s117
                                 + 1.33502 w_k1_KO_s117 + 1.26362 w_k1_MAT_s117
                                 + 1.09733 w_k1_MSFT_s117
                                 + 1.08209 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.15513 w_k1_ADM_s117
                                 + 1.20315 w_k1_AXP_s117
                                 + 1.40543 w_k1_DIS_s117
                                 + 1.21287 w_k1_FDX_s117
                                 + 1.23218 w_k1_IBM_s117
                                 + 1.20288 w_k1_INTC_s117
                                 + 1.32704 w_k1_KO_s117 + 1.32013 w_k1_MAT_s117
                                 + 1.23993 w_k1_MSFT_s117
                                 + 1.15684 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.33477 w_k1_ADM_s117
                                 + 1.70517 w_k1_AXP_s117 + 1.6312 w_k1_DIS_s117
                                 + 1.53761 w_k1_FDX_s117
                                 + 0.97746 w_k1_IBM_s117
                                 + 1.66002 w_k1_INTC_s117
                                 + 1.29181 w_k1_KO_s117 + 1.41013 w_k1_MAT_s117
                                 + 1.51035 w_k1_MSFT_s117
                                 + 1.16337 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.00831 w_k1_ADM_s117
                                 + 1.16848 w_k1_AXP_s117
                                 + 0.872829 w_k1_DIS_s117
                                 + 0.968749 w_k1_FDX_s117
                                 + 0.998538 w_k1_IBM_s117
                                 + 0.923568 w_k1_INTC_s117
                                 + 0.925663 w_k1_KO_s117
                                 + 0.79803 w_k1_MAT_s117
                                 + 0.901666 w_k1_MSFT_s117
                                 + 1.00042 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.810963 w_k1_ADM_s117
                                 + 0.807394 w_k1_AXP_s117
                                 + 0.519111 w_k1_DIS_s117
                                 + 0.735491 w_k1_FDX_s117
                                 + 0.837651 w_k1_IBM_s117
                                 + 0.720714 w_k1_INTC_s117
                                 + 0.65822 w_k1_KO_s117 + 0.84584 w_k1_MAT_s117
                                 + 0.701529 w_k1_MSFT_s117
                                 + 0.82839 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.952455 w_k1_ADM_s117
                                 + 0.777253 w_k1_AXP_s117
                                 + 0.717814 w_k1_DIS_s117
                                 + 0.865148 w_k1_FDX_s117
                                 + 0.642209 w_k1_IBM_s117
                                 + 0.947476 w_k1_INTC_s117
                                 + 0.832027 w_k1_KO_s117 + 0.6937 w_k1_MAT_s117
                                 + 0.955294 w_k1_MSFT_s117
                                 + 0.847693 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.051318501351673 w_k1_ADM_s000
                                 + 0.0605406669217971 w_k1_ADM_s001
                                 + 0.0116912249630065 w_k1_ADM_s002
                                 + 0.0324784475943515 w_k1_ADM_s003
                                 + 0.0337955938892973 w_k1_ADM_s004
                                 + 0.0558455168766432 w_k1_ADM_s005
                                 + 0.0585158868769904 w_k1_ADM_s006
                                 + 0.0307071653955814 w_k1_ADM_s007
                                 + 0.0228427568880045 w_k1_ADM_s010
                                 + 0.0268294667405739 w_k1_ADM_s011
                                 + 0.0138229627409488 w_k1_ADM_s012
                                 + 0.0148769264748123 w_k1_ADM_s013
                                 + 0.0114139731873237 w_k1_ADM_s014
                                 + 0.016341447361847 w_k1_ADM_s015
                                 + 0.00989032991986207 w_k1_ADM_s016
                                 + 0.0169030876880878 w_k1_ADM_s017
                                 + 0.0851460975490659 w_k1_ADM_s100
                                 + 0.0576681301769178 w_k1_ADM_s101
                                 + 0.0376898158556193 w_k1_ADM_s102
                                 + 0.049481460200963 w_k1_ADM_s103
                                 + 0.0194745484537329 w_k1_ADM_s104
                                 + 0.0546315926442113 w_k1_ADM_s105
                                 + 0.0538309345162339 w_k1_ADM_s106
                                 + 0.0350778676943211 w_k1_ADM_s107
                                 + 0.0188011153948884 w_k1_ADM_s110
                                 + 0.0209142385533806 w_k1_ADM_s111
                                 + 0.0227784444206337 w_k1_ADM_s112
                                 + 0.0048838381571391 w_k1_ADM_s113
                                 + 0.0214785842197376 w_k1_ADM_s114
                                 + 0.025707553462727 w_k1_ADM_s115
                                 + 0.0238928149121482 w_k1_ADM_s116
                                 + 0.0229372996811316 w_k1_ADM_s117
                                 + 0.0485104352544123 w_k1_AXP_s000
                                 + 0.0626216024194366 w_k1_AXP_s001
                                 + 0.0127118661813587 w_k1_AXP_s002
                                 + 0.0314118023160162 w_k1_AXP_s003
                                 + 0.0397292455732141 w_k1_AXP_s004
                                 + 0.0613048683244943 w_k1_AXP_s005
                                 + 0.0633921783411244 w_k1_AXP_s006
                                 + 0.0299617638946058 w_k1_AXP_s007
                                 + 0.0234293888848197 w_k1_AXP_s010
                                 + 0.0274398698329727 w_k1_AXP_s011
                                 + 0.0154482413151886 w_k1_AXP_s012
                                 + 0.0123430167800952 w_k1_AXP_s013
                                 + 0.0118746941350728 w_k1_AXP_s014
                                 + 0.0136087335941051 w_k1_AXP_s015
                                 + 0.00743704366432723 w_k1_AXP_s016
                                 + 0.018762272411303 w_k1_AXP_s017
                                 + 0.083690537360047 w_k1_AXP_s100
                                 + 0.0684162998169848 w_k1_AXP_s101
                                 + 0.0418197941480693 w_k1_AXP_s102
                                 + 0.0488600945078249 w_k1_AXP_s103
                                 + 0.0169070031140697 w_k1_AXP_s104
                                 + 0.060359429114421 w_k1_AXP_s105
                                 + 0.0561580634093645 w_k1_AXP_s106
                                 + 0.0297838098276663 w_k1_AXP_s107
                                 + 0.019972568356397 w_k1_AXP_s110
                                 + 0.0220948339005392 w_k1_AXP_s111
                                 + 0.0248585303442249 w_k1_AXP_s112
                                 + 0.00614821084337141 w_k1_AXP_s113
                                 + 0.0205776098344772 w_k1_AXP_s114
                                 + 0.0279771075507193 w_k1_AXP_s115
                                 + 0.0224262632789215 w_k1_AXP_s116
                                 + 0.0251197190725472 w_k1_AXP_s117
                                 + 0.0483758434941814 w_k1_DIS_s000
                                 + 0.0595352915959537 w_k1_DIS_s001
                                 + 0.013843905594644 w_k1_DIS_s002
                                 + 0.0305847169808988 w_k1_DIS_s003
                                 + 0.0396750209551796 w_k1_DIS_s004
                                 + 0.0546480188694498 w_k1_DIS_s005
                                 + 0.0578587544766227 w_k1_DIS_s006
                                 + 0.030930087159743 w_k1_DIS_s007
                                 + 0.0228828663391091 w_k1_DIS_s010
                                 + 0.0267713194064288 w_k1_DIS_s011
                                 + 0.0141361934880217 w_k1_DIS_s012
                                 + 0.0141791493390537 w_k1_DIS_s013
                                 + 0.0120363909450055 w_k1_DIS_s014
                                 + 0.0136543986737793 w_k1_DIS_s015
                                 + 0.00848935644564156 w_k1_DIS_s016
                                 + 0.0175305222293454 w_k1_DIS_s017
                                 + 0.0764232087591899 w_k1_DIS_s100
                                 + 0.0674378364248366 w_k1_DIS_s101
                                 + 0.0402352289544626 w_k1_DIS_s102
                                 + 0.05013392974529 w_k1_DIS_s103
                                 + 0.0172560153802721 w_k1_DIS_s104
                                 + 0.060122227926924 w_k1_DIS_s105
                                 + 0.0509879577922385 w_k1_DIS_s106
                                 + 0.0347207781834038 w_k1_DIS_s107
                                 + 0.0185890921574864 w_k1_DIS_s110
                                 + 0.0236249532935513 w_k1_DIS_s111
                                 + 0.0253340754186154 w_k1_DIS_s112
                                 + 0.00617933586356742 w_k1_DIS_s113
                                 + 0.0216504811736321 w_k1_DIS_s114
                                 + 0.0278188707758857 w_k1_DIS_s115
                                 + 0.0250090909574241 w_k1_DIS_s116
                                 + 0.0232895340663188 w_k1_DIS_s117
                                 + 0.047499865157291 w_k1_FDX_s000
                                 + 0.0611295420710064 w_k1_FDX_s001
                                 + 0.0132012696367412 w_k1_FDX_s002
                                 + 0.0299024723729223 w_k1_FDX_s003
                                 + 0.035349839865697 w_k1_FDX_s004
                                 + 0.0575192021616886 w_k1_FDX_s005
                                 + 0.0570286671528131 w_k1_FDX_s006
                                 + 0.0312346003770273 w_k1_FDX_s007
                                 + 0.0220031148254336 w_k1_FDX_s010
                                 + 0.0280503440680093 w_k1_FDX_s011
                                 + 0.0135817908033507 w_k1_FDX_s012
                                 + 0.0136658139642756 w_k1_FDX_s013
                                 + 0.0113807139770743 w_k1_FDX_s014
                                 + 0.0141204002718258 w_k1_FDX_s015
                                 + 0.00820024562225258 w_k1_FDX_s016
                                 + 0.0186650889198473 w_k1_FDX_s017
                                 + 0.0781812068021276 w_k1_FDX_s100
                                 + 0.0650156550191733 w_k1_FDX_s101
                                 + 0.0417590843709632 w_k1_FDX_s102
                                 + 0.0490818593506091 w_k1_FDX_s103
                                 + 0.0176787045805609 w_k1_FDX_s104
                                 + 0.0613619984973545 w_k1_FDX_s105
                                 + 0.0505951384009313 w_k1_FDX_s106
                                 + 0.0317404478680336 w_k1_FDX_s107
                                 + 0.0183043482250799 w_k1_FDX_s110
                                 + 0.0213259663083739 w_k1_FDX_s111
                                 + 0.0240176991824376 w_k1_FDX_s112
                                 + 0.00646072136987721 w_k1_FDX_s113
                                 + 0.0213602866083751 w_k1_FDX_s114
                                 + 0.028064125428591 w_k1_FDX_s115
                                 + 0.0254948831555949 w_k1_FDX_s116
                                 + 0.0240032111614709 w_k1_FDX_s117
                                 + 0.0480999246017812 w_k1_IBM_s000
                                 + 0.0609915944830369 w_k1_IBM_s001
                                 + 0.0134689706783507 w_k1_IBM_s002
                                 + 0.0294734643517549 w_k1_IBM_s003
                                 + 0.0363267921168987 w_k1_IBM_s004
                                 + 0.0583101437735695 w_k1_IBM_s005
                                 + 0.0587326797382082 w_k1_IBM_s006
                                 + 0.0312176366278888 w_k1_IBM_s007
                                 + 0.0225815014868245 w_k1_IBM_s010
                                 + 0.0278475868473652 w_k1_IBM_s011
                                 + 0.0134173870028269 w_k1_IBM_s012
                                 + 0.0139478248658382 w_k1_IBM_s013
                                 + 0.0117594332603507 w_k1_IBM_s014
                                 + 0.0136746827218195 w_k1_IBM_s015
                                 + 0.00858972063066973 w_k1_IBM_s016
                                 + 0.0187973989685467 w_k1_IBM_s017
                                 + 0.077691534710455 w_k1_IBM_s100
                                 + 0.0626648646952528 w_k1_IBM_s101
                                 + 0.0412471053756808 w_k1_IBM_s102
                                 + 0.0498666133053896 w_k1_IBM_s103
                                 + 0.0172745609672931 w_k1_IBM_s104
                                 + 0.0620167916576037 w_k1_IBM_s105
                                 + 0.0515842107084331 w_k1_IBM_s106
                                 + 0.0338037795667056 w_k1_IBM_s107
                                 + 0.0201477378093543 w_k1_IBM_s110
                                 + 0.0214388483973461 w_k1_IBM_s111
                                 + 0.0245046737977746 w_k1_IBM_s112
                                 + 0.00639890940963614 w_k1_IBM_s113
                                 + 0.0210206832094972 w_k1_IBM_s114
                                 + 0.0271372703748577 w_k1_IBM_s115
                                 + 0.0252337768440527 w_k1_IBM_s116
                                 + 0.0231478364298947 w_k1_IBM_s117
                                 + 0.0424717514401964 w_k1_INTC_s000
                                 + 0.0606586043297678 w_k1_INTC_s001
                                 + 0.0141715198815956 w_k1_INTC_s002
                                 + 0.0291738501748397 w_k1_INTC_s003
                                 + 0.0344982919953313 w_k1_INTC_s004
                                 + 0.0558150410529995 w_k1_INTC_s005
                                 + 0.0604354499138871 w_k1_INTC_s006
                                 + 0.0337134440549755 w_k1_INTC_s007
                                 + 0.0233722275169995 w_k1_INTC_s010
                                 + 0.0261634502944816 w_k1_INTC_s011
                                 + 0.0125625551740515 w_k1_INTC_s012
                                 + 0.0115496423822994 w_k1_INTC_s013
                                 + 0.0115543356124498 w_k1_INTC_s014
                                 + 0.0132152353654633 w_k1_INTC_s015
                                 + 0.00656577462523294 w_k1_INTC_s016
                                 + 0.0199996234009555 w_k1_INTC_s017
                                 + 0.0823985115043009 w_k1_INTC_s100
                                 + 0.057614507712658 w_k1_INTC_s101
                                 + 0.0383623821300675 w_k1_INTC_s102
                                 + 0.0476395210939417 w_k1_INTC_s103
                                 + 0.0174955736286702 w_k1_INTC_s104
                                 + 0.0635737053350385 w_k1_INTC_s105
                                 + 0.0583511801713906 w_k1_INTC_s106
                                 + 0.0278742415764773 w_k1_INTC_s107
                                 + 0.0202344800220095 w_k1_INTC_s110
                                 + 0.021324381652755 w_k1_INTC_s111
                                 + 0.0234321013631044 w_k1_INTC_s112
                                 + 0.0060633233128209 w_k1_INTC_s113
                                 + 0.0191857871993885 w_k1_INTC_s114
                                 + 0.0286376696509699 w_k1_INTC_s115
                                 + 0.0223357197083727 w_k1_INTC_s116
                                 + 0.0232708611988519 w_k1_INTC_s117
                                 + 0.0488241011645624 w_k1_KO_s000
                                 + 0.0575389609986036 w_k1_KO_s001
                                 + 0.0134652825248571 w_k1_KO_s002
                                 + 0.032616835146121 w_k1_KO_s003
                                 + 0.0372682847356669 w_k1_KO_s004
                                 + 0.0549421590394811 w_k1_KO_s005
                                 + 0.059665937146858 w_k1_KO_s006
                                 + 0.0310115118856451 w_k1_KO_s007
                                 + 0.02296284914575 w_k1_KO_s010
                                 + 0.0254451480264436 w_k1_KO_s011
                                 + 0.0133170787406298 w_k1_KO_s012
                                 + 0.0121915756448548 w_k1_KO_s013
                                 + 0.0110147510340874 w_k1_KO_s014
                                 + 0.0137800740389305 w_k1_KO_s015
                                 + 0.00874593403929179 w_k1_KO_s016
                                 + 0.0180067802431813 w_k1_KO_s017
                                 + 0.0780205172663886 w_k1_KO_s100
                                 + 0.0596269907064443 w_k1_KO_s101
                                 + 0.0397253248071057 w_k1_KO_s102
                                 + 0.0521843619331135 w_k1_KO_s103
                                 + 0.0184391037592433 w_k1_KO_s104
                                 + 0.0613823994439766 w_k1_KO_s105
                                 + 0.0561702530228207 w_k1_KO_s106
                                 + 0.0319141317913792 w_k1_KO_s107
                                 + 0.0185692106564811 w_k1_KO_s110
                                 + 0.0227388251401591 w_k1_KO_s111
                                 + 0.0251024969247772 w_k1_KO_s112
                                 + 0.00628344309150133 w_k1_KO_s113
                                 + 0.0228619147934622 w_k1_KO_s114
                                 + 0.0284619790958674 w_k1_KO_s115
                                 + 0.0240549645597909 w_k1_KO_s116
                                 + 0.0243465594694778 w_k1_KO_s117
                                 + 0.0504667296423386 w_k1_MAT_s000
                                 + 0.0520111604488959 w_k1_MAT_s001
                                 + 0.0119543647498295 w_k1_MAT_s002
                                 + 0.0371257524178606 w_k1_MAT_s003
                                 + 0.0362619464327637 w_k1_MAT_s004
                                 + 0.0531146800111703 w_k1_MAT_s005
                                 + 0.0513676711586254 w_k1_MAT_s006
                                 + 0.0305217879416566 w_k1_MAT_s007
                                 + 0.0152648222026043 w_k1_MAT_s010
                                 + 0.0287604213525027 w_k1_MAT_s011
                                 + 0.0133528160242652 w_k1_MAT_s012
                                 + 0.0157427750386785 w_k1_MAT_s013
                                 + 0.00931019819555525 w_k1_MAT_s014
                                 + 0.0129982781627558 w_k1_MAT_s015
                                 + 0.00957483848138785 w_k1_MAT_s016
                                 + 0.0241236879179728 w_k1_MAT_s017
                                 + 0.07199380594642 w_k1_MAT_s100
                                 + 0.0595295564094457 w_k1_MAT_s101
                                 + 0.0409906234262946 w_k1_MAT_s102
                                 + 0.0411933172809098 w_k1_MAT_s103
                                 + 0.0136845398551571 w_k1_MAT_s104
                                 + 0.0658334176209252 w_k1_MAT_s105
                                 + 0.0519681684629357 w_k1_MAT_s106
                                 + 0.0346752190478924 w_k1_MAT_s107
                                 + 0.0166562842024142 w_k1_MAT_s110
                                 + 0.022989909485271 w_k1_MAT_s111
                                 + 0.0246353101574128 w_k1_MAT_s112
                                 + 0.00529481807464394 w_k1_MAT_s113
                                 + 0.0182210122568162 w_k1_MAT_s114
                                 + 0.0298336087594591 w_k1_MAT_s115
                                 + 0.0278219878069694 w_k1_MAT_s116
                                 + 0.02340705139983 w_k1_MAT_s117
                                 + 0.041540990382723 w_k1_MSFT_s000
                                 + 0.0570083698814216 w_k1_MSFT_s001
                                 + 0.0129866763750926 w_k1_MSFT_s002
                                 + 0.0293127933672903 w_k1_MSFT_s003
                                 + 0.0373773265366056 w_k1_MSFT_s004
                                 + 0.0557380570556509 w_k1_MSFT_s005
                                 + 0.0573168549270594 w_k1_MSFT_s006
                                 + 0.0300015812923374 w_k1_MSFT_s007
                                 + 0.0208107119497017 w_k1_MSFT_s010
                                 + 0.0274598653171807 w_k1_MSFT_s011
                                 + 0.0131288017317688 w_k1_MSFT_s012
                                 + 0.0136721787624167 w_k1_MSFT_s013
                                 + 0.0106367798092218 w_k1_MSFT_s014
                                 + 0.0128097329585032 w_k1_MSFT_s015
                                 + 0.00834400207920655 w_k1_MSFT_s016
                                 + 0.0197307398269565 w_k1_MSFT_s017
                                 + 0.0810253859416035 w_k1_MSFT_s100
                                 + 0.0586395364426473 w_k1_MSFT_s101
                                 + 0.0373181817781558 w_k1_MSFT_s102
                                 + 0.0482389912794981 w_k1_MSFT_s103
                                 + 0.0161303203921993 w_k1_MSFT_s104
                                 + 0.0622374818510858 w_k1_MSFT_s105
                                 + 0.0553564232574396 w_k1_MSFT_s106
                                 + 0.0304999150101488 w_k1_MSFT_s107
                                 + 0.0179104670146952 w_k1_MSFT_s110
                                 + 0.0213314136212126 w_k1_MSFT_s111
                                 + 0.0259367425746204 w_k1_MSFT_s112
                                 + 0.00614500679266529 w_k1_MSFT_s113
                                 + 0.0196377909853787 w_k1_MSFT_s114
                                 + 0.0267998437400169 w_k1_MSFT_s115
                                 + 0.024382243860063 w_k1_MSFT_s116
                                 + 0.0233078653862683 w_k1_MSFT_s117
                                 + 0.045352277340868 w_k1_PG_s000
                                 + 0.0594352572879545 w_k1_PG_s001
                                 + 0.0131175554125766 w_k1_PG_s002
                                 + 0.0313097398950271 w_k1_PG_s003
                                 + 0.0367531721200941 w_k1_PG_s004
                                 + 0.0550765442067165 w_k1_PG_s005
                                 + 0.0597401812829086 w_k1_PG_s006
                                 + 0.0304391763812569 w_k1_PG_s007
                                 + 0.0223326300123208 w_k1_PG_s010
                                 + 0.0280144087871645 w_k1_PG_s011
                                 + 0.01426659344438 w_k1_PG_s012
                                 + 0.0126757330396553 w_k1_PG_s013
                                 + 0.011244880352756 w_k1_PG_s014
                                 + 0.0148559923966846 w_k1_PG_s015
                                 + 0.00812118102891905 w_k1_PG_s016
                                 + 0.0168980547251474 w_k1_PG_s017
                                 + 0.0768841720875125 w_k1_PG_s100
                                 + 0.0630432137929926 w_k1_PG_s101
                                 + 0.0392196679380224 w_k1_PG_s102
                                 + 0.045870507607805 w_k1_PG_s103
                                 + 0.0165973037961804 w_k1_PG_s104
                                 + 0.060477087211971 w_k1_PG_s105
                                 + 0.0514381970784298 w_k1_PG_s106
                                 + 0.0302457887754644 w_k1_PG_s107
                                 + 0.017984715505919 w_k1_PG_s110
                                 + 0.0206684308396795 w_k1_PG_s111
                                 + 0.0223803728425187 w_k1_PG_s112
                                 + 0.00594204245752296 w_k1_PG_s113
                                 + 0.0223924115904612 w_k1_PG_s114
                                 + 0.0268109025858171 w_k1_PG_s115
                                 + 0.025805058586952 w_k1_PG_s116
                                 + 0.0234538438831916 w_k1_PG_s117 >= 1.219
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
