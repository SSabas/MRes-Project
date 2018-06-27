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
 w_k1_ADM_s0_return_constraint:  1.17763 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.905112 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.13926 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.943442 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.20144 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.95281 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.756487 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.18537 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.887884 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.765637 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.0568 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.28867 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.867836 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.57176 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.19451 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.803358 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.36321 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.838396 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.904331 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.639418 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.970257 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.16081 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.938805 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.02873 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.983126 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.03503 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.03939 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.89718 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.13402 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.86387 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.746625 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.793374 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.12594 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.895986 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.963675 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.999047 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.872377 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.768104 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.19072 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.838007 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.09727 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.978722 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.26776 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.859381 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.829986 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.06167 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.27441 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.677024 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.07099 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.94117 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.797861 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.547343 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.66822 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.834319 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.55433 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.06846 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.01397 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.607815 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.01914 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.14787 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.01464 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.906742 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.15281 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.23144 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.47229 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.814151 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.31574 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.839231 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.988342 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.733533 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.56815 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.08613 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.13294 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.16263 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.795861 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.722537 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25552 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.81158 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.09913 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.965143 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.31051 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.828424 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.737448 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.29774 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.35944 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.671122 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.912791 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.91318 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.867602 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.434011 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.76368 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.748052 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.46239 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.980156 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.848772 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.575265 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.15434 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.24204 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.900466 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.877043 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.19689 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.42761 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.71192 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.830648 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.41737 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.666251 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.858288 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.798891 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.54546 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.04874 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.15242 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.44749 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.608318 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.579424 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.14774 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.889333 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.13844 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.949201 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.1733 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.86364 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.795379 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.26273 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.16874 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.772203 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.992247 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.43576 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.961904 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.625126 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.27115 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.825889 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.41909 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.895678 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.91017 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.718361 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.06206 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.16894 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.918943 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.09052 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.14315 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.10141 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.46263 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.892211 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.02441 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.794264 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.862273 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.842862 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.35186 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.03788 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.26225 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.10534 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.843259 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.733341 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14529 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.938682 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.843057 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.17857 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.19128 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.898102 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.05438 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.812376 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.25858 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.665054 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.00227 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.95618 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.700073 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.555269 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.66878 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.778257 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.11596 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.4312 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.994844 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.551104 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.11089 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.82986 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.27431 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.55615 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.903712 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.52557 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.989488 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.885759 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 2.07432 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.639275 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.37238 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.717091 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.4518 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.0378 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.713926 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.63993 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.575775 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.909067 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11167 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.858658 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.03477 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.9955 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.25004 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.963072 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.832693 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.06912 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.18055 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.909637 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.911982 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.43537 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.00213 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.794636 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.31051 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.8834 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.18406 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.941098 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.03834 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.695748 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.04576 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.12459 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.939727 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.00028 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.01509 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.12928 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.24623 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.938014 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.893448 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.745295 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.03683 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.897226 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.18074 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.09068 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.05145 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.29453 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.868462 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.747267 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.10482 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.912731 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.02194 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.13519 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.14958 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.810504 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.895397 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.05875 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.25168 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.736029 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.921958 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.61226 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.806988 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.558089 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.48593 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.751372 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.16986 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.04285 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.920269 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.597009 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.01104 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.068 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.05138 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.817953 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.07995 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.41037 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.2001 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.872793 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.69476 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.820759 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.04556 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.923057 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.45891 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.2157 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.03751 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.59011 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.697473 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.838721 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.21058 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.85662 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.897794 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.21468 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.25979 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.807446 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.35423 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.792129 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.29201 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.525849 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.03297 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 2.14837 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.604214 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.467361 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.55746 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.667566 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.1557 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.24971 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.89142 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.534855 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.897268 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.803305 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.25491 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.718241 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.974219 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.99971 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.25499 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.771131 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 2.13968 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.863818 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.59359 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.757017 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 2.01954 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.32053 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.0355 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.73937 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.553681 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.937359 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14789 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.887971 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.02626 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.971554 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.1596 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.849893 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.93185 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.24621 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.11258 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.807837 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.00813 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.46476 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.975933 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.742961 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.40119 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.81297 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.25885 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.870804 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.899721 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.723053 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.99034 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.16005 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.866195 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.06036 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.07314 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.25465 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.32656 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.870087 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.13566 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.80777 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.9007 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.945376 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.39291 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.09702 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.19705 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.34742 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.898132 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.811935 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06705 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.933226 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.00825 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.01547 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.1083 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.914938 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.941401 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.06695 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.19171 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.861989 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.993863 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.32088 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.977199 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.735925 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.34168 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.918541 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.22284 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.06875 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.02475 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.772535 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.08021 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.14112 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.983316 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.910601 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.07039 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.13057 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.15981 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.911639 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.13894 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.883626 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.930425 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.810658 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.13871 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.963191 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.00852 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.17101 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.756842 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.886978 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.694997140047293 z_k1_s0000
                                 - 0.63299954647055 z_k1_s0001
                                 - 0.291999413424431 z_k1_s0002
                                 - 0.412998325797839 z_k1_s0003
                                 - 0.277999559617551 z_k1_s0004
                                 - 0.371999633433124 z_k1_s0005
                                 - 0.254999623892303 z_k1_s0006
                                 - 0.579997461421058 z_k1_s0007
                                 - 0.449999274725744 z_k1_s0010
                                 - 0.159999559058643 z_k1_s0011
                                 - 0.343998530237792 z_k1_s0012
                                 - 0.526999608288427 z_k1_s0013
                                 - 0.166999364710532 z_k1_s0014
                                 - 0.346999898398129 z_k1_s0015
                                 - 0.324999018525649 z_k1_s0016
                                 - 0.425999313407038 z_k1_s0017
                                 - 0.870004754752088 z_k1_s0020
                                 - 0.665998858908662 z_k1_s0021
                                 - 0.813001318850208 z_k1_s0022
                                 - 0.73400285119485 z_k1_s0023
                                 - 1.26400588756141 z_k1_s0024
                                 - 0.598001555739126 z_k1_s0025
                                 - 0.451000920102122 z_k1_s0026
                                 - 0.685002031214335 z_k1_s0027
                                 - 0.282999876817809 z_k1_s0030
                                 - 0.252999982357557 z_k1_s0031
                                 - 0.411999321026957 z_k1_s0032
                                 - 0.470999124795445 z_k1_s0033
                                 - 0.998000137302009 z_k1_s0034
                                 - 0.229999983961416 z_k1_s0035
                                 - 0.425000487602281 z_k1_s0036
                                 - 0.326999903306061 z_k1_s0037
                                 - 0.215999763675484 z_k1_s0040
                                 - 0.161999746181698 z_k1_s0041
                                 - 0.376999747766293 z_k1_s0042
                                 - 0.351001032608591 z_k1_s0043
                                 - 0.605998562720968 z_k1_s0044
                                 - 0.590000531468767 z_k1_s0045
                                 - 0.599998152377667 z_k1_s0046
                                 - 0.160999677791148 z_k1_s0047
                                 - 0.25700052480805 z_k1_s0050
                                 - 0.101000048183731 z_k1_s0051
                                 - 0.221999745873204 z_k1_s0052
                                 - 0.325999626822813 z_k1_s0053
                                 - 0.251999711531745 z_k1_s0054
                                 - 0.173000620964135 z_k1_s0055
                                 - 0.223999743583774 z_k1_s0056
                                 - 0.0830000687886067 z_k1_s0057
                                 - 0.166000087530745 z_k1_s0060
                                 - 0.289000195028854 z_k1_s0061
                                 - 0.440000258659823 z_k1_s0062
                                 - 0.204000059596789 z_k1_s0063
                                 - 0.238000290729727 z_k1_s0064
                                 - 0.126000023798008 z_k1_s0065
                                 - 0.276000262795771 z_k1_s0066
                                 - 0.473001107559835 z_k1_s0067
                                 - 0.183000064802033 z_k1_s0070
                                 - 0.28500054765886 z_k1_s0071
                                 - 0.309999330172921 z_k1_s0072
                                 - 0.131000032373042 z_k1_s0073
                                 - 0.275999703553926 z_k1_s0074
                                 - 0.160000062789209 z_k1_s0075
                                 - 0.0969999248540941 z_k1_s0076
                                 - 0.161000494989723 z_k1_s0077
                                 - 0.543001936908581 z_k1_s0100
                                 - 0.292000739355408 z_k1_s0101
                                 - 0.352000725140442 z_k1_s0102
                                 - 0.74300062618957 z_k1_s0103
                                 - 0.310000621390697 z_k1_s0104
                                 - 0.399000606621843 z_k1_s0105
                                 - 0.772001762637895 z_k1_s0106
                                 - 0.379000737693744 z_k1_s0107
                                 - 0.573002806641331 z_k1_s0110
                                 - 1.04900256123067 z_k1_s0111
                                 - 0.608997965252158 z_k1_s0112
                                 - 0.39200065008625 z_k1_s0113
                                 - 1.15499883092939 z_k1_s0114
                                 - 1.09600243164589 z_k1_s0115
                                 - 0.39000029994054 z_k1_s0116
                                 - 0.308000431181968 z_k1_s0117
                                 - 0.408999017066709 z_k1_s0120
                                 - 0.110999872289601 z_k1_s0121
                                 - 0.160999845864165 z_k1_s0122
                                 - 0.248000261401438 z_k1_s0123
                                 - 0.110999872289601 z_k1_s0124
                                 - 0.173999740875651 z_k1_s0125
                                 - 0.226999396707009 z_k1_s0126
                                 - 0.376000513543996 z_k1_s0127
                                 - 0.437002761809895 z_k1_s0130
                                 - 0.290000742115351 z_k1_s0131
                                 - 0.225000722231325 z_k1_s0132
                                 - 0.352000727201549 z_k1_s0133
                                 - 1.01500338390614 z_k1_s0134
                                 - 0.363999607989003 z_k1_s0135
                                 - 0.233000710158545 z_k1_s0136
                                 - 0.230000675360717 z_k1_s0137
                                 - 0.234999457425652 z_k1_s0140
                                 - 0.170999481488828 z_k1_s0141
                                 - 0.84199802686136 z_k1_s0142
                                 - 0.361000425360695 z_k1_s0143
                                 - 0.338998948074416 z_k1_s0144
                                 - 0.477998270595351 z_k1_s0145
                                 - 0.61499963517457 z_k1_s0146
                                 - 0.378999830782209 z_k1_s0147
                                 - 0.137000310220028 z_k1_s0150
                                 - 0.393001525694959 z_k1_s0151
                                 - 0.258000440447709 z_k1_s0152
                                 - 0.236000366333505 z_k1_s0153
                                 - 0.418999494953838 z_k1_s0154
                                 - 0.447000295391218 z_k1_s0155
                                 - 0.356001854976337 z_k1_s0156
                                 - 0.528001223940009 z_k1_s0157
                                 - 0.204999721534752 z_k1_s0160
                                 - 0.196999608670107 z_k1_s0161
                                 - 0.245999665841702 z_k1_s0162
                                 - 0.237999552977057 z_k1_s0163
                                 - 0.0989998325265722 z_k1_s0164
                                 - 0.174999721041903 z_k1_s0165
                                 - 0.14099983321656 z_k1_s0166
                                 - 0.390000006407031 z_k1_s0167
                                 - 0.148999866411575 z_k1_s0170
                                 - 0.365999774931762 z_k1_s0171
                                 - 0.491999811891847 z_k1_s0172
                                 - 0.148999866411575 z_k1_s0173
                                 - 0.453999911554651 z_k1_s0174
                                 - 0.277999638308873 z_k1_s0175
                                 - 0.279000248054433 z_k1_s0176
                                 - 0.160000058014392 z_k1_s0177
                                 - 0.352999448799775 z_k1_s1000
                                 - 0.786998506845185 z_k1_s1001
                                 - 0.314999699864694 z_k1_s1002
                                 - 0.331999346862295 z_k1_s1003
                                 - 0.327999349221793 z_k1_s1004
                                 - 0.508997070282476 z_k1_s1005
                                 - 0.662000295452155 z_k1_s1006
                                 - 1.28700049840482 z_k1_s1007
                                 - 0.365000294661771 z_k1_s1010
                                 - 0.274000184096729 z_k1_s1011
                                 - 0.412000128546699 z_k1_s1012
                                 - 0.353999661240951 z_k1_s1013
                                 - 0.286000669919154 z_k1_s1014
                                 - 0.222000411102827 z_k1_s1015
                                 - 0.107000231694202 z_k1_s1016
                                 - 0.237001131231015 z_k1_s1017
                                 - 0.776001256673584 z_k1_s1020
                                 - 0.693000527890671 z_k1_s1021
                                 - 0.854003911765704 z_k1_s1022
                                 - 0.263000985697797 z_k1_s1023
                                 - 0.209000824763034 z_k1_s1024
                                 - 0.216000643739048 z_k1_s1025
                                 - 0.763002910665651 z_k1_s1026
                                 - 0.419001683565711 z_k1_s1027
                                 - 0.286999733415787 z_k1_s1030
                                 - 0.318999414089137 z_k1_s1031
                                 - 0.315000398505578 z_k1_s1032
                                 - 0.151000146053439 z_k1_s1033
                                 - 0.315999680151163 z_k1_s1034
                                 - 0.22800055829945 z_k1_s1035
                                 - 0.137000002408665 z_k1_s1036
                                 - 0.136000154062714 z_k1_s1037
                                 - 0.324001089435873 z_k1_s1040
                                 - 0.155000113331429 z_k1_s1041
                                 - 0.591000212440833 z_k1_s1042
                                 - 0.162000319217634 z_k1_s1043
                                 - 0.145000302423213 z_k1_s1044
                                 - 0.531001346991537 z_k1_s1045
                                 - 0.185000222556984 z_k1_s1046
                                 - 0.162000319217634 z_k1_s1047
                                 - 0.625997726517908 z_k1_s1050
                                 - 0.646001224357975 z_k1_s1051
                                 - 0.193999759086506 z_k1_s1052
                                 - 0.527998275081984 z_k1_s1053
                                 - 0.793997360407428 z_k1_s1054
                                 - 0.330999599317687 z_k1_s1055
                                 - 0.503998901668559 z_k1_s1056
                                 - 0.396999484202823 z_k1_s1057
                                 - 0.177999943901953 z_k1_s1060
                                 - 0.40500063058604 z_k1_s1061
                                 - 0.131000042778419 z_k1_s1062
                                 - 0.164999387571934 z_k1_s1063
                                 - 0.145999973766774 z_k1_s1064
                                 - 0.1060000110978 z_k1_s1065
                                 - 0.212000608995705 z_k1_s1066
                                 - 0.123999957623832 z_k1_s1067
                                 - 0.42900101173124 z_k1_s1070
                                 - 0.499998189932628 z_k1_s1071
                                 - 0.300999752406472 z_k1_s1072
                                 - 0.259999765844152 z_k1_s1073
                                 - 0.20499961141197 z_k1_s1074
                                 - 0.442997965604804 z_k1_s1075
                                 - 0.418999055254883 z_k1_s1076
                                 - 0.656998695046236 z_k1_s1077
                                 - 0.32199955793832 z_k1_s1100
                                 - 0.615998292022321 z_k1_s1101
                                 - 0.55500107399054 z_k1_s1102
                                 - 0.67899785729769 z_k1_s1103
                                 - 0.45899901214611 z_k1_s1104
                                 - 0.412998421470108 z_k1_s1105
                                 - 0.530000338406327 z_k1_s1106
                                 - 0.239999815018425 z_k1_s1107
                                 - 0.235999839741551 z_k1_s1110
                                 - 0.396998505106959 z_k1_s1111
                                 - 0.234999900995136 z_k1_s1112
                                 - 0.465000359057988 z_k1_s1113
                                 - 0.0980000163434545 z_k1_s1114
                                 - 0.364000777275076 z_k1_s1115
                                 - 0.454000280448073 z_k1_s1116
                                 - 0.259000938906936 z_k1_s1117
                                 - 0.454000188698221 z_k1_s1120
                                 - 0.408999193431747 z_k1_s1121
                                 - 0.229999812967221 z_k1_s1122
                                 - 0.341999740032981 z_k1_s1123
                                 - 0.546998768252391 z_k1_s1124
                                 - 0.168999796804676 z_k1_s1125
                                 - 0.165999817386824 z_k1_s1126
                                 - 0.291000350729248 z_k1_s1127
                                 - 1.38300087341632 z_k1_s1130
                                 - 0.298000163626785 z_k1_s1131
                                 - 0.947002686633428 z_k1_s1132
                                 - 0.934001340648216 z_k1_s1133
                                 - 0.440000537241072 z_k1_s1134
                                 - 0.855999557740808 z_k1_s1135
                                 - 0.466000082709564 z_k1_s1136
                                 - 0.969003028145675 z_k1_s1137
                                 - 1.07699797047491 z_k1_s1140
                                 - 0.443998741021422 z_k1_s1141
                                 - 0.610997632073965 z_k1_s1142
                                 - 0.738997251035143 z_k1_s1143
                                 - 0.520000334524773 z_k1_s1144
                                 - 0.550998751933404 z_k1_s1145
                                 - 0.411998836281127 z_k1_s1146
                                 - 0.665996605536145 z_k1_s1147
                                 - 0.170999807282717 z_k1_s1150
                                 - 0.384999256619313 z_k1_s1151
                                 - 0.320999737004293 z_k1_s1152
                                 - 0.340999727633836 z_k1_s1153
                                 - 0.113999946588398 z_k1_s1154
                                 - 0.175999917539982 z_k1_s1155
                                 - 0.333000632181056 z_k1_s1156
                                 - 0.410998568838441 z_k1_s1157
                                 - 0.165999954614426 z_k1_s1160
                                 - 0.54000093567284 z_k1_s1161
                                 - 0.35599898030373 z_k1_s1162
                                 - 0.186999976729488 z_k1_s1163
                                 - 0.394998947974601 z_k1_s1164
                                 - 0.371998703553462 z_k1_s1165
                                 - 0.199999880319825 z_k1_s1166
                                 - 0.353000958943352 z_k1_s1167
                                 - 0.221999833143999 z_k1_s1170
                                 - 0.188999732121345 z_k1_s1171
                                 - 0.457999007201214 z_k1_s1172
                                 - 0.126999893293648 z_k1_s1173
                                 - 0.469998961354974 z_k1_s1174
                                 - 0.310999568817656 z_k1_s1175
                                 - 0.210999799469781 z_k1_s1176
                                 - 0.283000432791595 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.946793 w_k1_ADM_s000
                                 + 1.09398 w_k1_AXP_s000
                                 + 0.738918 w_k1_DIS_s000
                                 + 0.953521 w_k1_FDX_s000
                                 + 0.744666 w_k1_IBM_s000
                                 + 1.13939 w_k1_INTC_s000
                                 + 0.803526 w_k1_KO_s000
                                 + 0.555247 w_k1_MAT_s000
                                 + 0.786338 w_k1_MSFT_s000
                                 + 0.897121 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.013 w_k1_ADM_s000
                                 + 1.30686 w_k1_AXP_s000
                                 + 1.15054 w_k1_DIS_s000
                                 + 1.07518 w_k1_FDX_s000
                                 + 1.31925 w_k1_IBM_s000
                                 + 1.34939 w_k1_INTC_s000
                                 + 1.14242 w_k1_KO_s000 + 0.83204 w_k1_MAT_s000
                                 + 1.0101 w_k1_MSFT_s000 + 1.02971 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.919215 w_k1_ADM_s000
                                 + 0.614353 w_k1_AXP_s000
                                 + 0.617542 w_k1_DIS_s000
                                 + 0.667145 w_k1_FDX_s000
                                 + 0.898355 w_k1_IBM_s000
                                 + 0.799437 w_k1_INTC_s000
                                 + 0.767487 w_k1_KO_s000
                                 + 0.729271 w_k1_MAT_s000
                                 + 0.735671 w_k1_MSFT_s000
                                 + 0.821819 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.2175 w_k1_ADM_s000
                                 + 0.89483 w_k1_AXP_s000
                                 + 1.09645 w_k1_DIS_s000
                                 + 1.13243 w_k1_FDX_s000
                                 + 0.827619 w_k1_IBM_s000
                                 + 1.33958 w_k1_INTC_s000
                                 + 0.976385 w_k1_KO_s000
                                 + 0.511949 w_k1_MAT_s000
                                 + 1.10497 w_k1_MSFT_s000
                                 + 0.984418 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.22126 w_k1_ADM_s000
                                 + 1.20252 w_k1_AXP_s000
                                 + 1.39027 w_k1_DIS_s000
                                 + 1.19067 w_k1_FDX_s000
                                 + 0.983325 w_k1_IBM_s000
                                 + 1.28576 w_k1_INTC_s000
                                 + 1.18369 w_k1_KO_s000
                                 + 0.718289 w_k1_MAT_s000
                                 + 1.03572 w_k1_MSFT_s000
                                 + 1.18594 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.48485 w_k1_ADM_s000
                                 + 1.20378 w_k1_AXP_s000
                                 + 1.66214 w_k1_DIS_s000
                                 + 1.40992 w_k1_FDX_s000
                                 + 1.21342 w_k1_IBM_s000
                                 + 1.13394 w_k1_INTC_s000
                                 + 1.23044 w_k1_KO_s000 + 1.0554 w_k1_MAT_s000
                                 + 1.14273 w_k1_MSFT_s000 + 1.268 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.932802 w_k1_ADM_s000
                                 + 0.818521 w_k1_AXP_s000
                                 + 0.957524 w_k1_DIS_s000
                                 + 0.937554 w_k1_FDX_s000
                                 + 0.747968 w_k1_IBM_s000
                                 + 0.985915 w_k1_INTC_s000
                                 + 0.817121 w_k1_KO_s000
                                 + 0.449125 w_k1_MAT_s000
                                 + 0.820243 w_k1_MSFT_s000
                                 + 0.880552 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.797105 w_k1_ADM_s000
                                 + 0.934073 w_k1_AXP_s000
                                 + 0.925383 w_k1_DIS_s000
                                 + 0.815778 w_k1_FDX_s000
                                 + 1.12533 w_k1_IBM_s000
                                 + 1.09867 w_k1_INTC_s000
                                 + 1.04556 w_k1_KO_s000
                                 + 0.683589 w_k1_MAT_s000
                                 + 0.840454 w_k1_MSFT_s000
                                 + 0.961023 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.8017 w_k1_ADM_s001
                                 + 1.10688 w_k1_AXP_s001
                                 + 1.04543 w_k1_DIS_s001
                                 + 1.00912 w_k1_FDX_s001
                                 + 1.18247 w_k1_IBM_s001
                                 + 1.20003 w_k1_INTC_s001 + 1.1465 w_k1_KO_s001
                                 + 1.25156 w_k1_MAT_s001
                                 + 1.09063 w_k1_MSFT_s001
                                 + 1.00878 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.684485 w_k1_ADM_s001
                                 + 1.05988 w_k1_AXP_s001
                                 + 0.671225 w_k1_DIS_s001
                                 + 0.846474 w_k1_FDX_s001
                                 + 0.960076 w_k1_IBM_s001
                                 + 0.958417 w_k1_INTC_s001
                                 + 0.816364 w_k1_KO_s001
                                 + 1.18918 w_k1_MAT_s001
                                 + 0.804992 w_k1_MSFT_s001
                                 + 1.0394 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.668502 w_k1_ADM_s001
                                 + 0.930971 w_k1_AXP_s001
                                 + 0.898143 w_k1_DIS_s001
                                 + 1.0077 w_k1_FDX_s001
                                 + 0.654832 w_k1_IBM_s001
                                 + 1.12426 w_k1_INTC_s001
                                 + 0.73244 w_k1_KO_s001
                                 + 0.953741 w_k1_MAT_s001
                                 + 0.953545 w_k1_MSFT_s001
                                 + 0.932237 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.795726 w_k1_ADM_s001
                                 + 0.925444 w_k1_AXP_s001
                                 + 0.857663 w_k1_DIS_s001
                                 + 0.87477 w_k1_FDX_s001
                                 + 0.914383 w_k1_IBM_s001
                                 + 1.11608 w_k1_INTC_s001
                                 + 0.924512 w_k1_KO_s001
                                 + 0.764181 w_k1_MAT_s001
                                 + 0.945268 w_k1_MSFT_s001
                                 + 1.07288 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.767405 w_k1_ADM_s001
                                 + 0.804977 w_k1_AXP_s001
                                 + 0.8556 w_k1_DIS_s001
                                 + 0.780719 w_k1_FDX_s001
                                 + 1.31198 w_k1_IBM_s001
                                 + 1.07081 w_k1_INTC_s001
                                 + 0.991572 w_k1_KO_s001
                                 + 1.30692 w_k1_MAT_s001
                                 + 0.910752 w_k1_MSFT_s001
                                 + 1.06121 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.714803 w_k1_ADM_s001
                                 + 0.771377 w_k1_AXP_s001
                                 + 0.543999 w_k1_DIS_s001
                                 + 0.718948 w_k1_FDX_s001
                                 + 0.681291 w_k1_IBM_s001
                                 + 0.929057 w_k1_INTC_s001
                                 + 0.611424 w_k1_KO_s001
                                 + 0.927777 w_k1_MAT_s001
                                 + 0.739685 w_k1_MSFT_s001
                                 + 0.937052 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.10311 w_k1_ADM_s001
                                 + 1.54186 w_k1_AXP_s001
                                 + 1.67742 w_k1_DIS_s001
                                 + 1.43382 w_k1_FDX_s001
                                 + 1.20009 w_k1_IBM_s001
                                 + 1.48672 w_k1_INTC_s001
                                 + 1.08404 w_k1_KO_s001 + 1.25308 w_k1_MAT_s001
                                 + 1.33242 w_k1_MSFT_s001
                                 + 1.17725 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.960638 w_k1_ADM_s001
                                 + 1.37904 w_k1_AXP_s001
                                 + 1.35999 w_k1_DIS_s001
                                 + 1.18465 w_k1_FDX_s001
                                 + 1.26363 w_k1_IBM_s001
                                 + 1.29271 w_k1_INTC_s001
                                 + 1.34962 w_k1_KO_s001 + 1.50906 w_k1_MAT_s001
                                 + 1.14641 w_k1_MSFT_s001
                                 + 1.07917 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.889503 w_k1_ADM_s002
                                 + 0.650592 w_k1_AXP_s002
                                 + 0.599723 w_k1_DIS_s002
                                 + 0.753485 w_k1_FDX_s002
                                 + 0.81441 w_k1_IBM_s002
                                 + 0.722668 w_k1_INTC_s002
                                 + 0.777448 w_k1_KO_s002
                                 + 0.842356 w_k1_MAT_s002
                                 + 0.830286 w_k1_MSFT_s002
                                 + 0.783135 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.4631 w_k1_ADM_s002
                                 + 1.65224 w_k1_AXP_s002
                                 + 0.99447 w_k1_DIS_s002 + 1.1677 w_k1_FDX_s002
                                 + 1.06907 w_k1_IBM_s002
                                 + 1.12775 w_k1_INTC_s002
                                 + 0.961983 w_k1_KO_s002
                                 + 1.40146 w_k1_MAT_s002
                                 + 1.18303 w_k1_MSFT_s002
                                 + 1.06426 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.22032 w_k1_ADM_s002
                                 + 0.830634 w_k1_AXP_s002
                                 + 0.928996 w_k1_DIS_s002
                                 + 0.891844 w_k1_FDX_s002
                                 + 1.25064 w_k1_IBM_s002
                                 + 0.794276 w_k1_INTC_s002
                                 + 1.04482 w_k1_KO_s002 + 1.06414 w_k1_MAT_s002
                                 + 0.883819 w_k1_MSFT_s002
                                 + 0.999576 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.41878 w_k1_ADM_s002
                                 + 1.28323 w_k1_AXP_s002
                                 + 1.56647 w_k1_DIS_s002
                                 + 1.36264 w_k1_FDX_s002
                                 + 0.991114 w_k1_IBM_s002
                                 + 1.17102 w_k1_INTC_s002
                                 + 1.42684 w_k1_KO_s002 + 1.10543 w_k1_MAT_s002
                                 + 1.52016 w_k1_MSFT_s002
                                 + 1.19573 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.03826 w_k1_ADM_s002
                                 + 0.88939 w_k1_AXP_s002
                                 + 0.985543 w_k1_DIS_s002
                                 + 0.961525 w_k1_FDX_s002
                                 + 0.926255 w_k1_IBM_s002
                                 + 0.947313 w_k1_INTC_s002
                                 + 0.953519 w_k1_KO_s002
                                 + 1.07174 w_k1_MAT_s002
                                 + 1.17935 w_k1_MSFT_s002
                                 + 0.938359 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.01703 w_k1_ADM_s002
                                 + 0.84526 w_k1_AXP_s002
                                 + 0.759388 w_k1_DIS_s002
                                 + 0.79022 w_k1_FDX_s002
                                 + 1.37172 w_k1_IBM_s002
                                 + 0.803418 w_k1_INTC_s002
                                 + 1.01107 w_k1_KO_s002 + 1.38964 w_k1_MAT_s002
                                 + 0.893701 w_k1_MSFT_s002
                                 + 0.851824 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.13566 w_k1_ADM_s002
                                 + 1.03486 w_k1_AXP_s002
                                 + 1.02586 w_k1_DIS_s002
                                 + 1.18151 w_k1_FDX_s002
                                 + 0.808765 w_k1_IBM_s002
                                 + 0.905725 w_k1_INTC_s002
                                 + 0.94238 w_k1_KO_s002
                                 + 0.891045 w_k1_MAT_s002
                                 + 1.21655 w_k1_MSFT_s002
                                 + 1.05893 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.43585 w_k1_ADM_s002
                                 + 1.22458 w_k1_AXP_s002
                                 + 1.51831 w_k1_DIS_s002
                                 + 1.31529 w_k1_FDX_s002
                                 + 1.44507 w_k1_IBM_s002
                                 + 1.27995 w_k1_INTC_s002
                                 + 1.49361 w_k1_KO_s002 + 1.37073 w_k1_MAT_s002
                                 + 1.40228 w_k1_MSFT_s002 + 1.2885 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.660832 w_k1_ADM_s003
                                 + 0.815345 w_k1_AXP_s003
                                 + 0.868274 w_k1_DIS_s003
                                 + 0.758536 w_k1_FDX_s003
                                 + 1.10505 w_k1_IBM_s003
                                 + 0.746945 w_k1_INTC_s003
                                 + 0.971165 w_k1_KO_s003
                                 + 1.16099 w_k1_MAT_s003
                                 + 0.851768 w_k1_MSFT_s003
                                 + 0.859434 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.727349 w_k1_ADM_s003
                                 + 0.668869 w_k1_AXP_s003
                                 + 0.916297 w_k1_DIS_s003
                                 + 0.799751 w_k1_FDX_s003
                                 + 1.21739 w_k1_IBM_s003
                                 + 0.794619 w_k1_INTC_s003
                                 + 0.985663 w_k1_KO_s003 + 1.2004 w_k1_MAT_s003
                                 + 0.976713 w_k1_MSFT_s003
                                 + 0.900049 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.95111 w_k1_ADM_s003
                                 + 1.14685 w_k1_AXP_s003
                                 + 1.47313 w_k1_DIS_s003
                                 + 1.14177 w_k1_FDX_s003
                                 + 1.64819 w_k1_IBM_s003
                                 + 1.12301 w_k1_INTC_s003
                                 + 1.51057 w_k1_KO_s003 + 1.68957 w_k1_MAT_s003
                                 + 1.23886 w_k1_MSFT_s003
                                 + 1.16698 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.848227 w_k1_ADM_s003
                                 + 1.08096 w_k1_AXP_s003
                                 + 0.989452 w_k1_DIS_s003
                                 + 0.971927 w_k1_FDX_s003
                                 + 1.36696 w_k1_IBM_s003
                                 + 0.99619 w_k1_INTC_s003
                                 + 1.10272 w_k1_KO_s003 + 1.34048 w_k1_MAT_s003
                                 + 0.963337 w_k1_MSFT_s003
                                 + 1.04054 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.949943 w_k1_ADM_s003
                                 + 0.865251 w_k1_AXP_s003
                                 + 0.903459 w_k1_DIS_s003
                                 + 0.935163 w_k1_FDX_s003
                                 + 0.831548 w_k1_IBM_s003
                                 + 0.951763 w_k1_INTC_s003
                                 + 0.876441 w_k1_KO_s003
                                 + 0.746999 w_k1_MAT_s003
                                 + 0.958214 w_k1_MSFT_s003
                                 + 0.944542 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.572095 w_k1_ADM_s003
                                 + 0.723449 w_k1_AXP_s003
                                 + 0.545875 w_k1_DIS_s003
                                 + 0.705979 w_k1_FDX_s003
                                 + 0.794882 w_k1_IBM_s003
                                 + 0.667063 w_k1_INTC_s003
                                 + 0.609675 w_k1_KO_s003
                                 + 0.965503 w_k1_MAT_s003
                                 + 0.6116 w_k1_MSFT_s003
                                 + 0.761732 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.29353 w_k1_ADM_s003
                                 + 1.1088 w_k1_AXP_s003 + 1.2567 w_k1_DIS_s003
                                 + 1.24427 w_k1_FDX_s003
                                 + 0.953685 w_k1_IBM_s003
                                 + 0.918088 w_k1_INTC_s003
                                 + 1.03421 w_k1_KO_s003 + 1.34736 w_k1_MAT_s003
                                 + 1.2145 w_k1_MSFT_s003 + 1.14706 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.26023 w_k1_ADM_s003
                                 + 1.26928 w_k1_AXP_s003
                                 + 1.11805 w_k1_DIS_s003
                                 + 1.16627 w_k1_FDX_s003
                                 + 1.15756 w_k1_IBM_s003
                                 + 0.870086 w_k1_INTC_s003
                                 + 1.1325 w_k1_KO_s003 + 1.46657 w_k1_MAT_s003
                                 + 1.0749 w_k1_MSFT_s003 + 1.02267 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.04469 w_k1_ADM_s004
                                 + 1.59414 w_k1_AXP_s004
                                 + 1.37145 w_k1_DIS_s004
                                 + 1.08022 w_k1_FDX_s004
                                 + 1.58655 w_k1_IBM_s004
                                 + 1.20731 w_k1_INTC_s004
                                 + 1.30246 w_k1_KO_s004 + 1.76412 w_k1_MAT_s004
                                 + 1.08155 w_k1_MSFT_s004
                                 + 1.42328 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.733592 w_k1_ADM_s004
                                 + 1.01375 w_k1_AXP_s004
                                 + 1.30742 w_k1_DIS_s004
                                 + 0.952574 w_k1_FDX_s004
                                 + 1.3729 w_k1_IBM_s004
                                 + 1.05462 w_k1_INTC_s004
                                 + 1.39584 w_k1_KO_s004 + 1.39627 w_k1_MAT_s004
                                 + 0.98985 w_k1_MSFT_s004 + 1.1793 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.985096 w_k1_ADM_s004
                                 + 1.02164 w_k1_AXP_s004
                                 + 0.862443 w_k1_DIS_s004
                                 + 1.0642 w_k1_FDX_s004
                                 + 0.733499 w_k1_IBM_s004
                                 + 1.20382 w_k1_INTC_s004
                                 + 0.873616 w_k1_KO_s004
                                 + 0.56908 w_k1_MAT_s004
                                 + 1.04687 w_k1_MSFT_s004
                                 + 1.09057 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.956261 w_k1_ADM_s004
                                 + 0.850867 w_k1_AXP_s004
                                 + 1.11134 w_k1_DIS_s004
                                 + 0.946788 w_k1_FDX_s004
                                 + 1.04416 w_k1_IBM_s004
                                 + 1.04036 w_k1_INTC_s004
                                 + 1.10966 w_k1_KO_s004
                                 + 0.949331 w_k1_MAT_s004
                                 + 0.88236 w_k1_MSFT_s004
                                 + 1.00239 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.641183 w_k1_ADM_s004
                                 + 0.763558 w_k1_AXP_s004
                                 + 0.693839 w_k1_DIS_s004
                                 + 0.811841 w_k1_FDX_s004
                                 + 0.885296 w_k1_IBM_s004
                                 + 0.930029 w_k1_INTC_s004
                                 + 0.943237 w_k1_KO_s004
                                 + 1.07698 w_k1_MAT_s004
                                 + 0.748758 w_k1_MSFT_s004
                                 + 0.930323 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.00393 w_k1_ADM_s004
                                 + 1.23985 w_k1_AXP_s004
                                 + 1.41435 w_k1_DIS_s004
                                 + 1.26396 w_k1_FDX_s004
                                 + 1.06299 w_k1_IBM_s004
                                 + 1.3309 w_k1_INTC_s004 + 1.25784 w_k1_KO_s004
                                 + 0.89904 w_k1_MAT_s004
                                 + 1.17659 w_k1_MSFT_s004
                                 + 1.25203 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.844815 w_k1_ADM_s004
                                 + 0.967291 w_k1_AXP_s004
                                 + 0.903068 w_k1_DIS_s004
                                 + 0.953779 w_k1_FDX_s004
                                 + 0.863656 w_k1_IBM_s004
                                 + 1.05014 w_k1_INTC_s004
                                 + 0.882881 w_k1_KO_s004
                                 + 0.918373 w_k1_MAT_s004
                                 + 0.910405 w_k1_MSFT_s004
                                 + 0.991817 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.922102 w_k1_ADM_s004
                                 + 1.36738 w_k1_AXP_s004 + 1.1793 w_k1_DIS_s004
                                 + 1.15418 w_k1_FDX_s004
                                 + 0.990776 w_k1_IBM_s004
                                 + 1.5301 w_k1_INTC_s004 + 1.19788 w_k1_KO_s004
                                 + 1.26058 w_k1_MAT_s004
                                 + 1.01651 w_k1_MSFT_s004
                                 + 1.14165 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.18358 w_k1_ADM_s005
                                 + 1.36191 w_k1_AXP_s005
                                 + 1.52433 w_k1_DIS_s005
                                 + 1.19907 w_k1_FDX_s005
                                 + 1.54378 w_k1_IBM_s005
                                 + 1.28053 w_k1_INTC_s005
                                 + 1.43902 w_k1_KO_s005 + 1.60069 w_k1_MAT_s005
                                 + 1.16065 w_k1_MSFT_s005
                                 + 1.31233 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.725817 w_k1_ADM_s005
                                 + 0.71246 w_k1_AXP_s005
                                 + 0.649671 w_k1_DIS_s005
                                 + 0.651599 w_k1_FDX_s005
                                 + 1.36619 w_k1_IBM_s005
                                 + 0.721111 w_k1_INTC_s005
                                 + 0.995443 w_k1_KO_s005
                                 + 1.49884 w_k1_MAT_s005
                                 + 0.719991 w_k1_MSFT_s005
                                 + 0.933844 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.19961 w_k1_ADM_s005
                                 + 1.00906 w_k1_AXP_s005
                                 + 1.59391 w_k1_DIS_s005
                                 + 1.35909 w_k1_FDX_s005
                                 + 0.833339 w_k1_IBM_s005
                                 + 1.49634 w_k1_INTC_s005
                                 + 1.22545 w_k1_KO_s005
                                 + 0.648211 w_k1_MAT_s005
                                 + 1.4227 w_k1_MSFT_s005 + 1.20984 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.714139 w_k1_ADM_s005
                                 + 0.761712 w_k1_AXP_s005
                                 + 0.831967 w_k1_DIS_s005
                                 + 0.832811 w_k1_FDX_s005
                                 + 1.03496 w_k1_IBM_s005
                                 + 0.872685 w_k1_INTC_s005
                                 + 1.00547 w_k1_KO_s005
                                 + 0.900148 w_k1_MAT_s005
                                 + 0.849081 w_k1_MSFT_s005
                                 + 1.02215 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.803751 w_k1_ADM_s005
                                 + 0.929989 w_k1_AXP_s005
                                 + 1.08808 w_k1_DIS_s005
                                 + 0.974912 w_k1_FDX_s005
                                 + 1.07279 w_k1_IBM_s005
                                 + 0.853498 w_k1_INTC_s005
                                 + 1.05231 w_k1_KO_s005 + 1.17226 w_k1_MAT_s005
                                 + 0.906648 w_k1_MSFT_s005
                                 + 1.1534 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.597319 w_k1_ADM_s005
                                 + 0.669672 w_k1_AXP_s005
                                 + 0.623056 w_k1_DIS_s005
                                 + 0.77855 w_k1_FDX_s005
                                 + 0.696951 w_k1_IBM_s005
                                 + 0.726169 w_k1_INTC_s005
                                 + 0.693631 w_k1_KO_s005
                                 + 0.799308 w_k1_MAT_s005
                                 + 0.722074 w_k1_MSFT_s005
                                 + 0.861447 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.06922 w_k1_ADM_s005
                                 + 0.814853 w_k1_AXP_s005
                                 + 0.931917 w_k1_DIS_s005
                                 + 0.924978 w_k1_FDX_s005
                                 + 0.799779 w_k1_IBM_s005
                                 + 1.07322 w_k1_INTC_s005
                                 + 0.986461 w_k1_KO_s005
                                 + 0.691362 w_k1_MAT_s005
                                 + 0.918967 w_k1_MSFT_s005
                                 + 1.04379 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.541382 w_k1_ADM_s005
                                 + 0.559117 w_k1_AXP_s005
                                 + 0.60582 w_k1_DIS_s005
                                 + 0.550534 w_k1_FDX_s005
                                 + 0.947867 w_k1_IBM_s005
                                 + 0.734155 w_k1_INTC_s005
                                 + 0.854767 w_k1_KO_s005
                                 + 0.838489 w_k1_MAT_s005
                                 + 0.695054 w_k1_MSFT_s005
                                 + 0.87953 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00007 w_k1_ADM_s006
                                 + 1.17377 w_k1_AXP_s006
                                 + 1.70993 w_k1_DIS_s006
                                 + 1.13581 w_k1_FDX_s006
                                 + 1.56477 w_k1_IBM_s006 + 1.188 w_k1_INTC_s006
                                 + 1.50772 w_k1_KO_s006 + 1.65377 w_k1_MAT_s006
                                 + 1.35608 w_k1_MSFT_s006
                                 + 1.24448 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.03586 w_k1_ADM_s006
                                 + 0.909845 w_k1_AXP_s006
                                 + 0.990757 w_k1_DIS_s006
                                 + 0.809257 w_k1_FDX_s006
                                 + 1.10186 w_k1_IBM_s006
                                 + 0.931184 w_k1_INTC_s006
                                 + 1.11699 w_k1_KO_s006
                                 + 0.933245 w_k1_MAT_s006
                                 + 0.941701 w_k1_MSFT_s006
                                 + 0.858428 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.703759 w_k1_ADM_s006
                                 + 0.756442 w_k1_AXP_s006
                                 + 0.74631 w_k1_DIS_s006
                                 + 0.819505 w_k1_FDX_s006
                                 + 0.72356 w_k1_IBM_s006
                                 + 0.884811 w_k1_INTC_s006
                                 + 0.843988 w_k1_KO_s006
                                 + 0.646864 w_k1_MAT_s006
                                 + 0.865098 w_k1_MSFT_s006
                                 + 0.88301 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.09427 w_k1_ADM_s006
                                 + 1.04067 w_k1_AXP_s006
                                 + 0.986984 w_k1_DIS_s006
                                 + 1.00611 w_k1_FDX_s006
                                 + 1.04341 w_k1_IBM_s006
                                 + 0.933216 w_k1_INTC_s006
                                 + 0.970164 w_k1_KO_s006
                                 + 0.967075 w_k1_MAT_s006
                                 + 0.93752 w_k1_MSFT_s006
                                 + 0.952293 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.652304 w_k1_ADM_s006
                                 + 0.831997 w_k1_AXP_s006
                                 + 1.17659 w_k1_DIS_s006
                                 + 0.952979 w_k1_FDX_s006
                                 + 1.08563 w_k1_IBM_s006
                                 + 1.04418 w_k1_INTC_s006
                                 + 1.13651 w_k1_KO_s006 + 1.10418 w_k1_MAT_s006
                                 + 1.13996 w_k1_MSFT_s006
                                 + 0.968188 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.905502 w_k1_ADM_s006
                                 + 1.07721 w_k1_AXP_s006
                                 + 0.705178 w_k1_DIS_s006
                                 + 0.83551 w_k1_FDX_s006
                                 + 1.03183 w_k1_IBM_s006
                                 + 0.891679 w_k1_INTC_s006
                                 + 0.7725 w_k1_KO_s006 + 1.19358 w_k1_MAT_s006
                                 + 0.675373 w_k1_MSFT_s006
                                 + 0.804401 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.28165 w_k1_ADM_s006
                                 + 1.41922 w_k1_AXP_s006
                                 + 1.38645 w_k1_DIS_s006
                                 + 1.21852 w_k1_FDX_s006 + 1.0871 w_k1_IBM_s006
                                 + 1.02802 w_k1_INTC_s006
                                 + 1.29995 w_k1_KO_s006 + 1.49008 w_k1_MAT_s006
                                 + 1.13798 w_k1_MSFT_s006
                                 + 1.14072 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.966003 w_k1_ADM_s006
                                 + 1.18734 w_k1_AXP_s006
                                 + 1.03583 w_k1_DIS_s006
                                 + 1.04435 w_k1_FDX_s006
                                 + 0.889889 w_k1_IBM_s006
                                 + 0.886857 w_k1_INTC_s006
                                 + 0.906102 w_k1_KO_s006
                                 + 0.887975 w_k1_MAT_s006
                                 + 1.02262 w_k1_MSFT_s006
                                 + 0.917946 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.48999 w_k1_ADM_s007
                                 + 1.3862 w_k1_AXP_s007 + 1.91718 w_k1_DIS_s007
                                 + 1.47141 w_k1_FDX_s007
                                 + 1.16236 w_k1_IBM_s007
                                 + 1.16864 w_k1_INTC_s007
                                 + 1.45316 w_k1_KO_s007 + 1.281 w_k1_MAT_s007
                                 + 1.2261 w_k1_MSFT_s007 + 1.27257 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.977874 w_k1_ADM_s007
                                 + 1.26276 w_k1_AXP_s007
                                 + 1.05852 w_k1_DIS_s007
                                 + 1.04627 w_k1_FDX_s007
                                 + 0.934395 w_k1_IBM_s007
                                 + 0.968737 w_k1_INTC_s007
                                 + 0.993065 w_k1_KO_s007
                                 + 1.06222 w_k1_MAT_s007
                                 + 0.822368 w_k1_MSFT_s007
                                 + 1.04748 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.969809 w_k1_ADM_s007
                                 + 0.626935 w_k1_AXP_s007
                                 + 0.721545 w_k1_DIS_s007
                                 + 0.746522 w_k1_FDX_s007
                                 + 0.897413 w_k1_IBM_s007
                                 + 0.652082 w_k1_INTC_s007
                                 + 0.831668 w_k1_KO_s007
                                 + 0.771264 w_k1_MAT_s007
                                 + 0.761344 w_k1_MSFT_s007
                                 + 0.953005 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.48296 w_k1_ADM_s007
                                 + 1.15001 w_k1_AXP_s007
                                 + 1.12414 w_k1_DIS_s007
                                 + 1.29741 w_k1_FDX_s007
                                 + 0.659579 w_k1_IBM_s007
                                 + 0.830646 w_k1_INTC_s007
                                 + 0.947891 w_k1_KO_s007
                                 + 0.831869 w_k1_MAT_s007
                                 + 0.910084 w_k1_MSFT_s007
                                 + 1.15265 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.18283 w_k1_ADM_s007
                                 + 0.778534 w_k1_AXP_s007
                                 + 1.27168 w_k1_DIS_s007
                                 + 1.04877 w_k1_FDX_s007
                                 + 1.18673 w_k1_IBM_s007
                                 + 0.89389 w_k1_INTC_s007
                                 + 1.24434 w_k1_KO_s007 + 1.01726 w_k1_MAT_s007
                                 + 0.960999 w_k1_MSFT_s007
                                 + 1.0594 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.921033 w_k1_ADM_s007
                                 + 0.999818 w_k1_AXP_s007
                                 + 1.0764 w_k1_DIS_s007
                                 + 0.936705 w_k1_FDX_s007
                                 + 1.77288 w_k1_IBM_s007
                                 + 0.812849 w_k1_INTC_s007
                                 + 1.29922 w_k1_KO_s007 + 1.50997 w_k1_MAT_s007
                                 + 0.837515 w_k1_MSFT_s007
                                 + 1.05733 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.07989 w_k1_ADM_s007
                                 + 0.849446 w_k1_AXP_s007
                                 + 1.10308 w_k1_DIS_s007
                                 + 0.998649 w_k1_FDX_s007
                                 + 0.715887 w_k1_IBM_s007
                                 + 0.913272 w_k1_INTC_s007
                                 + 0.86909 w_k1_KO_s007 + 0.67635 w_k1_MAT_s007
                                 + 0.990305 w_k1_MSFT_s007
                                 + 0.969523 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.925564 w_k1_ADM_s007
                                 + 0.709682 w_k1_AXP_s007
                                 + 0.746532 w_k1_DIS_s007
                                 + 0.88928 w_k1_FDX_s007
                                 + 0.687504 w_k1_IBM_s007
                                 + 0.878433 w_k1_INTC_s007
                                 + 0.923822 w_k1_KO_s007
                                 + 0.50352 w_k1_MAT_s007
                                 + 0.836924 w_k1_MSFT_s007
                                 + 0.931838 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.839396 w_k1_ADM_s010
                                 + 0.712507 w_k1_AXP_s010
                                 + 0.696186 w_k1_DIS_s010
                                 + 0.862797 w_k1_FDX_s010
                                 + 0.646074 w_k1_IBM_s010
                                 + 0.821754 w_k1_INTC_s010
                                 + 0.781495 w_k1_KO_s010
                                 + 0.785934 w_k1_MAT_s010
                                 + 0.680949 w_k1_MSFT_s010
                                 + 0.901304 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.27122 w_k1_ADM_s010
                                 + 1.11674 w_k1_AXP_s010
                                 + 0.828668 w_k1_DIS_s010
                                 + 0.931898 w_k1_FDX_s010
                                 + 1.17804 w_k1_IBM_s010
                                 + 0.835195 w_k1_INTC_s010
                                 + 1.02298 w_k1_KO_s010 + 1.38067 w_k1_MAT_s010
                                 + 0.732481 w_k1_MSFT_s010
                                 + 0.965226 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.865549 w_k1_ADM_s010
                                 + 0.814548 w_k1_AXP_s010
                                 + 0.800663 w_k1_DIS_s010
                                 + 0.811342 w_k1_FDX_s010
                                 + 1.25039 w_k1_IBM_s010
                                 + 0.93758 w_k1_INTC_s010 + 1.0881 w_k1_KO_s010
                                 + 1.17221 w_k1_MAT_s010
                                 + 0.820317 w_k1_MSFT_s010
                                 + 0.968006 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.13139 w_k1_ADM_s010
                                 + 1.10844 w_k1_AXP_s010
                                 + 1.33837 w_k1_DIS_s010
                                 + 1.30506 w_k1_FDX_s010
                                 + 0.806427 w_k1_IBM_s010
                                 + 1.06364 w_k1_INTC_s010
                                 + 0.991749 w_k1_KO_s010
                                 + 1.21728 w_k1_MAT_s010
                                 + 1.13248 w_k1_MSFT_s010
                                 + 0.985514 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.35024 w_k1_ADM_s010
                                 + 1.42821 w_k1_AXP_s010
                                 + 1.39306 w_k1_DIS_s010
                                 + 1.19828 w_k1_FDX_s010
                                 + 1.70091 w_k1_IBM_s010
                                 + 0.909868 w_k1_INTC_s010
                                 + 1.40016 w_k1_KO_s010 + 2.18337 w_k1_MAT_s010
                                 + 0.992735 w_k1_MSFT_s010
                                 + 1.19235 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.35286 w_k1_ADM_s010
                                 + 1.57212 w_k1_AXP_s010
                                 + 1.56962 w_k1_DIS_s010
                                 + 1.30619 w_k1_FDX_s010
                                 + 1.28633 w_k1_IBM_s010
                                 + 1.31736 w_k1_INTC_s010
                                 + 1.29603 w_k1_KO_s010 + 1.83197 w_k1_MAT_s010
                                 + 1.12725 w_k1_MSFT_s010
                                 + 1.18407 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.887564 w_k1_ADM_s010
                                 + 0.695817 w_k1_AXP_s010
                                 + 0.822607 w_k1_DIS_s010
                                 + 0.917133 w_k1_FDX_s010
                                 + 0.847204 w_k1_IBM_s010
                                 + 0.867833 w_k1_INTC_s010
                                 + 0.936965 w_k1_KO_s010
                                 + 0.866764 w_k1_MAT_s010
                                 + 0.799529 w_k1_MSFT_s010
                                 + 0.874533 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.959447 w_k1_ADM_s010
                                 + 1.0988 w_k1_AXP_s010 + 1.06134 w_k1_DIS_s010
                                 + 1.15346 w_k1_FDX_s010
                                 + 1.01513 w_k1_IBM_s010
                                 + 1.12744 w_k1_INTC_s010
                                 + 0.98544 w_k1_KO_s010 + 1.21308 w_k1_MAT_s010
                                 + 0.942851 w_k1_MSFT_s010
                                 + 0.997561 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.31718 w_k1_ADM_s011
                                 + 1.29037 w_k1_AXP_s011
                                 + 1.55801 w_k1_DIS_s011
                                 + 1.36085 w_k1_FDX_s011
                                 + 1.36792 w_k1_IBM_s011
                                 + 1.1112 w_k1_INTC_s011 + 1.51835 w_k1_KO_s011
                                 + 1.81201 w_k1_MAT_s011
                                 + 1.41128 w_k1_MSFT_s011
                                 + 1.29447 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.03872 w_k1_ADM_s011
                                 + 0.958002 w_k1_AXP_s011
                                 + 1.01357 w_k1_DIS_s011
                                 + 1.02278 w_k1_FDX_s011
                                 + 0.960583 w_k1_IBM_s011
                                 + 1.06706 w_k1_INTC_s011 + 1.154 w_k1_KO_s011
                                 + 0.910174 w_k1_MAT_s011
                                 + 1.17304 w_k1_MSFT_s011
                                 + 1.01401 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.95495 w_k1_ADM_s011
                                 + 0.922778 w_k1_AXP_s011
                                 + 0.614069 w_k1_DIS_s011
                                 + 0.829812 w_k1_FDX_s011
                                 + 0.77506 w_k1_IBM_s011
                                 + 0.787319 w_k1_INTC_s011
                                 + 0.829738 w_k1_KO_s011
                                 + 0.649254 w_k1_MAT_s011
                                 + 0.775714 w_k1_MSFT_s011
                                 + 0.813728 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.851629 w_k1_ADM_s011
                                 + 0.780195 w_k1_AXP_s011
                                 + 0.932635 w_k1_DIS_s011
                                 + 0.822505 w_k1_FDX_s011
                                 + 1.51387 w_k1_IBM_s011
                                 + 0.855925 w_k1_INTC_s011
                                 + 1.26406 w_k1_KO_s011 + 1.72177 w_k1_MAT_s011
                                 + 0.952822 w_k1_MSFT_s011
                                 + 1.05882 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.20438 w_k1_ADM_s011
                                 + 0.888889 w_k1_AXP_s011
                                 + 0.904255 w_k1_DIS_s011
                                 + 1.11915 w_k1_FDX_s011
                                 + 0.752107 w_k1_IBM_s011
                                 + 0.887651 w_k1_INTC_s011
                                 + 0.906095 w_k1_KO_s011
                                 + 0.660155 w_k1_MAT_s011
                                 + 1.03848 w_k1_MSFT_s011
                                 + 0.939926 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.913251 w_k1_ADM_s011
                                 + 0.743671 w_k1_AXP_s011
                                 + 0.77813 w_k1_DIS_s011
                                 + 0.816778 w_k1_FDX_s011
                                 + 1.09358 w_k1_IBM_s011
                                 + 0.849093 w_k1_INTC_s011
                                 + 0.978765 w_k1_KO_s011 + 1.1078 w_k1_MAT_s011
                                 + 0.867831 w_k1_MSFT_s011
                                 + 0.903214 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.04247 w_k1_ADM_s011
                                 + 1.17006 w_k1_AXP_s011
                                 + 1.59692 w_k1_DIS_s011
                                 + 1.38476 w_k1_FDX_s011
                                 + 0.819124 w_k1_IBM_s011
                                 + 1.09639 w_k1_INTC_s011
                                 + 1.07472 w_k1_KO_s011
                                 + 0.679819 w_k1_MAT_s011
                                 + 1.27458 w_k1_MSFT_s011 + 1.015 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.45267 w_k1_ADM_s011
                                 + 0.963775 w_k1_AXP_s011
                                 + 1.63315 w_k1_DIS_s011
                                 + 1.23535 w_k1_FDX_s011
                                 + 1.12445 w_k1_IBM_s011
                                 + 1.02076 w_k1_INTC_s011
                                 + 1.32049 w_k1_KO_s011 + 1.18776 w_k1_MAT_s011
                                 + 1.25916 w_k1_MSFT_s011
                                 + 1.08217 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.03208 w_k1_ADM_s012
                                 + 0.916826 w_k1_AXP_s012
                                 + 1.01544 w_k1_DIS_s012
                                 + 0.987405 w_k1_FDX_s012
                                 + 0.873384 w_k1_IBM_s012
                                 + 1.12617 w_k1_INTC_s012
                                 + 0.96549 w_k1_KO_s012
                                 + 0.799269 w_k1_MAT_s012
                                 + 1.09159 w_k1_MSFT_s012
                                 + 1.09172 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.00622 w_k1_ADM_s012
                                 + 1.11306 w_k1_AXP_s012
                                 + 1.86407 w_k1_DIS_s012
                                 + 1.25699 w_k1_FDX_s012 + 1.2474 w_k1_IBM_s012
                                 + 1.33954 w_k1_INTC_s012 + 1.4939 w_k1_KO_s012
                                 + 1.36065 w_k1_MAT_s012
                                 + 1.28466 w_k1_MSFT_s012
                                 + 1.23172 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.43701 w_k1_ADM_s012
                                 + 1.49968 w_k1_AXP_s012
                                 + 1.43295 w_k1_DIS_s012
                                 + 1.34924 w_k1_FDX_s012
                                 + 0.965281 w_k1_IBM_s012
                                 + 1.17446 w_k1_INTC_s012
                                 + 1.06553 w_k1_KO_s012 + 1.10532 w_k1_MAT_s012
                                 + 1.21312 w_k1_MSFT_s012
                                 + 1.26264 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.841002 w_k1_ADM_s012
                                 + 0.88321 w_k1_AXP_s012
                                 + 0.784793 w_k1_DIS_s012
                                 + 0.788053 w_k1_FDX_s012
                                 + 0.735512 w_k1_IBM_s012
                                 + 0.744638 w_k1_INTC_s012
                                 + 0.746488 w_k1_KO_s012
                                 + 1.10508 w_k1_MAT_s012
                                 + 0.863822 w_k1_MSFT_s012
                                 + 0.813695 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.15969 w_k1_ADM_s012
                                 + 1.08398 w_k1_AXP_s012
                                 + 1.31167 w_k1_DIS_s012
                                 + 1.10726 w_k1_FDX_s012
                                 + 1.29826 w_k1_IBM_s012
                                 + 1.21061 w_k1_INTC_s012
                                 + 1.30409 w_k1_KO_s012 + 1.17348 w_k1_MAT_s012
                                 + 1.25039 w_k1_MSFT_s012
                                 + 1.16871 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.875342 w_k1_ADM_s012
                                 + 0.909801 w_k1_AXP_s012
                                 + 1.11412 w_k1_DIS_s012
                                 + 0.855517 w_k1_FDX_s012
                                 + 1.04089 w_k1_IBM_s012
                                 + 0.945449 w_k1_INTC_s012
                                 + 1.18966 w_k1_KO_s012 + 1.07811 w_k1_MAT_s012
                                 + 1.01869 w_k1_MSFT_s012
                                 + 1.11856 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.990745 w_k1_ADM_s012
                                 + 1.40755 w_k1_AXP_s012 + 1.1252 w_k1_DIS_s012
                                 + 1.08795 w_k1_FDX_s012
                                 + 1.29175 w_k1_IBM_s012
                                 + 1.03799 w_k1_INTC_s012
                                 + 1.24974 w_k1_KO_s012 + 1.68116 w_k1_MAT_s012
                                 + 1.09021 w_k1_MSFT_s012
                                 + 1.20671 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.72082 w_k1_ADM_s012
                                 + 0.883189 w_k1_AXP_s012
                                 + 0.692239 w_k1_DIS_s012
                                 + 0.799526 w_k1_FDX_s012
                                 + 1.1021 w_k1_IBM_s012
                                 + 0.88334 w_k1_INTC_s012
                                 + 0.858387 w_k1_KO_s012
                                 + 1.19063 w_k1_MAT_s012
                                 + 0.81162 w_k1_MSFT_s012
                                 + 0.925427 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.35532 w_k1_ADM_s013
                                 + 1.17132 w_k1_AXP_s013
                                 + 1.20706 w_k1_DIS_s013
                                 + 1.12672 w_k1_FDX_s013
                                 + 1.14306 w_k1_IBM_s013
                                 + 1.09857 w_k1_INTC_s013
                                 + 1.32408 w_k1_KO_s013 + 1.1403 w_k1_MAT_s013
                                 + 1.11154 w_k1_MSFT_s013
                                 + 1.13857 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.06738 w_k1_ADM_s013
                                 + 1.10112 w_k1_AXP_s013
                                 + 0.897708 w_k1_DIS_s013
                                 + 0.947941 w_k1_FDX_s013
                                 + 1.02592 w_k1_IBM_s013
                                 + 0.89428 w_k1_INTC_s013
                                 + 0.95049 w_k1_KO_s013 + 1.18817 w_k1_MAT_s013
                                 + 0.995402 w_k1_MSFT_s013
                                 + 0.940095 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.20856 w_k1_ADM_s013
                                 + 1.15958 w_k1_AXP_s013
                                 + 1.24999 w_k1_DIS_s013
                                 + 1.32489 w_k1_FDX_s013
                                 + 0.782275 w_k1_IBM_s013
                                 + 1.28839 w_k1_INTC_s013
                                 + 1.08032 w_k1_KO_s013
                                 + 0.860238 w_k1_MAT_s013
                                 + 1.25589 w_k1_MSFT_s013
                                 + 0.988296 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.206 w_k1_ADM_s013
                                 + 1.16533 w_k1_AXP_s013
                                 + 1.17149 w_k1_DIS_s013 + 1.0322 w_k1_FDX_s013
                                 + 1.66781 w_k1_IBM_s013
                                 + 1.01419 w_k1_INTC_s013
                                 + 1.24604 w_k1_KO_s013 + 1.48348 w_k1_MAT_s013
                                 + 1.01015 w_k1_MSFT_s013
                                 + 1.08538 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.13517 w_k1_ADM_s013
                                 + 0.833729 w_k1_AXP_s013
                                 + 0.713981 w_k1_DIS_s013
                                 + 0.826253 w_k1_FDX_s013
                                 + 0.746743 w_k1_IBM_s013
                                 + 0.829276 w_k1_INTC_s013
                                 + 0.898254 w_k1_KO_s013
                                 + 0.804458 w_k1_MAT_s013
                                 + 0.867517 w_k1_MSFT_s013
                                 + 0.754666 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.770304 w_k1_ADM_s013
                                 + 0.746261 w_k1_AXP_s013
                                 + 0.776666 w_k1_DIS_s013
                                 + 0.808272 w_k1_FDX_s013
                                 + 1.00028 w_k1_IBM_s013
                                 + 0.957001 w_k1_INTC_s013
                                 + 1.00135 w_k1_KO_s013 + 0.89293 w_k1_MAT_s013
                                 + 0.940047 w_k1_MSFT_s013
                                 + 0.786436 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.980125 w_k1_ADM_s013
                                 + 0.824256 w_k1_AXP_s013
                                 + 0.79939 w_k1_DIS_s013
                                 + 0.796425 w_k1_FDX_s013
                                 + 1.24564 w_k1_IBM_s013
                                 + 0.930172 w_k1_INTC_s013
                                 + 1.13373 w_k1_KO_s013 + 1.03294 w_k1_MAT_s013
                                 + 0.902081 w_k1_MSFT_s013
                                 + 0.930991 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.63986 w_k1_ADM_s013
                                 + 1.75297 w_k1_AXP_s013
                                 + 1.58861 w_k1_DIS_s013 + 1.3173 w_k1_FDX_s013
                                 + 1.31312 w_k1_IBM_s013
                                 + 1.24187 w_k1_INTC_s013
                                 + 1.36622 w_k1_KO_s013 + 1.64969 w_k1_MAT_s013
                                 + 1.2336 w_k1_MSFT_s013 + 1.20616 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.02701 w_k1_ADM_s014
                                 + 0.816187 w_k1_AXP_s014
                                 + 1.27919 w_k1_DIS_s014
                                 + 0.951368 w_k1_FDX_s014
                                 + 1.23345 w_k1_IBM_s014
                                 + 0.730953 w_k1_INTC_s014
                                 + 1.12846 w_k1_KO_s014 + 1.35175 w_k1_MAT_s014
                                 + 0.950576 w_k1_MSFT_s014
                                 + 1.06827 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.915554 w_k1_ADM_s014
                                 + 1.01968 w_k1_AXP_s014
                                 + 1.15566 w_k1_DIS_s014
                                 + 1.03136 w_k1_FDX_s014
                                 + 0.985971 w_k1_IBM_s014
                                 + 0.776898 w_k1_INTC_s014
                                 + 0.97423 w_k1_KO_s014 + 1.52295 w_k1_MAT_s014
                                 + 1.11788 w_k1_MSFT_s014
                                 + 1.02556 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.02633 w_k1_ADM_s014
                                 + 0.703135 w_k1_AXP_s014
                                 + 0.746522 w_k1_DIS_s014
                                 + 0.866718 w_k1_FDX_s014
                                 + 0.764862 w_k1_IBM_s014
                                 + 0.750249 w_k1_INTC_s014
                                 + 0.804078 w_k1_KO_s014
                                 + 0.842693 w_k1_MAT_s014
                                 + 0.834503 w_k1_MSFT_s014
                                 + 0.769352 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.28584 w_k1_ADM_s014
                                 + 1.16019 w_k1_AXP_s014
                                 + 1.25436 w_k1_DIS_s014
                                 + 1.28512 w_k1_FDX_s014
                                 + 0.925759 w_k1_IBM_s014
                                 + 0.946091 w_k1_INTC_s014
                                 + 1.10259 w_k1_KO_s014 + 1.0661 w_k1_MAT_s014
                                 + 1.00935 w_k1_MSFT_s014
                                 + 1.05901 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.78291 w_k1_ADM_s014
                                 + 0.8664 w_k1_AXP_s014 + 0.78537 w_k1_DIS_s014
                                 + 0.701648 w_k1_FDX_s014
                                 + 1.28949 w_k1_IBM_s014
                                 + 0.799809 w_k1_INTC_s014
                                 + 0.994032 w_k1_KO_s014
                                 + 1.36837 w_k1_MAT_s014
                                 + 0.764408 w_k1_MSFT_s014
                                 + 0.785099 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.64995 w_k1_ADM_s014
                                 + 1.41226 w_k1_AXP_s014
                                 + 1.68769 w_k1_DIS_s014
                                 + 1.37465 w_k1_FDX_s014
                                 + 1.17737 w_k1_IBM_s014
                                 + 1.10124 w_k1_INTC_s014 + 1.3037 w_k1_KO_s014
                                 + 1.5066 w_k1_MAT_s014
                                 + 1.33798 w_k1_MSFT_s014
                                 + 1.24459 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.928872 w_k1_ADM_s014
                                 + 0.980141 w_k1_AXP_s014
                                 + 0.892946 w_k1_DIS_s014
                                 + 0.952501 w_k1_FDX_s014
                                 + 0.951999 w_k1_IBM_s014
                                 + 0.825311 w_k1_INTC_s014
                                 + 0.930432 w_k1_KO_s014
                                 + 1.07483 w_k1_MAT_s014
                                 + 0.9289 w_k1_MSFT_s014
                                 + 0.880281 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.02501 w_k1_ADM_s014
                                 + 1.04138 w_k1_AXP_s014
                                 + 0.998481 w_k1_DIS_s014
                                 + 0.954888 w_k1_FDX_s014
                                 + 1.18251 w_k1_IBM_s014
                                 + 0.788746 w_k1_INTC_s014
                                 + 1.10495 w_k1_KO_s014 + 1.33996 w_k1_MAT_s014
                                 + 0.875518 w_k1_MSFT_s014
                                 + 0.981862 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.28837 w_k1_ADM_s015
                                 + 0.973203 w_k1_AXP_s015
                                 + 1.30467 w_k1_DIS_s015
                                 + 1.18189 w_k1_FDX_s015
                                 + 0.848538 w_k1_IBM_s015
                                 + 1.21422 w_k1_INTC_s015
                                 + 1.09632 w_k1_KO_s015
                                 + 0.599026 w_k1_MAT_s015
                                 + 1.11533 w_k1_MSFT_s015
                                 + 1.01333 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.30348 w_k1_ADM_s015
                                 + 1.48829 w_k1_AXP_s015
                                 + 1.22302 w_k1_DIS_s015
                                 + 1.14115 w_k1_FDX_s015
                                 + 1.58576 w_k1_IBM_s015
                                 + 1.33879 w_k1_INTC_s015
                                 + 1.32047 w_k1_KO_s015 + 1.42464 w_k1_MAT_s015
                                 + 1.14294 w_k1_MSFT_s015
                                 + 1.10202 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.466 w_k1_ADM_s015
                                 + 1.57512 w_k1_AXP_s015
                                 + 1.69917 w_k1_DIS_s015 + 1.5108 w_k1_FDX_s015
                                 + 1.0603 w_k1_IBM_s015
                                 + 1.60106 w_k1_INTC_s015
                                 + 1.25156 w_k1_KO_s015
                                 + 0.947531 w_k1_MAT_s015
                                 + 1.28159 w_k1_MSFT_s015
                                 + 1.18916 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.18506 w_k1_ADM_s015
                                 + 1.29723 w_k1_AXP_s015
                                 + 1.09936 w_k1_DIS_s015
                                 + 1.20061 w_k1_FDX_s015
                                 + 0.964532 w_k1_IBM_s015
                                 + 1.26561 w_k1_INTC_s015
                                 + 1.02091 w_k1_KO_s015 + 1.11373 w_k1_MAT_s015
                                 + 1.11222 w_k1_MSFT_s015
                                 + 1.06765 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.845662 w_k1_ADM_s015
                                 + 0.775516 w_k1_AXP_s015
                                 + 0.658727 w_k1_DIS_s015
                                 + 0.768862 w_k1_FDX_s015
                                 + 0.711209 w_k1_IBM_s015
                                 + 0.850646 w_k1_INTC_s015
                                 + 0.780728 w_k1_KO_s015
                                 + 0.596915 w_k1_MAT_s015
                                 + 0.871965 w_k1_MSFT_s015
                                 + 0.921936 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.31196 w_k1_ADM_s015
                                 + 0.990654 w_k1_AXP_s015
                                 + 0.992985 w_k1_DIS_s015
                                 + 1.02761 w_k1_FDX_s015
                                 + 0.789627 w_k1_IBM_s015
                                 + 0.982721 w_k1_INTC_s015
                                 + 0.975582 w_k1_KO_s015 + 0.7733 w_k1_MAT_s015
                                 + 0.930438 w_k1_MSFT_s015
                                 + 0.943519 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.813185 w_k1_ADM_s015
                                 + 0.753507 w_k1_AXP_s015
                                 + 0.576637 w_k1_DIS_s015
                                 + 0.690865 w_k1_FDX_s015
                                 + 1.11174 w_k1_IBM_s015
                                 + 0.850503 w_k1_INTC_s015
                                 + 0.894351 w_k1_KO_s015
                                 + 0.711851 w_k1_MAT_s015
                                 + 0.729963 w_k1_MSFT_s015
                                 + 0.819111 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.02659 w_k1_ADM_s015
                                 + 0.757386 w_k1_AXP_s015
                                 + 1.09009 w_k1_DIS_s015
                                 + 0.919458 w_k1_FDX_s015
                                 + 1.29491 w_k1_IBM_s015
                                 + 1.11418 w_k1_INTC_s015
                                 + 1.16853 w_k1_KO_s015 + 1.03676 w_k1_MAT_s015
                                 + 1.04296 w_k1_MSFT_s015
                                 + 0.941887 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.11575 w_k1_ADM_s016
                                 + 0.910775 w_k1_AXP_s016
                                 + 1.08879 w_k1_DIS_s016
                                 + 0.918031 w_k1_FDX_s016
                                 + 1.45766 w_k1_IBM_s016
                                 + 0.771468 w_k1_INTC_s016
                                 + 1.07812 w_k1_KO_s016 + 1.62421 w_k1_MAT_s016
                                 + 0.901387 w_k1_MSFT_s016
                                 + 1.0018 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.00567 w_k1_ADM_s016
                                 + 1.02358 w_k1_AXP_s016
                                 + 1.26103 w_k1_DIS_s016
                                 + 1.13194 w_k1_FDX_s016 + 1.0106 w_k1_IBM_s016
                                 + 1.10659 w_k1_INTC_s016
                                 + 1.11243 w_k1_KO_s016 + 1.19158 w_k1_MAT_s016
                                 + 1.17958 w_k1_MSFT_s016
                                 + 1.07741 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.51817 w_k1_ADM_s016
                                 + 1.53843 w_k1_AXP_s016
                                 + 1.60854 w_k1_DIS_s016
                                 + 1.34415 w_k1_FDX_s016
                                 + 1.25307 w_k1_IBM_s016
                                 + 1.33071 w_k1_INTC_s016 + 1.3267 w_k1_KO_s016
                                 + 1.65688 w_k1_MAT_s016
                                 + 1.42413 w_k1_MSFT_s016
                                 + 1.26859 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.21953 w_k1_ADM_s016
                                 + 1.06656 w_k1_AXP_s016
                                 + 1.05771 w_k1_DIS_s016
                                 + 0.967009 w_k1_FDX_s016
                                 + 1.18032 w_k1_IBM_s016
                                 + 0.980052 w_k1_INTC_s016
                                 + 1.08479 w_k1_KO_s016 + 1.0787 w_k1_MAT_s016
                                 + 0.97119 w_k1_MSFT_s016
                                 + 1.01943 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.859959 w_k1_ADM_s016
                                 + 0.931276 w_k1_AXP_s016
                                 + 0.714635 w_k1_DIS_s016
                                 + 0.769865 w_k1_FDX_s016
                                 + 1.01335 w_k1_IBM_s016
                                 + 0.917712 w_k1_INTC_s016
                                 + 1.04124 w_k1_KO_s016 + 1.30637 w_k1_MAT_s016
                                 + 0.929178 w_k1_MSFT_s016
                                 + 0.900655 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.802709 w_k1_ADM_s016
                                 + 0.740579 w_k1_AXP_s016
                                 + 0.745668 w_k1_DIS_s016
                                 + 0.732547 w_k1_FDX_s016
                                 + 0.86407 w_k1_IBM_s016
                                 + 0.766246 w_k1_INTC_s016
                                 + 0.911847 w_k1_KO_s016
                                 + 0.973859 w_k1_MAT_s016
                                 + 0.965937 w_k1_MSFT_s016
                                 + 0.901128 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.07346 w_k1_ADM_s016
                                 + 0.985764 w_k1_AXP_s016
                                 + 0.96728 w_k1_DIS_s016
                                 + 0.979251 w_k1_FDX_s016
                                 + 0.663722 w_k1_IBM_s016
                                 + 1.07737 w_k1_INTC_s016
                                 + 0.806258 w_k1_KO_s016
                                 + 0.631571 w_k1_MAT_s016
                                 + 1.03444 w_k1_MSFT_s016
                                 + 0.948306 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.11885 w_k1_ADM_s016
                                 + 0.992554 w_k1_AXP_s016
                                 + 0.845442 w_k1_DIS_s016
                                 + 1.00527 w_k1_FDX_s016
                                 + 0.887729 w_k1_IBM_s016
                                 + 0.943415 w_k1_INTC_s016
                                 + 0.96274 w_k1_KO_s016 + 1.19327 w_k1_MAT_s016
                                 + 0.957974 w_k1_MSFT_s016
                                 + 0.897861 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.999685 w_k1_ADM_s017
                                 + 1.11056 w_k1_AXP_s017
                                 + 1.75053 w_k1_DIS_s017
                                 + 1.25696 w_k1_FDX_s017
                                 + 1.49983 w_k1_IBM_s017
                                 + 1.07004 w_k1_INTC_s017
                                 + 1.31733 w_k1_KO_s017 + 1.19956 w_k1_MAT_s017
                                 + 1.25616 w_k1_MSFT_s017
                                 + 0.982237 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.928868 w_k1_ADM_s017
                                 + 0.949752 w_k1_AXP_s017
                                 + 1.14541 w_k1_DIS_s017
                                 + 1.00654 w_k1_FDX_s017
                                 + 1.08534 w_k1_IBM_s017
                                 + 0.951158 w_k1_INTC_s017
                                 + 1.14807 w_k1_KO_s017
                                 + 0.878886 w_k1_MAT_s017
                                 + 1.04922 w_k1_MSFT_s017
                                 + 0.92606 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.80943 w_k1_ADM_s017
                                 + 0.776777 w_k1_AXP_s017
                                 + 1.06453 w_k1_DIS_s017
                                 + 0.989946 w_k1_FDX_s017
                                 + 0.805031 w_k1_IBM_s017
                                 + 1.01303 w_k1_INTC_s017
                                 + 0.871434 w_k1_KO_s017
                                 + 0.808608 w_k1_MAT_s017
                                 + 1.02077 w_k1_MSFT_s017
                                 + 0.926434 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.06972 w_k1_ADM_s017
                                 + 1.12647 w_k1_AXP_s017 + 1.3137 w_k1_DIS_s017
                                 + 1.36591 w_k1_FDX_s017
                                 + 0.723162 w_k1_IBM_s017
                                 + 1.19908 w_k1_INTC_s017
                                 + 0.861467 w_k1_KO_s017
                                 + 0.639506 w_k1_MAT_s017
                                 + 1.39115 w_k1_MSFT_s017
                                 + 0.894708 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.75449 w_k1_ADM_s017
                                 + 0.718481 w_k1_AXP_s017
                                 + 0.65581 w_k1_DIS_s017
                                 + 0.765399 w_k1_FDX_s017
                                 + 0.960097 w_k1_IBM_s017
                                 + 0.775718 w_k1_INTC_s017
                                 + 0.721274 w_k1_KO_s017
                                 + 0.925078 w_k1_MAT_s017
                                 + 0.79029 w_k1_MSFT_s017
                                 + 0.847405 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.04175 w_k1_ADM_s017
                                 + 1.32835 w_k1_AXP_s017
                                 + 1.42205 w_k1_DIS_s017
                                 + 1.25536 w_k1_FDX_s017
                                 + 1.20677 w_k1_IBM_s017
                                 + 1.15579 w_k1_INTC_s017
                                 + 1.18572 w_k1_KO_s017 + 1.60344 w_k1_MAT_s017
                                 + 1.17261 w_k1_MSFT_s017
                                 + 1.08832 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.07559 w_k1_ADM_s017
                                 + 1.09427 w_k1_AXP_s017
                                 + 0.778625 w_k1_DIS_s017
                                 + 1.01079 w_k1_FDX_s017
                                 + 0.827133 w_k1_IBM_s017
                                 + 1.07464 w_k1_INTC_s017
                                 + 0.798085 w_k1_KO_s017
                                 + 0.679998 w_k1_MAT_s017
                                 + 0.850393 w_k1_MSFT_s017
                                 + 0.872939 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.791399 w_k1_ADM_s017
                                 + 0.843183 w_k1_AXP_s017
                                 + 1.06058 w_k1_DIS_s017
                                 + 0.90397 w_k1_FDX_s017
                                 + 1.42199 w_k1_IBM_s017
                                 + 0.800871 w_k1_INTC_s017
                                 + 1.02173 w_k1_KO_s017 + 1.29981 w_k1_MAT_s017
                                 + 0.851491 w_k1_MSFT_s017
                                 + 0.949569 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.710313 w_k1_ADM_s100
                                 + 0.829641 w_k1_AXP_s100
                                 + 0.633313 w_k1_DIS_s100
                                 + 0.736517 w_k1_FDX_s100
                                 + 0.719275 w_k1_IBM_s100
                                 + 0.688885 w_k1_INTC_s100
                                 + 0.640116 w_k1_KO_s100
                                 + 0.658942 w_k1_MAT_s100
                                 + 0.760795 w_k1_MSFT_s100
                                 + 0.829181 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.852696 w_k1_ADM_s100
                                 + 0.72525 w_k1_AXP_s100
                                 + 0.88977 w_k1_DIS_s100
                                 + 0.857809 w_k1_FDX_s100
                                 + 0.696907 w_k1_IBM_s100
                                 + 0.72019 w_k1_INTC_s100
                                 + 0.870974 w_k1_KO_s100
                                 + 0.71648 w_k1_MAT_s100
                                 + 0.903426 w_k1_MSFT_s100
                                 + 0.917026 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.45518 w_k1_ADM_s100
                                 + 1.64304 w_k1_AXP_s100
                                 + 1.97848 w_k1_DIS_s100 + 1.4778 w_k1_FDX_s100
                                 + 1.30311 w_k1_IBM_s100
                                 + 1.41226 w_k1_INTC_s100
                                 + 1.45117 w_k1_KO_s100 + 1.21737 w_k1_MAT_s100
                                 + 1.67915 w_k1_MSFT_s100
                                 + 1.26988 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.04539 w_k1_ADM_s100
                                 + 1.26553 w_k1_AXP_s100
                                 + 1.03228 w_k1_DIS_s100
                                 + 0.858224 w_k1_FDX_s100
                                 + 1.53787 w_k1_IBM_s100
                                 + 0.869843 w_k1_INTC_s100
                                 + 1.21276 w_k1_KO_s100 + 1.71119 w_k1_MAT_s100
                                 + 0.960384 w_k1_MSFT_s100
                                 + 1.01426 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.676673 w_k1_ADM_s100
                                 + 0.68226 w_k1_AXP_s100
                                 + 0.623866 w_k1_DIS_s100
                                 + 0.711862 w_k1_FDX_s100
                                 + 1.11783 w_k1_IBM_s100
                                 + 0.84643 w_k1_INTC_s100
                                 + 0.840879 w_k1_KO_s100
                                 + 1.10936 w_k1_MAT_s100
                                 + 0.833552 w_k1_MSFT_s100
                                 + 0.827892 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.886965 w_k1_ADM_s100
                                 + 0.848969 w_k1_AXP_s100
                                 + 1.12936 w_k1_DIS_s100
                                 + 0.898308 w_k1_FDX_s100
                                 + 1.11678 w_k1_IBM_s100
                                 + 1.05098 w_k1_INTC_s100
                                 + 1.01132 w_k1_KO_s100
                                 + 0.828808 w_k1_MAT_s100
                                 + 1.17385 w_k1_MSFT_s100
                                 + 0.966609 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.942071 w_k1_ADM_s100
                                 + 1.24516 w_k1_AXP_s100
                                 + 1.30488 w_k1_DIS_s100
                                 + 1.11373 w_k1_FDX_s100
                                 + 1.25206 w_k1_IBM_s100
                                 + 1.26702 w_k1_INTC_s100
                                 + 1.17254 w_k1_KO_s100 + 1.24644 w_k1_MAT_s100
                                 + 1.32423 w_k1_MSFT_s100
                                 + 1.16537 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.852853 w_k1_ADM_s100
                                 + 1.06524 w_k1_AXP_s100
                                 + 1.01559 w_k1_DIS_s100
                                 + 1.07209 w_k1_FDX_s100
                                 + 0.898545 w_k1_IBM_s100
                                 + 1.00992 w_k1_INTC_s100
                                 + 0.917892 w_k1_KO_s100
                                 + 0.994849 w_k1_MAT_s100
                                 + 1.05912 w_k1_MSFT_s100
                                 + 1.05816 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.867154 w_k1_ADM_s101
                                 + 1.14696 w_k1_AXP_s101
                                 + 0.877244 w_k1_DIS_s101
                                 + 0.976818 w_k1_FDX_s101
                                 + 0.888748 w_k1_IBM_s101
                                 + 0.872804 w_k1_INTC_s101
                                 + 0.770614 w_k1_KO_s101
                                 + 0.824899 w_k1_MAT_s101
                                 + 0.84987 w_k1_MSFT_s101
                                 + 0.909252 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.744309 w_k1_ADM_s101
                                 + 0.760384 w_k1_AXP_s101
                                 + 0.621798 w_k1_DIS_s101
                                 + 0.705498 w_k1_FDX_s101
                                 + 0.795922 w_k1_IBM_s101
                                 + 0.763928 w_k1_INTC_s101
                                 + 0.728126 w_k1_KO_s101
                                 + 0.613888 w_k1_MAT_s101
                                 + 0.70306 w_k1_MSFT_s101
                                 + 0.751934 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.965621 w_k1_ADM_s101
                                 + 1.55302 w_k1_AXP_s101
                                 + 1.28932 w_k1_DIS_s101
                                 + 1.17114 w_k1_FDX_s101
                                 + 1.28091 w_k1_IBM_s101
                                 + 0.956033 w_k1_INTC_s101
                                 + 1.10757 w_k1_KO_s101 + 1.52592 w_k1_MAT_s101
                                 + 1.0219 w_k1_MSFT_s101 + 1.10372 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.725475 w_k1_ADM_s101
                                 + 0.859377 w_k1_AXP_s101
                                 + 0.810905 w_k1_DIS_s101
                                 + 0.81763 w_k1_FDX_s101
                                 + 1.01706 w_k1_IBM_s101
                                 + 0.92132 w_k1_INTC_s101
                                 + 0.823707 w_k1_KO_s101
                                 + 0.786602 w_k1_MAT_s101
                                 + 0.832765 w_k1_MSFT_s101
                                 + 0.88717 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.03587 w_k1_ADM_s101
                                 + 1.16617 w_k1_AXP_s101
                                 + 0.87695 w_k1_DIS_s101
                                 + 0.963223 w_k1_FDX_s101
                                 + 1.35361 w_k1_IBM_s101
                                 + 1.02359 w_k1_INTC_s101
                                 + 1.00427 w_k1_KO_s101 + 1.20255 w_k1_MAT_s101
                                 + 0.88029 w_k1_MSFT_s101
                                 + 0.961104 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.08034 w_k1_ADM_s101
                                 + 0.981454 w_k1_AXP_s101 + 1.278 w_k1_DIS_s101
                                 + 1.13109 w_k1_FDX_s101
                                 + 0.978656 w_k1_IBM_s101
                                 + 1.02967 w_k1_INTC_s101
                                 + 1.06415 w_k1_KO_s101
                                 + 0.570423 w_k1_MAT_s101
                                 + 1.18306 w_k1_MSFT_s101
                                 + 1.13888 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.26379 w_k1_ADM_s101
                                 + 1.54678 w_k1_AXP_s101
                                 + 1.66624 w_k1_DIS_s101
                                 + 1.48977 w_k1_FDX_s101
                                 + 1.20289 w_k1_IBM_s101
                                 + 1.22328 w_k1_INTC_s101
                                 + 1.45406 w_k1_KO_s101 + 1.08346 w_k1_MAT_s101
                                 + 1.30332 w_k1_MSFT_s101
                                 + 1.47357 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.998358 w_k1_ADM_s101
                                 + 1.00787 w_k1_AXP_s101
                                 + 1.06788 w_k1_DIS_s101
                                 + 1.01446 w_k1_FDX_s101
                                 + 0.656842 w_k1_IBM_s101
                                 + 0.928285 w_k1_INTC_s101
                                 + 0.825038 w_k1_KO_s101
                                 + 0.42021 w_k1_MAT_s101
                                 + 1.0262 w_k1_MSFT_s101
                                 + 0.923722 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.22545 w_k1_ADM_s102
                                 + 1.24593 w_k1_AXP_s102
                                 + 1.46282 w_k1_DIS_s102
                                 + 1.16572 w_k1_FDX_s102
                                 + 1.08178 w_k1_IBM_s102
                                 + 1.25363 w_k1_INTC_s102
                                 + 1.27469 w_k1_KO_s102 + 1.34709 w_k1_MAT_s102
                                 + 1.26377 w_k1_MSFT_s102
                                 + 1.05964 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.28011 w_k1_ADM_s102
                                 + 1.32361 w_k1_AXP_s102
                                 + 1.01123 w_k1_DIS_s102
                                 + 1.10095 w_k1_FDX_s102
                                 + 1.16045 w_k1_IBM_s102
                                 + 1.02711 w_k1_INTC_s102
                                 + 1.05381 w_k1_KO_s102 + 1.53526 w_k1_MAT_s102
                                 + 1.00768 w_k1_MSFT_s102 + 1.049 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.01594 w_k1_ADM_s102
                                 + 0.80576 w_k1_AXP_s102
                                 + 0.770833 w_k1_DIS_s102
                                 + 0.843088 w_k1_FDX_s102
                                 + 0.983811 w_k1_IBM_s102
                                 + 0.86144 w_k1_INTC_s102
                                 + 0.954893 w_k1_KO_s102
                                 + 0.908473 w_k1_MAT_s102
                                 + 0.87011 w_k1_MSFT_s102
                                 + 0.938051 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.15904 w_k1_ADM_s102
                                 + 0.82886 w_k1_AXP_s102
                                 + 0.876817 w_k1_DIS_s102
                                 + 0.871415 w_k1_FDX_s102
                                 + 1.43612 w_k1_IBM_s102
                                 + 0.996908 w_k1_INTC_s102
                                 + 1.05115 w_k1_KO_s102 + 1.38754 w_k1_MAT_s102
                                 + 0.946205 w_k1_MSFT_s102
                                 + 0.992826 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.39487 w_k1_ADM_s102
                                 + 1.64498 w_k1_AXP_s102
                                 + 1.49196 w_k1_DIS_s102 + 1.3155 w_k1_FDX_s102
                                 + 1.81523 w_k1_IBM_s102
                                 + 1.2981 w_k1_INTC_s102 + 1.43505 w_k1_KO_s102
                                 + 2.45115 w_k1_MAT_s102
                                 + 1.35769 w_k1_MSFT_s102
                                 + 1.18172 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.26626 w_k1_ADM_s102
                                 + 1.29195 w_k1_AXP_s102
                                 + 0.969946 w_k1_DIS_s102
                                 + 1.01602 w_k1_FDX_s102
                                 + 1.57818 w_k1_IBM_s102
                                 + 1.05871 w_k1_INTC_s102
                                 + 1.24617 w_k1_KO_s102 + 2.22921 w_k1_MAT_s102
                                 + 1.03662 w_k1_MSFT_s102
                                 + 1.09967 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.880735 w_k1_ADM_s102
                                 + 0.723702 w_k1_AXP_s102
                                 + 0.84552 w_k1_DIS_s102
                                 + 0.925967 w_k1_FDX_s102
                                 + 0.690217 w_k1_IBM_s102
                                 + 0.878935 w_k1_INTC_s102
                                 + 0.721641 w_k1_KO_s102
                                 + 0.649081 w_k1_MAT_s102
                                 + 0.91125 w_k1_MSFT_s102
                                 + 0.863789 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.10489 w_k1_ADM_s102
                                 + 1.36337 w_k1_AXP_s102
                                 + 1.14839 w_k1_DIS_s102
                                 + 1.30911 w_k1_FDX_s102
                                 + 0.800279 w_k1_IBM_s102
                                 + 1.13403 w_k1_INTC_s102
                                 + 0.880332 w_k1_KO_s102
                                 + 1.17894 w_k1_MAT_s102
                                 + 1.29862 w_k1_MSFT_s102 + 1.1558 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.09448 w_k1_ADM_s103
                                 + 1.12246 w_k1_AXP_s103
                                 + 1.24039 w_k1_DIS_s103 + 1.0941 w_k1_FDX_s103
                                 + 1.48203 w_k1_IBM_s103
                                 + 1.0278 w_k1_INTC_s103 + 1.3761 w_k1_KO_s103
                                 + 0.939438 w_k1_MAT_s103
                                 + 0.929849 w_k1_MSFT_s103
                                 + 1.27424 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.992508 w_k1_ADM_s103
                                 + 1.00936 w_k1_AXP_s103
                                 + 1.04553 w_k1_DIS_s103
                                 + 1.10451 w_k1_FDX_s103
                                 + 1.13105 w_k1_IBM_s103
                                 + 1.22782 w_k1_INTC_s103
                                 + 1.12637 w_k1_KO_s103
                                 + 0.918918 w_k1_MAT_s103
                                 + 1.05884 w_k1_MSFT_s103
                                 + 1.12825 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.017 w_k1_ADM_s103
                                 + 0.948344 w_k1_AXP_s103
                                 + 0.921199 w_k1_DIS_s103
                                 + 1.02042 w_k1_FDX_s103
                                 + 0.841036 w_k1_IBM_s103
                                 + 0.851224 w_k1_INTC_s103
                                 + 0.858646 w_k1_KO_s103
                                 + 0.63602 w_k1_MAT_s103
                                 + 0.825977 w_k1_MSFT_s103
                                 + 0.937508 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.15631 w_k1_ADM_s103
                                 + 1.0467 w_k1_AXP_s103
                                 + 0.719674 w_k1_DIS_s103
                                 + 0.964937 w_k1_FDX_s103
                                 + 0.94235 w_k1_IBM_s103
                                 + 0.790437 w_k1_INTC_s103
                                 + 0.769799 w_k1_KO_s103
                                 + 0.945783 w_k1_MAT_s103
                                 + 0.738889 w_k1_MSFT_s103
                                 + 1.06241 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.87017 w_k1_ADM_s103
                                 + 0.691868 w_k1_AXP_s103
                                 + 0.626146 w_k1_DIS_s103
                                 + 0.751622 w_k1_FDX_s103
                                 + 0.880243 w_k1_IBM_s103
                                 + 0.683796 w_k1_INTC_s103
                                 + 0.746111 w_k1_KO_s103
                                 + 0.662014 w_k1_MAT_s103
                                 + 0.695192 w_k1_MSFT_s103
                                 + 0.923457 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.06654 w_k1_ADM_s103
                                 + 1.85273 w_k1_AXP_s103
                                 + 1.41005 w_k1_DIS_s103
                                 + 1.35156 w_k1_FDX_s103
                                 + 1.00813 w_k1_IBM_s103
                                 + 1.22162 w_k1_INTC_s103
                                 + 1.09625 w_k1_KO_s103 + 1.0504 w_k1_MAT_s103
                                 + 1.05766 w_k1_MSFT_s103
                                 + 1.17371 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.947964 w_k1_ADM_s103
                                 + 0.964362 w_k1_AXP_s103
                                 + 1.31134 w_k1_DIS_s103
                                 + 1.16142 w_k1_FDX_s103
                                 + 0.860787 w_k1_IBM_s103
                                 + 0.999225 w_k1_INTC_s103
                                 + 1.10427 w_k1_KO_s103
                                 + 0.659779 w_k1_MAT_s103
                                 + 0.920106 w_k1_MSFT_s103
                                 + 1.17485 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.15749 w_k1_ADM_s103
                                 + 0.971561 w_k1_AXP_s103
                                 + 1.11415 w_k1_DIS_s103
                                 + 1.02267 w_k1_FDX_s103
                                 + 0.852236 w_k1_IBM_s103
                                 + 0.965679 w_k1_INTC_s103
                                 + 0.90999 w_k1_KO_s103
                                 + 0.662156 w_k1_MAT_s103
                                 + 0.93185 w_k1_MSFT_s103
                                 + 0.959129 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.34522 w_k1_ADM_s104
                                 + 1.00881 w_k1_AXP_s104
                                 + 0.985157 w_k1_DIS_s104
                                 + 1.13433 w_k1_FDX_s104
                                 + 0.89557 w_k1_IBM_s104
                                 + 1.0764 w_k1_INTC_s104 + 1.01417 w_k1_KO_s104
                                 + 0.795827 w_k1_MAT_s104
                                 + 1.11105 w_k1_MSFT_s104 + 1.0372 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.67647 w_k1_ADM_s104
                                 + 1.10874 w_k1_AXP_s104
                                 + 1.71692 w_k1_DIS_s104
                                 + 1.20505 w_k1_FDX_s104 + 1.0328 w_k1_IBM_s104
                                 + 1.35769 w_k1_INTC_s104
                                 + 1.58354 w_k1_KO_s104
                                 + 0.756925 w_k1_MAT_s104
                                 + 1.51962 w_k1_MSFT_s104
                                 + 1.22188 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.923343 w_k1_ADM_s104
                                 + 0.852846 w_k1_AXP_s104
                                 + 0.780439 w_k1_DIS_s104
                                 + 0.821589 w_k1_FDX_s104
                                 + 1.05944 w_k1_IBM_s104
                                 + 1.06085 w_k1_INTC_s104
                                 + 1.01045 w_k1_KO_s104 + 1.05091 w_k1_MAT_s104
                                 + 0.978001 w_k1_MSFT_s104
                                 + 0.885851 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.2402 w_k1_ADM_s104
                                 + 0.909902 w_k1_AXP_s104
                                 + 1.22297 w_k1_DIS_s104
                                 + 0.962739 w_k1_FDX_s104
                                 + 1.21969 w_k1_IBM_s104
                                 + 1.04552 w_k1_INTC_s104
                                 + 1.30102 w_k1_KO_s104
                                 + 0.781933 w_k1_MAT_s104
                                 + 1.00356 w_k1_MSFT_s104
                                 + 0.979154 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.13078 w_k1_ADM_s104
                                 + 1.23505 w_k1_AXP_s104
                                 + 1.28305 w_k1_DIS_s104
                                 + 1.13343 w_k1_FDX_s104 + 1.1204 w_k1_IBM_s104
                                 + 1.25358 w_k1_INTC_s104
                                 + 1.20643 w_k1_KO_s104 + 1.05289 w_k1_MAT_s104
                                 + 1.27684 w_k1_MSFT_s104
                                 + 1.17963 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.2229 w_k1_ADM_s104
                                 + 0.682345 w_k1_AXP_s104
                                 + 0.85461 w_k1_DIS_s104
                                 + 0.918555 w_k1_FDX_s104
                                 + 0.678608 w_k1_IBM_s104
                                 + 0.869595 w_k1_INTC_s104
                                 + 0.870382 w_k1_KO_s104
                                 + 0.538364 w_k1_MAT_s104
                                 + 0.962166 w_k1_MSFT_s104
                                 + 0.808057 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.27303 w_k1_ADM_s104
                                 + 1.6864 w_k1_AXP_s104 + 1.75 w_k1_DIS_s104
                                 + 1.40622 w_k1_FDX_s104
                                 + 1.53795 w_k1_IBM_s104
                                 + 1.33099 w_k1_INTC_s104
                                 + 1.44393 w_k1_KO_s104 + 1.77966 w_k1_MAT_s104
                                 + 1.37995 w_k1_MSFT_s104
                                 + 1.34753 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.61747 w_k1_ADM_s104
                                 + 1.06937 w_k1_AXP_s104
                                 + 1.14356 w_k1_DIS_s104
                                 + 0.997324 w_k1_FDX_s104
                                 + 1.73274 w_k1_IBM_s104
                                 + 0.885286 w_k1_INTC_s104
                                 + 1.56708 w_k1_KO_s104 + 1.67767 w_k1_MAT_s104
                                 + 1.23378 w_k1_MSFT_s104
                                 + 1.26112 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.967495 w_k1_ADM_s105
                                 + 0.789122 w_k1_AXP_s105
                                 + 0.836956 w_k1_DIS_s105
                                 + 0.837102 w_k1_FDX_s105
                                 + 0.989487 w_k1_IBM_s105
                                 + 0.73078 w_k1_INTC_s105
                                 + 1.00427 w_k1_KO_s105
                                 + 0.885965 w_k1_MAT_s105
                                 + 0.876973 w_k1_MSFT_s105
                                 + 0.885958 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.14711 w_k1_ADM_s105
                                 + 1.22228 w_k1_AXP_s105
                                 + 1.15329 w_k1_DIS_s105 + 1.1343 w_k1_FDX_s105
                                 + 1.27038 w_k1_IBM_s105
                                 + 1.11486 w_k1_INTC_s105
                                 + 1.03766 w_k1_KO_s105 + 1.21215 w_k1_MAT_s105
                                 + 1.14279 w_k1_MSFT_s105 + 1.0964 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.900931 w_k1_ADM_s105
                                 + 1.03197 w_k1_AXP_s105
                                 + 0.806309 w_k1_DIS_s105
                                 + 0.895799 w_k1_FDX_s105
                                 + 1.16618 w_k1_IBM_s105
                                 + 0.901766 w_k1_INTC_s105
                                 + 0.99248 w_k1_KO_s105 + 1.36111 w_k1_MAT_s105
                                 + 0.863646 w_k1_MSFT_s105
                                 + 0.921952 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.732463 w_k1_ADM_s105
                                 + 0.736719 w_k1_AXP_s105
                                 + 0.610907 w_k1_DIS_s105
                                 + 0.661092 w_k1_FDX_s105
                                 + 0.854944 w_k1_IBM_s105
                                 + 0.77041 w_k1_INTC_s105
                                 + 0.758855 w_k1_KO_s105
                                 + 0.66994 w_k1_MAT_s105
                                 + 0.727154 w_k1_MSFT_s105
                                 + 0.801726 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.978008 w_k1_ADM_s105
                                 + 1.07132 w_k1_AXP_s105
                                 + 1.31014 w_k1_DIS_s105
                                 + 1.16866 w_k1_FDX_s105
                                 + 0.901318 w_k1_IBM_s105
                                 + 1.1206 w_k1_INTC_s105 + 1.11345 w_k1_KO_s105
                                 + 0.815301 w_k1_MAT_s105
                                 + 1.15769 w_k1_MSFT_s105
                                 + 1.04003 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.750216 w_k1_ADM_s105
                                 + 0.731052 w_k1_AXP_s105
                                 + 0.802781 w_k1_DIS_s105
                                 + 0.895787 w_k1_FDX_s105
                                 + 0.63816 w_k1_IBM_s105
                                 + 0.82533 w_k1_INTC_s105
                                 + 0.754678 w_k1_KO_s105
                                 + 0.586999 w_k1_MAT_s105
                                 + 0.933414 w_k1_MSFT_s105
                                 + 0.8463 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.837264 w_k1_ADM_s105
                                 + 0.834523 w_k1_AXP_s105
                                 + 1.15815 w_k1_DIS_s105
                                 + 1.05697 w_k1_FDX_s105
                                 + 0.753218 w_k1_IBM_s105
                                 + 0.910382 w_k1_INTC_s105
                                 + 1.14298 w_k1_KO_s105 + 1.16552 w_k1_MAT_s105
                                 + 1.30662 w_k1_MSFT_s105
                                 + 0.994641 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.768606 w_k1_ADM_s105
                                 + 1.58641 w_k1_AXP_s105
                                 + 1.29384 w_k1_DIS_s105
                                 + 1.15176 w_k1_FDX_s105
                                 + 1.42487 w_k1_IBM_s105
                                 + 1.07258 w_k1_INTC_s105
                                 + 1.28038 w_k1_KO_s105 + 2.04183 w_k1_MAT_s105
                                 + 1.006 w_k1_MSFT_s105 + 1.14567 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.55382 w_k1_ADM_s106
                                 + 0.650902 w_k1_AXP_s106
                                 + 0.601549 w_k1_DIS_s106
                                 + 0.64953 w_k1_FDX_s106
                                 + 1.02835 w_k1_IBM_s106
                                 + 0.916209 w_k1_INTC_s106
                                 + 0.743423 w_k1_KO_s106
                                 + 0.86461 w_k1_MAT_s106
                                 + 0.812371 w_k1_MSFT_s106
                                 + 0.959271 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.828615 w_k1_ADM_s106
                                 + 1.14667 w_k1_AXP_s106
                                 + 1.07309 w_k1_DIS_s106
                                 + 1.03301 w_k1_FDX_s106
                                 + 0.987778 w_k1_IBM_s106
                                 + 1.21295 w_k1_INTC_s106
                                 + 0.932193 w_k1_KO_s106 + 1.199 w_k1_MAT_s106
                                 + 0.977664 w_k1_MSFT_s106
                                 + 1.03461 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.858268 w_k1_ADM_s106
                                 + 0.752757 w_k1_AXP_s106
                                 + 0.957217 w_k1_DIS_s106
                                 + 0.94371 w_k1_FDX_s106
                                 + 0.848565 w_k1_IBM_s106
                                 + 1.06753 w_k1_INTC_s106
                                 + 0.954168 w_k1_KO_s106
                                 + 0.759068 w_k1_MAT_s106
                                 + 0.91402 w_k1_MSFT_s106
                                 + 1.09336 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.44075 w_k1_ADM_s106
                                 + 1.49835 w_k1_AXP_s106
                                 + 1.63972 w_k1_DIS_s106
                                 + 1.36751 w_k1_FDX_s106
                                 + 0.86461 w_k1_IBM_s106
                                 + 1.28089 w_k1_INTC_s106
                                 + 1.12307 w_k1_KO_s106 + 1.08311 w_k1_MAT_s106
                                 + 1.21446 w_k1_MSFT_s106
                                 + 1.13661 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.645301 w_k1_ADM_s106
                                 + 0.783238 w_k1_AXP_s106
                                 + 0.546989 w_k1_DIS_s106
                                 + 0.780509 w_k1_FDX_s106
                                 + 0.623512 w_k1_IBM_s106
                                 + 1.19028 w_k1_INTC_s106
                                 + 0.564684 w_k1_KO_s106
                                 + 0.550239 w_k1_MAT_s106
                                 + 0.764497 w_k1_MSFT_s106
                                 + 0.79818 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.660662 w_k1_ADM_s106
                                 + 0.697845 w_k1_AXP_s106
                                 + 0.547039 w_k1_DIS_s106
                                 + 0.696478 w_k1_FDX_s106
                                 + 0.841485 w_k1_IBM_s106
                                 + 0.866475 w_k1_INTC_s106
                                 + 0.638641 w_k1_KO_s106
                                 + 0.624224 w_k1_MAT_s106
                                 + 0.648457 w_k1_MSFT_s106
                                 + 0.850682 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.728748 w_k1_ADM_s106
                                 + 1.07307 w_k1_AXP_s106
                                 + 0.905958 w_k1_DIS_s106
                                 + 0.919986 w_k1_FDX_s106
                                 + 1.51468 w_k1_IBM_s106
                                 + 1.1154 w_k1_INTC_s106 + 1.05139 w_k1_KO_s106
                                 + 1.72609 w_k1_MAT_s106
                                 + 0.903721 w_k1_MSFT_s106
                                 + 1.05275 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.25158 w_k1_ADM_s106
                                 + 1.51082 w_k1_AXP_s106
                                 + 1.75784 w_k1_DIS_s106
                                 + 1.39129 w_k1_FDX_s106
                                 + 1.45623 w_k1_IBM_s106
                                 + 1.97239 w_k1_INTC_s106
                                 + 1.51259 w_k1_KO_s106 + 1.76627 w_k1_MAT_s106
                                 + 1.40537 w_k1_MSFT_s106
                                 + 1.46274 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.776868 w_k1_ADM_s107
                                 + 0.89022 w_k1_AXP_s107
                                 + 0.772736 w_k1_DIS_s107
                                 + 0.853282 w_k1_FDX_s107
                                 + 0.926157 w_k1_IBM_s107
                                 + 0.973179 w_k1_INTC_s107
                                 + 0.785475 w_k1_KO_s107
                                 + 0.770847 w_k1_MAT_s107
                                 + 0.833406 w_k1_MSFT_s107
                                 + 0.849572 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.995995 w_k1_ADM_s107
                                 + 0.929326 w_k1_AXP_s107
                                 + 1.14287 w_k1_DIS_s107
                                 + 1.11638 w_k1_FDX_s107
                                 + 0.801277 w_k1_IBM_s107
                                 + 1.14229 w_k1_INTC_s107
                                 + 0.865095 w_k1_KO_s107
                                 + 0.671993 w_k1_MAT_s107
                                 + 1.13867 w_k1_MSFT_s107
                                 + 1.03653 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.831705 w_k1_ADM_s107
                                 + 0.741513 w_k1_AXP_s107
                                 + 1.23128 w_k1_DIS_s107 + 1.0371 w_k1_FDX_s107
                                 + 1.22596 w_k1_IBM_s107
                                 + 1.15277 w_k1_INTC_s107
                                 + 0.974153 w_k1_KO_s107
                                 + 0.684751 w_k1_MAT_s107
                                 + 1.04504 w_k1_MSFT_s107
                                 + 0.988905 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.810078 w_k1_ADM_s107
                                 + 0.563778 w_k1_AXP_s107
                                 + 0.562968 w_k1_DIS_s107
                                 + 0.815194 w_k1_FDX_s107
                                 + 0.672242 w_k1_IBM_s107
                                 + 0.770643 w_k1_INTC_s107
                                 + 0.596629 w_k1_KO_s107
                                 + 0.462739 w_k1_MAT_s107
                                 + 0.787814 w_k1_MSFT_s107
                                 + 0.780052 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.970996 w_k1_ADM_s107
                                 + 0.675798 w_k1_AXP_s107
                                 + 0.901304 w_k1_DIS_s107
                                 + 0.962908 w_k1_FDX_s107
                                 + 0.950767 w_k1_IBM_s107
                                 + 0.896656 w_k1_INTC_s107
                                 + 0.862046 w_k1_KO_s107
                                 + 0.823508 w_k1_MAT_s107
                                 + 0.965524 w_k1_MSFT_s107
                                 + 0.878633 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.983993 w_k1_ADM_s107
                                 + 1.4562 w_k1_AXP_s107 + 1.55507 w_k1_DIS_s107
                                 + 1.37284 w_k1_FDX_s107
                                 + 1.31128 w_k1_IBM_s107
                                 + 1.51015 w_k1_INTC_s107
                                 + 1.30891 w_k1_KO_s107 + 1.62892 w_k1_MAT_s107
                                 + 1.37547 w_k1_MSFT_s107
                                 + 1.16193 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.747176 w_k1_ADM_s107
                                 + 0.662833 w_k1_AXP_s107
                                 + 0.65872 w_k1_DIS_s107
                                 + 0.775941 w_k1_FDX_s107
                                 + 1.19791 w_k1_IBM_s107
                                 + 0.846096 w_k1_INTC_s107
                                 + 0.754459 w_k1_KO_s107
                                 + 1.16411 w_k1_MAT_s107
                                 + 0.743755 w_k1_MSFT_s107
                                 + 0.874909 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.876306 w_k1_ADM_s107
                                 + 1.21537 w_k1_AXP_s107
                                 + 1.18763 w_k1_DIS_s107
                                 + 1.07033 w_k1_FDX_s107
                                 + 1.38704 w_k1_IBM_s107 + 1.14 w_k1_INTC_s107
                                 + 0.959106 w_k1_KO_s107
                                 + 1.21448 w_k1_MAT_s107
                                 + 0.994015 w_k1_MSFT_s107
                                 + 1.03211 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.14382 w_k1_ADM_s110
                                 + 0.962694 w_k1_AXP_s110
                                 + 0.878909 w_k1_DIS_s110
                                 + 0.821449 w_k1_FDX_s110
                                 + 1.0443 w_k1_IBM_s110
                                 + 0.631012 w_k1_INTC_s110
                                 + 0.889066 w_k1_KO_s110
                                 + 0.896573 w_k1_MAT_s110
                                 + 0.771103 w_k1_MSFT_s110
                                 + 0.926497 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.978284 w_k1_ADM_s110
                                 + 0.805698 w_k1_AXP_s110
                                 + 0.818563 w_k1_DIS_s110
                                 + 0.850012 w_k1_FDX_s110
                                 + 0.71525 w_k1_IBM_s110
                                 + 0.741609 w_k1_INTC_s110
                                 + 0.816832 w_k1_KO_s110
                                 + 0.687905 w_k1_MAT_s110
                                 + 0.858231 w_k1_MSFT_s110
                                 + 0.843782 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.50426 w_k1_ADM_s110
                                 + 1.58544 w_k1_AXP_s110
                                 + 1.60366 w_k1_DIS_s110
                                 + 1.39521 w_k1_FDX_s110
                                 + 1.20956 w_k1_IBM_s110
                                 + 1.07299 w_k1_INTC_s110
                                 + 1.39147 w_k1_KO_s110 + 1.32431 w_k1_MAT_s110
                                 + 1.40271 w_k1_MSFT_s110
                                 + 1.26764 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.26693 w_k1_ADM_s110
                                 + 1.10548 w_k1_AXP_s110
                                 + 1.18326 w_k1_DIS_s110
                                 + 1.02551 w_k1_FDX_s110
                                 + 1.03597 w_k1_IBM_s110
                                 + 0.89454 w_k1_INTC_s110 + 1.2716 w_k1_KO_s110
                                 + 0.991095 w_k1_MAT_s110
                                 + 1.25052 w_k1_MSFT_s110
                                 + 1.12141 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.10918 w_k1_ADM_s110
                                 + 1.04076 w_k1_AXP_s110
                                 + 0.916222 w_k1_DIS_s110
                                 + 0.868579 w_k1_FDX_s110
                                 + 1.33346 w_k1_IBM_s110
                                 + 0.789583 w_k1_INTC_s110
                                 + 1.14606 w_k1_KO_s110 + 1.03982 w_k1_MAT_s110
                                 + 0.961228 w_k1_MSFT_s110
                                 + 1.01499 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.96137 w_k1_ADM_s110
                                 + 0.689099 w_k1_AXP_s110
                                 + 0.88123 w_k1_DIS_s110
                                 + 0.833559 w_k1_FDX_s110
                                 + 0.939328 w_k1_IBM_s110
                                 + 0.855044 w_k1_INTC_s110
                                 + 1.09119 w_k1_KO_s110
                                 + 0.800151 w_k1_MAT_s110
                                 + 1.02926 w_k1_MSFT_s110
                                 + 0.938855 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.33048 w_k1_ADM_s110
                                 + 0.985655 w_k1_AXP_s110
                                 + 1.18753 w_k1_DIS_s110
                                 + 1.15392 w_k1_FDX_s110
                                 + 0.770176 w_k1_IBM_s110
                                 + 0.814641 w_k1_INTC_s110
                                 + 1.00062 w_k1_KO_s110
                                 + 0.768063 w_k1_MAT_s110
                                 + 1.09005 w_k1_MSFT_s110 + 1.0463 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.16202 w_k1_ADM_s110
                                 + 1.21774 w_k1_AXP_s110
                                 + 1.35656 w_k1_DIS_s110
                                 + 1.05745 w_k1_FDX_s110
                                 + 1.76543 w_k1_IBM_s110
                                 + 0.962853 w_k1_INTC_s110
                                 + 1.58463 w_k1_KO_s110 + 1.61497 w_k1_MAT_s110
                                 + 1.25878 w_k1_MSFT_s110
                                 + 1.24456 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.09255 w_k1_ADM_s111
                                 + 1.17364 w_k1_AXP_s111
                                 + 1.11675 w_k1_DIS_s111
                                 + 1.08517 w_k1_FDX_s111
                                 + 1.52165 w_k1_IBM_s111
                                 + 1.15225 w_k1_INTC_s111 + 1.1218 w_k1_KO_s111
                                 + 1.18642 w_k1_MAT_s111
                                 + 1.14953 w_k1_MSFT_s111
                                 + 1.20606 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.05508 w_k1_ADM_s111
                                 + 0.931609 w_k1_AXP_s111
                                 + 1.10632 w_k1_DIS_s111
                                 + 1.09289 w_k1_FDX_s111
                                 + 0.79676 w_k1_IBM_s111
                                 + 0.928653 w_k1_INTC_s111
                                 + 0.760746 w_k1_KO_s111
                                 + 0.873712 w_k1_MAT_s111
                                 + 0.98913 w_k1_MSFT_s111
                                 + 1.00572 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.825899 w_k1_ADM_s111
                                 + 0.717647 w_k1_AXP_s111
                                 + 0.527086 w_k1_DIS_s111
                                 + 0.74842 w_k1_FDX_s111
                                 + 0.829122 w_k1_IBM_s111
                                 + 0.760429 w_k1_INTC_s111
                                 + 0.666683 w_k1_KO_s111
                                 + 0.823959 w_k1_MAT_s111
                                 + 0.833075 w_k1_MSFT_s111
                                 + 0.842837 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.79045 w_k1_ADM_s111
                                 + 0.973086 w_k1_AXP_s111
                                 + 0.804239 w_k1_DIS_s111
                                 + 0.893824 w_k1_FDX_s111
                                 + 1.12272 w_k1_IBM_s111
                                 + 0.844681 w_k1_INTC_s111
                                 + 0.89287 w_k1_KO_s111 + 1.33547 w_k1_MAT_s111
                                 + 0.878031 w_k1_MSFT_s111
                                 + 0.978506 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.25891 w_k1_ADM_s111
                                 + 0.971964 w_k1_AXP_s111
                                 + 1.48578 w_k1_DIS_s111
                                 + 1.20708 w_k1_FDX_s111
                                 + 1.32776 w_k1_IBM_s111
                                 + 1.13346 w_k1_INTC_s111
                                 + 1.20131 w_k1_KO_s111
                                 + 0.908568 w_k1_MAT_s111
                                 + 1.0396 w_k1_MSFT_s111 + 1.2312 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.91989 w_k1_ADM_s111
                                 + 0.670906 w_k1_AXP_s111
                                 + 0.721897 w_k1_DIS_s111
                                 + 0.84097 w_k1_FDX_s111
                                 + 0.743467 w_k1_IBM_s111
                                 + 0.891878 w_k1_INTC_s111
                                 + 0.824357 w_k1_KO_s111
                                 + 0.784162 w_k1_MAT_s111
                                 + 0.981261 w_k1_MSFT_s111
                                 + 0.972639 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.07533 w_k1_ADM_s111
                                 + 1.11474 w_k1_AXP_s111
                                 + 1.09041 w_k1_DIS_s111
                                 + 1.13043 w_k1_FDX_s111 + 0.9901 w_k1_IBM_s111
                                 + 0.934707 w_k1_INTC_s111
                                 + 0.935002 w_k1_KO_s111
                                 + 0.877065 w_k1_MAT_s111
                                 + 1.13919 w_k1_MSFT_s111
                                 + 1.22431 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.21531 w_k1_ADM_s111
                                 + 1.62175 w_k1_AXP_s111
                                 + 1.41481 w_k1_DIS_s111
                                 + 1.38514 w_k1_FDX_s111
                                 + 1.16959 w_k1_IBM_s111
                                 + 1.30009 w_k1_INTC_s111
                                 + 1.34493 w_k1_KO_s111 + 1.66557 w_k1_MAT_s111
                                 + 1.24429 w_k1_MSFT_s111 + 1.223 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.864727 w_k1_ADM_s112
                                 + 0.760234 w_k1_AXP_s112
                                 + 0.815318 w_k1_DIS_s112
                                 + 0.79818 w_k1_FDX_s112
                                 + 0.721478 w_k1_IBM_s112
                                 + 0.970113 w_k1_INTC_s112
                                 + 0.850353 w_k1_KO_s112
                                 + 0.636657 w_k1_MAT_s112
                                 + 0.794875 w_k1_MSFT_s112
                                 + 0.922862 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.29795 w_k1_ADM_s112
                                 + 1.14339 w_k1_AXP_s112
                                 + 1.52593 w_k1_DIS_s112
                                 + 1.27111 w_k1_FDX_s112
                                 + 0.731761 w_k1_IBM_s112
                                 + 1.2414 w_k1_INTC_s112 + 1.22076 w_k1_KO_s112
                                 + 0.889121 w_k1_MAT_s112
                                 + 1.38482 w_k1_MSFT_s112
                                 + 1.25915 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.993662 w_k1_ADM_s112
                                 + 0.986157 w_k1_AXP_s112
                                 + 1.14814 w_k1_DIS_s112
                                 + 1.03349 w_k1_FDX_s112
                                 + 1.24631 w_k1_IBM_s112
                                 + 1.01949 w_k1_INTC_s112
                                 + 1.25737 w_k1_KO_s112 + 1.40582 w_k1_MAT_s112
                                 + 1.03784 w_k1_MSFT_s112
                                 + 1.07001 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.960896 w_k1_ADM_s112
                                 + 1.14749 w_k1_AXP_s112
                                 + 1.16278 w_k1_DIS_s112
                                 + 0.881468 w_k1_FDX_s112
                                 + 1.70436 w_k1_IBM_s112
                                 + 0.871077 w_k1_INTC_s112
                                 + 1.31758 w_k1_KO_s112 + 1.91034 w_k1_MAT_s112
                                 + 1.0702 w_k1_MSFT_s112 + 1.23322 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.05854 w_k1_ADM_s112
                                 + 0.86828 w_k1_AXP_s112
                                 + 0.786567 w_k1_DIS_s112
                                 + 0.864638 w_k1_FDX_s112
                                 + 0.934214 w_k1_IBM_s112
                                 + 1.04336 w_k1_INTC_s112
                                 + 0.987519 w_k1_KO_s112
                                 + 0.802574 w_k1_MAT_s112
                                 + 0.82828 w_k1_MSFT_s112
                                 + 0.990176 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.24563 w_k1_ADM_s112
                                 + 1.21195 w_k1_AXP_s112 + 1.3391 w_k1_DIS_s112
                                 + 1.17176 w_k1_FDX_s112 + 1.1065 w_k1_IBM_s112
                                 + 1.29542 w_k1_INTC_s112 + 1.237 w_k1_KO_s112
                                 + 1.07719 w_k1_MAT_s112
                                 + 1.29741 w_k1_MSFT_s112
                                 + 1.18572 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.70615 w_k1_ADM_s112
                                 + 1.50332 w_k1_AXP_s112
                                 + 2.02508 w_k1_DIS_s112
                                 + 1.41991 w_k1_FDX_s112
                                 + 1.62093 w_k1_IBM_s112
                                 + 1.39874 w_k1_INTC_s112
                                 + 1.50153 w_k1_KO_s112 + 1.43262 w_k1_MAT_s112
                                 + 1.3452 w_k1_MSFT_s112 + 1.49543 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.17184 w_k1_ADM_s112
                                 + 1.13278 w_k1_AXP_s112
                                 + 0.987263 w_k1_DIS_s112
                                 + 1.04651 w_k1_FDX_s112
                                 + 0.948984 w_k1_IBM_s112
                                 + 1.26834 w_k1_INTC_s112
                                 + 0.975842 w_k1_KO_s112
                                 + 0.757391 w_k1_MAT_s112
                                 + 1.01319 w_k1_MSFT_s112 + 1.0674 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.28461 w_k1_ADM_s113
                                 + 1.35634 w_k1_AXP_s113
                                 + 1.53777 w_k1_DIS_s113
                                 + 1.21313 w_k1_FDX_s113 + 1.1562 w_k1_IBM_s113
                                 + 1.13254 w_k1_INTC_s113
                                 + 1.13042 w_k1_KO_s113
                                 + 0.951874 w_k1_MAT_s113
                                 + 1.28084 w_k1_MSFT_s113
                                 + 1.17748 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.19389 w_k1_ADM_s113
                                 + 1.48816 w_k1_AXP_s113
                                 + 1.18485 w_k1_DIS_s113
                                 + 1.08676 w_k1_FDX_s113
                                 + 1.42961 w_k1_IBM_s113
                                 + 0.916629 w_k1_INTC_s113
                                 + 1.20955 w_k1_KO_s113 + 1.78568 w_k1_MAT_s113
                                 + 1.04605 w_k1_MSFT_s113
                                 + 1.20684 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.91114 w_k1_ADM_s113
                                 + 0.684958 w_k1_AXP_s113
                                 + 0.807545 w_k1_DIS_s113
                                 + 0.83167 w_k1_FDX_s113
                                 + 0.969905 w_k1_IBM_s113
                                 + 0.843736 w_k1_INTC_s113
                                 + 0.828032 w_k1_KO_s113
                                 + 0.82442 w_k1_MAT_s113
                                 + 0.921129 w_k1_MSFT_s113
                                 + 0.901755 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.970181 w_k1_ADM_s113
                                 + 0.778675 w_k1_AXP_s113
                                 + 0.727913 w_k1_DIS_s113
                                 + 0.851506 w_k1_FDX_s113
                                 + 0.671347 w_k1_IBM_s113
                                 + 0.733313 w_k1_INTC_s113
                                 + 0.598105 w_k1_KO_s113
                                 + 0.53558 w_k1_MAT_s113
                                 + 0.808195 w_k1_MSFT_s113
                                 + 0.862548 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.00063 w_k1_ADM_s113
                                 + 0.894988 w_k1_AXP_s113
                                 + 1.06518 w_k1_DIS_s113
                                 + 1.00322 w_k1_FDX_s113
                                 + 0.742703 w_k1_IBM_s113
                                 + 1.03567 w_k1_INTC_s113
                                 + 0.92374 w_k1_KO_s113
                                 + 0.788597 w_k1_MAT_s113
                                 + 1.16483 w_k1_MSFT_s113
                                 + 0.97286 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.11143 w_k1_ADM_s113
                                 + 0.961454 w_k1_AXP_s113
                                 + 0.814328 w_k1_DIS_s113
                                 + 0.924761 w_k1_FDX_s113
                                 + 0.902767 w_k1_IBM_s113
                                 + 0.809555 w_k1_INTC_s113
                                 + 0.76627 w_k1_KO_s113
                                 + 0.834612 w_k1_MAT_s113
                                 + 0.808295 w_k1_MSFT_s113
                                 + 0.969807 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.16889 w_k1_ADM_s113
                                 + 1.42086 w_k1_AXP_s113
                                 + 1.16167 w_k1_DIS_s113
                                 + 1.09825 w_k1_FDX_s113
                                 + 0.937634 w_k1_IBM_s113
                                 + 0.913816 w_k1_INTC_s113
                                 + 0.87651 w_k1_KO_s113 + 1.11925 w_k1_MAT_s113
                                 + 0.969818 w_k1_MSFT_s113
                                 + 1.04101 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.994702 w_k1_ADM_s113
                                 + 0.888041 w_k1_AXP_s113
                                 + 1.10423 w_k1_DIS_s113
                                 + 0.922906 w_k1_FDX_s113
                                 + 1.35152 w_k1_IBM_s113
                                 + 0.845342 w_k1_INTC_s113
                                 + 1.03232 w_k1_KO_s113 + 0.74577 w_k1_MAT_s113
                                 + 1.00832 w_k1_MSFT_s113
                                 + 1.06515 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.15752 w_k1_ADM_s114
                                 + 0.999611 w_k1_AXP_s114
                                 + 0.873678 w_k1_DIS_s114
                                 + 0.933464 w_k1_FDX_s114
                                 + 0.982405 w_k1_IBM_s114
                                 + 0.822777 w_k1_INTC_s114
                                 + 0.820615 w_k1_KO_s114
                                 + 0.864651 w_k1_MAT_s114
                                 + 0.906611 w_k1_MSFT_s114
                                 + 0.985723 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.962028 w_k1_ADM_s114
                                 + 0.674114 w_k1_AXP_s114
                                 + 0.613633 w_k1_DIS_s114
                                 + 0.737223 w_k1_FDX_s114
                                 + 0.730447 w_k1_IBM_s114
                                 + 0.846354 w_k1_INTC_s114
                                 + 0.711618 w_k1_KO_s114
                                 + 0.430699 w_k1_MAT_s114
                                 + 0.806221 w_k1_MSFT_s114
                                 + 0.812487 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.03004 w_k1_ADM_s114
                                 + 0.927614 w_k1_AXP_s114
                                 + 1.16607 w_k1_DIS_s114
                                 + 0.879929 w_k1_FDX_s114
                                 + 1.22047 w_k1_IBM_s114
                                 + 1.03557 w_k1_INTC_s114
                                 + 1.03643 w_k1_KO_s114
                                 + 0.777103 w_k1_MAT_s114
                                 + 0.970572 w_k1_MSFT_s114
                                 + 0.964377 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.49495 w_k1_ADM_s114
                                 + 1.23206 w_k1_AXP_s114
                                 + 1.54844 w_k1_DIS_s114
                                 + 1.32863 w_k1_FDX_s114
                                 + 1.16557 w_k1_IBM_s114
                                 + 1.30494 w_k1_INTC_s114
                                 + 1.28236 w_k1_KO_s114 + 0.94309 w_k1_MAT_s114
                                 + 1.44915 w_k1_MSFT_s114
                                 + 1.24749 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.2864 w_k1_ADM_s114
                                 + 1.02253 w_k1_AXP_s114
                                 + 1.21294 w_k1_DIS_s114
                                 + 1.17457 w_k1_FDX_s114
                                 + 0.799524 w_k1_IBM_s114
                                 + 1.15468 w_k1_INTC_s114
                                 + 1.04689 w_k1_KO_s114
                                 + 0.902695 w_k1_MAT_s114
                                 + 1.28715 w_k1_MSFT_s114
                                 + 1.17572 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.04381 w_k1_ADM_s114
                                 + 0.802592 w_k1_AXP_s114
                                 + 1.00966 w_k1_DIS_s114
                                 + 0.89733 w_k1_FDX_s114
                                 + 0.734597 w_k1_IBM_s114
                                 + 1.00981 w_k1_INTC_s114
                                 + 0.892816 w_k1_KO_s114
                                 + 0.70634 w_k1_MAT_s114
                                 + 1.0481 w_k1_MSFT_s114 + 1.02567 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.855983 w_k1_ADM_s114
                                 + 0.697642 w_k1_AXP_s114
                                 + 0.680963 w_k1_DIS_s114
                                 + 0.693379 w_k1_FDX_s114
                                 + 1.11874 w_k1_IBM_s114
                                 + 0.894199 w_k1_INTC_s114
                                 + 0.909373 w_k1_KO_s114
                                 + 0.778509 w_k1_MAT_s114
                                 + 0.868405 w_k1_MSFT_s114
                                 + 0.855557 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.34577 w_k1_ADM_s114
                                 + 1.35822 w_k1_AXP_s114
                                 + 1.45261 w_k1_DIS_s114
                                 + 1.06852 w_k1_FDX_s114
                                 + 1.60201 w_k1_IBM_s114
                                 + 1.11962 w_k1_INTC_s114
                                 + 1.43721 w_k1_KO_s114 + 1.48437 w_k1_MAT_s114
                                 + 1.14477 w_k1_MSFT_s114
                                 + 1.17814 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.12248 w_k1_ADM_s115
                                 + 0.858685 w_k1_AXP_s115
                                 + 1.40339 w_k1_DIS_s115
                                 + 1.32996 w_k1_FDX_s115
                                 + 0.675823 w_k1_IBM_s115
                                 + 1.00634 w_k1_INTC_s115
                                 + 1.00129 w_k1_KO_s115 + 0.67684 w_k1_MAT_s115
                                 + 1.17699 w_k1_MSFT_s115
                                 + 1.04241 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.912943 w_k1_ADM_s115
                                 + 1.10821 w_k1_AXP_s115
                                 + 1.28862 w_k1_DIS_s115
                                 + 1.01062 w_k1_FDX_s115
                                 + 1.44398 w_k1_IBM_s115
                                 + 0.951961 w_k1_INTC_s115
                                 + 1.21729 w_k1_KO_s115 + 1.46857 w_k1_MAT_s115
                                 + 1.00965 w_k1_MSFT_s115
                                 + 1.12389 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.33122 w_k1_ADM_s115
                                 + 1.39997 w_k1_AXP_s115
                                 + 1.47176 w_k1_DIS_s115
                                 + 1.21782 w_k1_FDX_s115
                                 + 1.05318 w_k1_IBM_s115
                                 + 1.1283 w_k1_INTC_s115 + 1.07765 w_k1_KO_s115
                                 + 0.974096 w_k1_MAT_s115
                                 + 1.06121 w_k1_MSFT_s115
                                 + 1.16291 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.05819 w_k1_ADM_s115
                                 + 0.871136 w_k1_AXP_s115
                                 + 0.836526 w_k1_DIS_s115
                                 + 0.907793 w_k1_FDX_s115
                                 + 0.882326 w_k1_IBM_s115
                                 + 0.871856 w_k1_INTC_s115
                                 + 0.837684 w_k1_KO_s115
                                 + 0.717798 w_k1_MAT_s115
                                 + 0.867025 w_k1_MSFT_s115
                                 + 0.90391 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.49655 w_k1_ADM_s115
                                 + 1.0853 w_k1_AXP_s115 + 1.17157 w_k1_DIS_s115
                                 + 1.13235 w_k1_FDX_s115
                                 + 0.785988 w_k1_IBM_s115
                                 + 0.76503 w_k1_INTC_s115
                                 + 0.870635 w_k1_KO_s115
                                 + 0.697313 w_k1_MAT_s115
                                 + 0.910607 w_k1_MSFT_s115
                                 + 1.00471 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.837079 w_k1_ADM_s115
                                 + 0.877688 w_k1_AXP_s115
                                 + 0.722318 w_k1_DIS_s115
                                 + 0.746467 w_k1_FDX_s115
                                 + 1.56258 w_k1_IBM_s115
                                 + 0.596024 w_k1_INTC_s115
                                 + 1.07476 w_k1_KO_s115 + 1.3901 w_k1_MAT_s115
                                 + 0.704642 w_k1_MSFT_s115
                                 + 0.899414 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.967072 w_k1_ADM_s115
                                 + 1.05297 w_k1_AXP_s115
                                 + 0.955322 w_k1_DIS_s115
                                 + 0.934355 w_k1_FDX_s115
                                 + 1.03309 w_k1_IBM_s115
                                 + 1.03017 w_k1_INTC_s115
                                 + 0.98797 w_k1_KO_s115
                                 + 0.957016 w_k1_MAT_s115
                                 + 0.888006 w_k1_MSFT_s115
                                 + 0.947662 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.01831 w_k1_ADM_s115
                                 + 0.781493 w_k1_AXP_s115
                                 + 0.771519 w_k1_DIS_s115
                                 + 0.909497 w_k1_FDX_s115
                                 + 0.689385 w_k1_IBM_s115
                                 + 0.815572 w_k1_INTC_s115
                                 + 0.743833 w_k1_KO_s115
                                 + 0.692322 w_k1_MAT_s115
                                 + 0.840096 w_k1_MSFT_s115
                                 + 0.852725 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.04016 w_k1_ADM_s116
                                 + 1.54491 w_k1_AXP_s116
                                 + 1.30889 w_k1_DIS_s116 + 1.2225 w_k1_FDX_s116
                                 + 1.4099 w_k1_IBM_s116
                                 + 1.21918 w_k1_INTC_s116
                                 + 1.18629 w_k1_KO_s116 + 2.07029 w_k1_MAT_s116
                                 + 1.1067 w_k1_MSFT_s116 + 1.22891 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.7782 w_k1_ADM_s116
                                 + 0.684455 w_k1_AXP_s116
                                 + 0.677688 w_k1_DIS_s116
                                 + 0.745689 w_k1_FDX_s116
                                 + 0.822266 w_k1_IBM_s116
                                 + 0.86787 w_k1_INTC_s116
                                 + 0.801729 w_k1_KO_s116
                                 + 0.613592 w_k1_MAT_s116
                                 + 0.697562 w_k1_MSFT_s116
                                 + 0.842561 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.796013 w_k1_ADM_s116
                                 + 0.86896 w_k1_AXP_s116
                                 + 1.01876 w_k1_DIS_s116
                                 + 1.04511 w_k1_FDX_s116
                                 + 0.745849 w_k1_IBM_s116
                                 + 1.08649 w_k1_INTC_s116
                                 + 0.90407 w_k1_KO_s116
                                 + 0.799215 w_k1_MAT_s116
                                 + 1.03577 w_k1_MSFT_s116
                                 + 0.978992 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.21417 w_k1_ADM_s116
                                 + 1.28249 w_k1_AXP_s116
                                 + 1.19632 w_k1_DIS_s116
                                 + 1.25789 w_k1_FDX_s116
                                 + 0.867846 w_k1_IBM_s116
                                 + 1.2949 w_k1_INTC_s116 + 0.79349 w_k1_KO_s116
                                 + 1.01528 w_k1_MAT_s116
                                 + 1.09044 w_k1_MSFT_s116
                                 + 1.04646 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.864112 w_k1_ADM_s116
                                 + 0.914066 w_k1_AXP_s116
                                 + 0.947186 w_k1_DIS_s116
                                 + 0.883456 w_k1_FDX_s116
                                 + 0.791858 w_k1_IBM_s116
                                 + 0.825657 w_k1_INTC_s116
                                 + 0.817258 w_k1_KO_s116 + 1.2 w_k1_MAT_s116
                                 + 0.798079 w_k1_MSFT_s116
                                 + 0.988033 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.877926 w_k1_ADM_s116
                                 + 1.06843 w_k1_AXP_s116
                                 + 1.19549 w_k1_DIS_s116
                                 + 1.01128 w_k1_FDX_s116
                                 + 1.16882 w_k1_IBM_s116
                                 + 0.919716 w_k1_INTC_s116
                                 + 1.18284 w_k1_KO_s116 + 1.58286 w_k1_MAT_s116
                                 + 1.05688 w_k1_MSFT_s116
                                 + 1.06162 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.06417 w_k1_ADM_s116
                                 + 1.19202 w_k1_AXP_s116
                                 + 1.10932 w_k1_DIS_s116
                                 + 0.972988 w_k1_FDX_s116
                                 + 1.56827 w_k1_IBM_s116
                                 + 0.770701 w_k1_INTC_s116
                                 + 1.25725 w_k1_KO_s116 + 2.31055 w_k1_MAT_s116
                                 + 0.797449 w_k1_MSFT_s116
                                 + 1.09672 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.03933 w_k1_ADM_s116
                                 + 1.12789 w_k1_AXP_s116
                                 + 1.62048 w_k1_DIS_s116
                                 + 1.32204 w_k1_FDX_s116
                                 + 0.944664 w_k1_IBM_s116
                                 + 1.04742 w_k1_INTC_s116
                                 + 1.04939 w_k1_KO_s116 + 1.05643 w_k1_MAT_s116
                                 + 1.31111 w_k1_MSFT_s116
                                 + 1.31703 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.918136 w_k1_ADM_s117
                                 + 1.61239 w_k1_AXP_s117
                                 + 1.16596 w_k1_DIS_s117
                                 + 1.12517 w_k1_FDX_s117
                                 + 0.951393 w_k1_IBM_s117
                                 + 1.13283 w_k1_INTC_s117
                                 + 0.928214 w_k1_KO_s117
                                 + 1.16586 w_k1_MAT_s117
                                 + 1.05759 w_k1_MSFT_s117
                                 + 1.02324 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.1112 w_k1_ADM_s117
                                 + 0.917136 w_k1_AXP_s117
                                 + 1.60963 w_k1_DIS_s117
                                 + 1.02814 w_k1_FDX_s117
                                 + 1.06392 w_k1_IBM_s117
                                 + 1.15138 w_k1_INTC_s117 + 1.3493 w_k1_KO_s117
                                 + 0.958069 w_k1_MAT_s117
                                 + 1.24426 w_k1_MSFT_s117
                                 + 0.943648 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.977017 w_k1_ADM_s117
                                 + 0.982942 w_k1_AXP_s117
                                 + 1.1748 w_k1_DIS_s117 + 1.0764 w_k1_FDX_s117
                                 + 0.981702 w_k1_IBM_s117
                                 + 0.969839 w_k1_INTC_s117
                                 + 1.0449 w_k1_KO_s117 + 1.08315 w_k1_MAT_s117
                                 + 1.04507 w_k1_MSFT_s117 + 1.0055 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.961157 w_k1_ADM_s117
                                 + 0.765962 w_k1_AXP_s117
                                 + 1.04121 w_k1_DIS_s117
                                 + 0.872619 w_k1_FDX_s117
                                 + 1.22958 w_k1_IBM_s117
                                 + 1.08493 w_k1_INTC_s117 + 1.0275 w_k1_KO_s117
                                 + 0.943284 w_k1_MAT_s117
                                 + 0.904795 w_k1_MSFT_s117
                                 + 0.787917 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.662216 w_k1_ADM_s117
                                 + 0.607777 w_k1_AXP_s117
                                 + 0.755849 w_k1_DIS_s117
                                 + 0.794541 w_k1_FDX_s117
                                 + 0.728263 w_k1_IBM_s117
                                 + 0.82943 w_k1_INTC_s117
                                 + 0.800771 w_k1_KO_s117
                                 + 0.598398 w_k1_MAT_s117
                                 + 0.838449 w_k1_MSFT_s117
                                 + 0.783392 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.965569 w_k1_ADM_s117
                                 + 1.00264 w_k1_AXP_s117
                                 + 1.22499 w_k1_DIS_s117
                                 + 1.08977 w_k1_FDX_s117
                                 + 0.960649 w_k1_IBM_s117
                                 + 1.22134 w_k1_INTC_s117
                                 + 1.00114 w_k1_KO_s117
                                 + 0.740719 w_k1_MAT_s117
                                 + 0.943821 w_k1_MSFT_s117
                                 + 0.893813 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.731359 w_k1_ADM_s117
                                 + 0.991241 w_k1_AXP_s117
                                 + 0.638162 w_k1_DIS_s117
                                 + 0.743662 w_k1_FDX_s117
                                 + 1.20285 w_k1_IBM_s117
                                 + 0.920015 w_k1_INTC_s117
                                 + 1.05495 w_k1_KO_s117 + 1.06221 w_k1_MAT_s117
                                 + 0.79862 w_k1_MSFT_s117
                                 + 0.836009 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.27822 w_k1_ADM_s117
                                 + 1.3713 w_k1_AXP_s117 + 1.41743 w_k1_DIS_s117
                                 + 1.13174 w_k1_FDX_s117
                                 + 1.54676 w_k1_IBM_s117
                                 + 1.14659 w_k1_INTC_s117 + 1.2815 w_k1_KO_s117
                                 + 1.4835 w_k1_MAT_s117
                                 + 1.05492 w_k1_MSFT_s117
                                 + 1.22063 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.03662539103795 w_k1_ADM_s000
                                 + 0.0226352622410473 w_k1_ADM_s001
                                 + 0.0719810125096818 w_k1_ADM_s002
                                 + 0.0320387656173994 w_k1_ADM_s003
                                 + 0.0268774482687405 w_k1_ADM_s004
                                 + 0.0146693172290404 w_k1_ADM_s005
                                 + 0.0207825877772025 w_k1_ADM_s006
                                 + 0.0177386241868032 w_k1_ADM_s007
                                 + 0.0397948748075889 w_k1_ADM_s010
                                 + 0.0600573891354511 w_k1_ADM_s011
                                 + 0.0175069903191022 w_k1_ADM_s012
                                 + 0.0363638979900249 w_k1_ADM_s013
                                 + 0.0374007614457985 w_k1_ADM_s014
                                 + 0.0311899456158119 w_k1_ADM_s015
                                 + 0.0190388077947737 w_k1_ADM_s016
                                 + 0.0210540423442692 w_k1_ADM_s017
                                 + 0.0412194504321567 w_k1_ADM_s100
                                 + 0.0208303938517449 w_k1_ADM_s101
                                 + 0.0471050913658945 w_k1_ADM_s102
                                 + 0.0193111836860613 w_k1_ADM_s103
                                 + 0.0275318096943387 w_k1_ADM_s104
                                 + 0.0366017702022655 w_k1_ADM_s105
                                 + 0.0125826053397525 w_k1_ADM_s106
                                 + 0.0281599458097655 w_k1_ADM_s107
                                 + 0.0455623573041245 w_k1_ADM_s110
                                 + 0.0249953295294698 w_k1_ADM_s111
                                 + 0.0289437271311134 w_k1_ADM_s112
                                 + 0.0680163196274268 w_k1_ADM_s113
                                 + 0.0590091054228501 w_k1_ADM_s114
                                 + 0.02390082006734 w_k1_ADM_s115
                                 + 0.0235095384303315 w_k1_ADM_s116
                                 + 0.0211096798399352 w_k1_ADM_s117
                                 + 0.0366909703117399 w_k1_AXP_s000
                                 + 0.0296630827914843 w_k1_AXP_s001
                                 + 0.0621883269517115 w_k1_AXP_s002
                                 + 0.0329780786814151 w_k1_AXP_s003
                                 + 0.0318712208560312 w_k1_AXP_s004
                                 + 0.0147344321447693 w_k1_AXP_s005
                                 + 0.0228998496440556 w_k1_AXP_s006
                                 + 0.015300642061189 w_k1_AXP_s007
                                 + 0.0384691546969399 w_k1_AXP_s010
                                 + 0.0520703690152473 w_k1_AXP_s011
                                 + 0.0188923430859405 w_k1_AXP_s012
                                 + 0.0321541202199368 w_k1_AXP_s013
                                 + 0.0334327134700107 w_k1_AXP_s014
                                 + 0.0286667183025222 w_k1_AXP_s015
                                 + 0.0176853346236314 w_k1_AXP_s016
                                 + 0.0219878238763815 w_k1_AXP_s017
                                 + 0.0465250999165816 w_k1_AXP_s100
                                 + 0.0252683044169523 w_k1_AXP_s101
                                 + 0.045365241295734 w_k1_AXP_s102
                                 + 0.0200621514325852 w_k1_AXP_s103
                                 + 0.0217677834062756 w_k1_AXP_s104
                                 + 0.0401577704544159 w_k1_AXP_s105
                                 + 0.0152925803879987 w_k1_AXP_s106
                                 + 0.0307619884561179 w_k1_AXP_s107
                                 + 0.0401379455775893 w_k1_AXP_s110
                                 + 0.0253354838426499 w_k1_AXP_s111
                                 + 0.0269100679148814 w_k1_AXP_s112
                                 + 0.0643466658356131 w_k1_AXP_s113
                                 + 0.0501908443024382 w_k1_AXP_s114
                                 + 0.0226997147412375 w_k1_AXP_s115
                                 + 0.0257029559652829 w_k1_AXP_s116
                                 + 0.0227345763381338 w_k1_AXP_s117
                                 + 0.0366068908018577 w_k1_DIS_s000
                                 + 0.0279495329686326 w_k1_DIS_s001
                                 + 0.0629170276403441 w_k1_DIS_s002
                                 + 0.0347740794764274 w_k1_DIS_s003
                                 + 0.0320989560502076 w_k1_DIS_s004
                                 + 0.0172345615049165 w_k1_DIS_s005
                                 + 0.0234138939220756 w_k1_DIS_s006
                                 + 0.0177386175626195 w_k1_DIS_s007
                                 + 0.0399167721449772 w_k1_DIS_s010
                                 + 0.05718597949744 w_k1_DIS_s011
                                 + 0.0190271282656656 w_k1_DIS_s012
                                 + 0.0304047758572592 w_k1_DIS_s013
                                 + 0.0358015886070585 w_k1_DIS_s014
                                 + 0.0285794845630241 w_k1_DIS_s015
                                 + 0.0178640801440145 w_k1_DIS_s016
                                 + 0.0247953471993623 w_k1_DIS_s017
                                 + 0.0484010804832107 w_k1_DIS_s100
                                 + 0.0227472883294981 w_k1_DIS_s101
                                 + 0.0437247261002244 w_k1_DIS_s102
                                 + 0.0193889626572365 w_k1_DIS_s103
                                 + 0.0239352505473286 w_k1_DIS_s104
                                 + 0.0415126893093259 w_k1_DIS_s105
                                 + 0.0148550833162318 w_k1_DIS_s106
                                 + 0.0334985821446262 w_k1_DIS_s107
                                 + 0.0422016278911011 w_k1_DIS_s110
                                 + 0.024704578863593 w_k1_DIS_s111
                                 + 0.0293601670556294 w_k1_DIS_s112
                                 + 0.067015264594042 w_k1_DIS_s113
                                 + 0.0550520052287705 w_k1_DIS_s114
                                 + 0.0238968973332968 w_k1_DIS_s115
                                 + 0.0278237023285709 w_k1_DIS_s116
                                 + 0.0250535832105149 w_k1_DIS_s117
                                 + 0.0357350100838326 w_k1_FDX_s000
                                 + 0.0274769414047002 w_k1_FDX_s001
                                 + 0.0628023886270224 w_k1_FDX_s002
                                 + 0.0335104130960083 w_k1_FDX_s003
                                 + 0.0311697050497395 w_k1_FDX_s004
                                 + 0.0158044323042237 w_k1_FDX_s005
                                 + 0.0215062300802552 w_k1_FDX_s006
                                 + 0.0164821418636115 w_k1_FDX_s007
                                 + 0.0403369734715164 w_k1_FDX_s010
                                 + 0.0578879878699018 w_k1_FDX_s011
                                 + 0.0177538982109328 w_k1_FDX_s012
                                 + 0.0305012746511429 w_k1_FDX_s013
                                 + 0.03436265116037 w_k1_FDX_s014
                                 + 0.0279644338708065 w_k1_FDX_s015
                                 + 0.0170653755780033 w_k1_FDX_s016
                                 + 0.0236938055628349 w_k1_FDX_s017
                                 + 0.0449331722744741 w_k1_FDX_s100
                                 + 0.0224821478431294 w_k1_FDX_s101
                                 + 0.0436617893920591 w_k1_FDX_s102
                                 + 0.019773493522838 w_k1_FDX_s103
                                 + 0.02269649795601 w_k1_FDX_s104
                                 + 0.0399400130182308 w_k1_FDX_s105
                                 + 0.0143858883039733 w_k1_FDX_s106
                                 + 0.032822489650361 w_k1_FDX_s107
                                 + 0.0386707554520517 w_k1_FDX_s110
                                 + 0.0257785844011346 w_k1_FDX_s111
                                 + 0.0263264386956451 w_k1_FDX_s112
                                 + 0.0622361162286954 w_k1_FDX_s113
                                 + 0.0495465990744388 w_k1_FDX_s114
                                 + 0.0226239665889488 w_k1_FDX_s115
                                 + 0.0259932910809606 w_k1_FDX_s116
                                 + 0.0223746527562449 w_k1_FDX_s117
                                 + 0.0352492560052291 w_k1_IBM_s000
                                 + 0.0277670442626148 w_k1_IBM_s001
                                 + 0.0651050455549206 w_k1_IBM_s002
                                 + 0.0374016525046797 w_k1_IBM_s003
                                 + 0.0304949344236931 w_k1_IBM_s004
                                 + 0.0170587418946093 w_k1_IBM_s005
                                 + 0.0221876122967078 w_k1_IBM_s006
                                 + 0.0163494485032176 w_k1_IBM_s007
                                 + 0.0381342954447551 w_k1_IBM_s010
                                 + 0.0558996274909404 w_k1_IBM_s011
                                 + 0.0186632961984721 w_k1_IBM_s012
                                 + 0.0327442240230194 w_k1_IBM_s013
                                 + 0.0347023900311557 w_k1_IBM_s014
                                 + 0.0297109912216567 w_k1_IBM_s015
                                 + 0.0177841062850185 w_k1_IBM_s016
                                 + 0.0235418917860132 w_k1_IBM_s017
                                 + 0.0464379675484887 w_k1_IBM_s100
                                 + 0.0231902724340537 w_k1_IBM_s101
                                 + 0.0444376281143742 w_k1_IBM_s102
                                 + 0.0193521092267558 w_k1_IBM_s103
                                 + 0.0236199483344654 w_k1_IBM_s104
                                 + 0.0398989976112345 w_k1_IBM_s105
                                 + 0.015188348226855 w_k1_IBM_s106
                                 + 0.0353076362484636 w_k1_IBM_s107
                                 + 0.0398347964404408 w_k1_IBM_s110
                                 + 0.0254550412949258 w_k1_IBM_s111
                                 + 0.0273962349403 w_k1_IBM_s112
                                 + 0.0641671362849009 w_k1_IBM_s113
                                 + 0.0533779062324434 w_k1_IBM_s114
                                 + 0.0230241267068018 w_k1_IBM_s115
                                 + 0.0250058012246924 w_k1_IBM_s116
                                 + 0.0235064388901169 w_k1_IBM_s117
                                 + 0.0410061522499071 w_k1_INTC_s000
                                 + 0.0320335982505575 w_k1_INTC_s001
                                 + 0.0584818984506326 w_k1_INTC_s002
                                 + 0.031222978811171 w_k1_INTC_s003
                                 + 0.0347588972898861 w_k1_INTC_s004
                                 + 0.0166065606988035 w_k1_INTC_s005
                                 + 0.0211009850227686 w_k1_INTC_s006
                                 + 0.0140769613661063 w_k1_INTC_s007
                                 + 0.037153618789894 w_k1_INTC_s010
                                 + 0.05268899026108 w_k1_INTC_s011
                                 + 0.0184969432342542 w_k1_INTC_s012
                                 + 0.0307873189605361 w_k1_INTC_s013
                                 + 0.0288189593426458 w_k1_INTC_s014
                                 + 0.0309101950910315 w_k1_INTC_s015
                                 + 0.0168154169227182 w_k1_INTC_s016
                                 + 0.022860820986208 w_k1_INTC_s017
                                 + 0.0449472265171499 w_k1_INTC_s100
                                 + 0.0212015324416248 w_k1_INTC_s101
                                 + 0.0432824359866476 w_k1_INTC_s102
                                 + 0.0183697986777557 w_k1_INTC_s103
                                 + 0.0238870968192282 w_k1_INTC_s104
                                 + 0.0380696765767332 w_k1_INTC_s105
                                 + 0.0175219217076862 w_k1_INTC_s106
                                 + 0.0349228763844073 w_k1_INTC_s107
                                 + 0.032413109949148 w_k1_INTC_s110
                                 + 0.0240888753066808 w_k1_INTC_s111
                                 + 0.0287147402169154 w_k1_INTC_s112
                                 + 0.0571704635285945 w_k1_INTC_s113
                                 + 0.0512989726307758 w_k1_INTC_s114
                                 + 0.0206843462521837 w_k1_INTC_s115
                                 + 0.0249211721689048 w_k1_INTC_s116
                                 + 0.0233934577456661 w_k1_INTC_s117
                                 + 0.0351052695910058 w_k1_KO_s000
                                 + 0.0269072288319562 w_k1_KO_s001
                                 + 0.0647182289681268 w_k1_KO_s002
                                 + 0.0349072688957445 w_k1_KO_s003
                                 + 0.032626133930058 w_k1_KO_s004
                                 + 0.0174729426055086 w_k1_KO_s005
                                 + 0.0229755895192633 w_k1_KO_s006
                                 + 0.0171529293431207 w_k1_KO_s007
                                 + 0.0389093467680781 w_k1_KO_s010
                                 + 0.0602650896000891 w_k1_KO_s011
                                 + 0.0187558508914224 w_k1_KO_s012
                                 + 0.0339056246348001 w_k1_KO_s013
                                 + 0.0345798121232351 w_k1_KO_s014
                                 + 0.0293156703506036 w_k1_KO_s015
                                 + 0.0177651627215079 w_k1_KO_s016
                                 + 0.0221681002320584 w_k1_KO_s017
                                 + 0.0451928544482588 w_k1_KO_s100
                                 + 0.02103275302913 w_k1_KO_s101
                                 + 0.0429996270093657 w_k1_KO_s102
                                 + 0.0190172551905644 w_k1_KO_s103
                                 + 0.025400844451533 w_k1_KO_s104
                                 + 0.0411046071998522 w_k1_KO_s105
                                 + 0.0138076613460978 w_k1_KO_s106
                                 + 0.029206712767298 w_k1_KO_s107
                                 + 0.0431246699648545 w_k1_KO_s110
                                 + 0.0232923924594474 w_k1_KO_s111
                                 + 0.0290760565546748 w_k1_KO_s112
                                 + 0.0573774516160113 w_k1_KO_s113
                                 + 0.0514883622738219 w_k1_KO_s114
                                 + 0.0219457047245412 w_k1_KO_s115
                                 + 0.0248480562854654 w_k1_KO_s116
                                 + 0.023431111843394 w_k1_KO_s117
                                 + 0.0244025583332596 w_k1_MAT_s000
                                 + 0.0307457958629385 w_k1_MAT_s001
                                 + 0.0686926798218888 w_k1_MAT_s002
                                 + 0.0397949548968858 w_k1_MAT_s003
                                 + 0.0309206155335691 w_k1_MAT_s004
                                 + 0.0165826118921608 w_k1_MAT_s005
                                 + 0.0227059953621266 w_k1_MAT_s006
                                 + 0.015542541757216 w_k1_MAT_s007
                                 + 0.0468368366606448 w_k1_MAT_s010
                                 + 0.0567098095011403 w_k1_MAT_s011
                                 + 0.0207709547808151 w_k1_MAT_s012
                                 + 0.0332028397098978 w_k1_MAT_s013
                                 + 0.0402538542982061 w_k1_MAT_s014
                                 + 0.0254585978208561 w_k1_MAT_s015
                                 + 0.0208620761341086 w_k1_MAT_s016
                                 + 0.0225695800672233 w_k1_MAT_s017
                                 + 0.0463930500893733 w_k1_MAT_s100
                                 + 0.0206251484326418 w_k1_MAT_s101
                                 + 0.0523307511045867 w_k1_MAT_s102
                                 + 0.0153504372504507 w_k1_MAT_s103
                                 + 0.0216249565385661 w_k1_MAT_s104
                                 + 0.0419511416881863 w_k1_MAT_s105
                                 + 0.0164909888154111 w_k1_MAT_s106
                                 + 0.0316921117114457 w_k1_MAT_s107
                                 + 0.037227928723331 w_k1_MAT_s110
                                 + 0.0264552880684054 w_k1_MAT_s111
                                 + 0.0270863387601477 w_k1_MAT_s112
                                 + 0.0543517048593082 w_k1_MAT_s113
                                 + 0.0446213054009315 w_k1_MAT_s114
                                 + 0.0216597276447818 w_k1_MAT_s115
                                 + 0.0304723649358308 w_k1_MAT_s116
                                 + 0.022113410689517 w_k1_MAT_s117
                                 + 0.0326670945147221 w_k1_MSFT_s000
                                 + 0.0277592570868543 w_k1_MSFT_s001
                                 + 0.0687896693429869 w_k1_MSFT_s002
                                 + 0.0341691995558844 w_k1_MSFT_s003
                                 + 0.0295618308777132 w_k1_MSFT_s004
                                 + 0.015805780994231 w_k1_MSFT_s005
                                 + 0.022233496481588 w_k1_MSFT_s006
                                 + 0.0144403145387819 w_k1_MSFT_s007
                                 + 0.0344592806999811 w_k1_MSFT_s010
                                 + 0.0592060093384343 w_k1_MSFT_s011
                                 + 0.018672880232258 w_k1_MSFT_s012
                                 + 0.0312918683702387 w_k1_MSFT_s013
                                 + 0.0328334737230756 w_k1_MSFT_s014
                                 + 0.0278692412573753 w_k1_MSFT_s015
                                 + 0.017791320779028 w_k1_MSFT_s016
                                 + 0.0233895698151115 w_k1_MSFT_s017
                                 + 0.0493795307907028 w_k1_MSFT_s100
                                 + 0.0211573226803384 w_k1_MSFT_s101
                                 + 0.0441801935717136 w_k1_MSFT_s102
                                 + 0.0169000497371901 w_k1_MSFT_s103
                                 + 0.0248731596089719 w_k1_MSFT_s104
                                 + 0.0412478900723044 w_k1_MSFT_s105
                                 + 0.0140688655820902 w_k1_MSFT_s106
                                 + 0.0321821484965393 w_k1_MSFT_s107
                                 + 0.0415068890604892 w_k1_MSFT_s110
                                 + 0.0256655474137665 w_k1_MSFT_s111
                                 + 0.0272244655655742 w_k1_MSFT_s112
                                 + 0.0634371915098121 w_k1_MSFT_s113
                                 + 0.0536532938211542 w_k1_MSFT_s114
                                 + 0.0209509387870313 w_k1_MSFT_s115
                                 + 0.0246375274796929 w_k1_MSFT_s116
                                 + 0.022181490607357 w_k1_MSFT_s117
                                 + 0.0350515380620977 w_k1_PG_s000
                                 + 0.0285105842584419 w_k1_PG_s001
                                 + 0.0613613985415072 w_k1_PG_s002
                                 + 0.0338158593811416 w_k1_PG_s003
                                 + 0.0334282540024226 w_k1_PG_s004
                                 + 0.0177988955488697 w_k1_PG_s005
                                 + 0.0211827417058199 w_k1_PG_s006
                                 + 0.0168347785482155 w_k1_PG_s007
                                 + 0.0373952411802484 w_k1_PG_s010
                                 + 0.0552074815732813 w_k1_PG_s011
                                 + 0.0193455386783683 w_k1_PG_s012
                                 + 0.0292119846341282 w_k1_PG_s013
                                 + 0.0323106665347999 w_k1_PG_s014
                                 + 0.0272765817491208 w_k1_PG_s015
                                 + 0.0170305329617046 w_k1_PG_s016
                                 + 0.0215716331360573 w_k1_PG_s017
                                 + 0.0464802092314392 w_k1_PG_s100
                                 + 0.0221100167011246 w_k1_PG_s101
                                 + 0.0423931851504676 w_k1_PG_s102
                                 + 0.0203217188453544 w_k1_PG_s103
                                 + 0.0226132811286183 w_k1_PG_s104
                                 + 0.039270862534224 w_k1_PG_s105
                                 + 0.0153180794691452 w_k1_PG_s106
                                 + 0.0312273688617627 w_k1_PG_s107
                                 + 0.039899350080373 w_k1_PG_s110
                                 + 0.0268426575993898 w_k1_PG_s111
                                 + 0.0290270024853203 w_k1_PG_s112
                                 + 0.0642313738924531 w_k1_PG_s113
                                 + 0.0524713203208867 w_k1_PG_s114
                                 + 0.0223135006747448 w_k1_PG_s115
                                 + 0.0267264086289265 w_k1_PG_s116
                                 + 0.0213409713871893 w_k1_PG_s117 >= 1.209
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
