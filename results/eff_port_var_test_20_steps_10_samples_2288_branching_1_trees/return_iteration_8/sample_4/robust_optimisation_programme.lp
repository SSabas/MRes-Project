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
 w_k1_ADM_s0_return_constraint:  1.21135 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.995279 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.23977 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.942193 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.928446 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.06952 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.853104 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.749514 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.988397 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.23718 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.01852 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.14779 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.07818 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.760954 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.07525 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.23837 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.788415 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.12275 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.26074 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.983725 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.830226 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.866302 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.36213 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.36976 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.29796 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.10617 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.904489 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.861764 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.08264 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.705432 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.0888 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.05519 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.82543 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.02597 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.818643 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.36079 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.733205 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.970918 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.32253 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.933598 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.34306 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.880457 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.908275 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.23864 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.992286 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.596974 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.16285 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.53492 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.0185 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.907299 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.13364 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.776328 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.26517 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.27798 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.603805 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.36498 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.60662 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.07867 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.964538 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.756577 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.2887 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.37164 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.59679 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.44263 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.995922 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.731408 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.03856 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.663703 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.38178 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.21558 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.795765 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.902429 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.774584 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.5282 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.607805 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.02467 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.35384 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.936653 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.54041 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.889991 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.817073 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.38752 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.13778 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.588234 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.10901 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.70644 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.14668 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.78374 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.17847 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.864437 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.25398 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.1078 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.705151 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.51405 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.77 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.24483 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.713243 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.03079 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.67159 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.28248 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.59692 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.30342 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.01361 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.726322 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.00457 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.664286 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.1746 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.21367 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.650748 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.855115 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.992921 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.63388 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.52193 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.985474 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.26568 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.942015 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.28544 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.923848 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.970375 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.15445 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.933275 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.716376 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.06062 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.4544 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.19195 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.943592 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.11279 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.815275 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.19843 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.05205 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.701844 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.46181 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.31359 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.01349 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.869411 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.970292 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.34934 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.32253 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.23419 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.16157 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.842849 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.858282 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.949834 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.699004 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.1353 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.913859 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.748745 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.869058 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.909753 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.47634 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.605645 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.09821 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.09594 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.02819 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.30446 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.903349 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.806762 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.36852 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.73749 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.643756 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.16403 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.31929 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.60788 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.674982 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.811656 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.936478 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.924627 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.18228 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.946762 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.779383 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.72263 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.63923 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.902332 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.760625 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.00941 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.806206 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.68918 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.29755 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.59919 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.618513 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.960106 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.986516 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.34091 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 2.03641 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.761804 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.16627 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.860934 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.03817 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.96534 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.712784 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.19883 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.91049 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.21862 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.00436 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.00713 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.11612 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.07917 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.962341 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.01055 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.35098 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.31218 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.94516 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.04335 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.864802 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.944094 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.906741 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.743729 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.29253 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.34091 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.912541 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.787368 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.865061 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.16889 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.25937 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.28086 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.2494 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.992625 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.817463 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.02107 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.785191 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.17778 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.05266 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.814501 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.1833 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.12365 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.42279 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.879759 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.14124 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.17123 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.954072 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.34005 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.949036 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.84935 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.22834 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.39426 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.718435 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.14324 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.49391 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.877551 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.746801 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.00508 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.88764 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.978508 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.08171 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.794935 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.03966 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.73575 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.15833 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.829706 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.941699 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.3276 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.02863 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.58253 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.23586 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.31779 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.794615 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.938376 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.811197 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.4983 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.50977 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.819614 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.11537 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.01348 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.23526 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.840111 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.96816 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.21379 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.13369 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.16288 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.871712 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.704589 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.22063 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.68697 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.617402 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.60936 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.55563 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.724631 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.615104 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.2658 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.935542 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.02713 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.34928 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.653376 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.755697 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 2.10715 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.3869 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.08411 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.636638 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.29761 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.842954 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.8755 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.60796 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.40725 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.417163 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.02222 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.748649 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.54801 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.92397 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.773248 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.27423 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.816071 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.0933 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.779532 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.934918 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19467 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.977849 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.24307 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.957331 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.940391 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.05576 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.11252 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.801687 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.00451 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.45629 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.22718 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.904765 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.14627 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.871774 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.08779 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.04115 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.736444 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.30398 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.33058 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.997601 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.790927 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.0436 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.31535 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.10532 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.25194 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.10726 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.937044 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.829911 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.92104 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.690102 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.22736 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.05076 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.783986 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.960733 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.08339 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.44951 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.686858 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.12777 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09461 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.961764 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.13249 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.97723 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.919896 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10265 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.23858 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.872361 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.26463 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.49548 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.04643 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.00318 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.15402 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.93995 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.00686 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.03811 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.885614 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.18284 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.3413 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.18358 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.887355 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.91032 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.25662 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.11557 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.39513 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.10486 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.02483 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.850463 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.962944 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.801056 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.2013 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.12616 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.867451 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.04388 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.983294 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.12977 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.826228 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.01113 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.111000652751072 z_k1_s0000
                                 - 0.0760001207442295 z_k1_s0001
                                 - 0.126000346581514 z_k1_s0002
                                 - 0.0850002106430386 z_k1_s0003
                                 - 0.143000060389468 z_k1_s0004
                                 - 0.132999960501903 z_k1_s0005
                                 - 0.0980000668963298 z_k1_s0006
                                 - 0.140000486424106 z_k1_s0007
                                 - 0.0700002167776588 z_k1_s0010
                                 - 0.0410000756697307 z_k1_s0011
                                 - 0.0820001513394614 z_k1_s0012
                                 - 0.0799999627456686 z_k1_s0013
                                 - 0.0320000249982992 z_k1_s0014
                                 - 0.0459999486537356 z_k1_s0015
                                 - 0.0260000178173662 z_k1_s0016
                                 - 0.0219999997300668 z_k1_s0017
                                 - 0.0630000914080526 z_k1_s0020
                                 - 0.17100043222218 z_k1_s0021
                                 - 0.0990002050127618 z_k1_s0022
                                 - 0.121000274437862 z_k1_s0023
                                 - 0.0770001355876617 z_k1_s0024
                                 - 0.288000049636165 z_k1_s0025
                                 - 0.161000400665317 z_k1_s0026
                                 - 0.09400018923433 z_k1_s0027
                                 - 0.176000498735017 z_k1_s0030
                                 - 0.0470000868348459 z_k1_s0031
                                 - 0.0480000742185431 z_k1_s0032
                                 - 0.0980001232044807 z_k1_s0033
                                 - 0.0590001208395062 z_k1_s0034
                                 - 0.0810001522813344 z_k1_s0035
                                 - 0.0650001069417874 z_k1_s0036
                                 - 0.0440000628836564 z_k1_s0037
                                 - 0.0520000794857193 z_k1_s0040
                                 - 0.0310000582895854 z_k1_s0041
                                 - 0.0230000367148405 z_k1_s0042
                                 - 0.0240000647242345 z_k1_s0043
                                 - 0.0340000613176045 z_k1_s0044
                                 - 0.10500010598067 z_k1_s0045
                                 - 0.0390000798643302 z_k1_s0046
                                 - 0.0970001249060064 z_k1_s0047
                                 - 0.112999783256373 z_k1_s0050
                                 - 0.0860000213802037 z_k1_s0051
                                 - 0.130000452330839 z_k1_s0052
                                 - 0.0510000690220213 z_k1_s0053
                                 - 0.158000414217385 z_k1_s0054
                                 - 0.0910003166315871 z_k1_s0055
                                 - 0.144000433274112 z_k1_s0056
                                 - 0.108000104705427 z_k1_s0057
                                 - 0.0420000787620704 z_k1_s0060
                                 - 0.11999998289123 z_k1_s0061
                                 - 0.0629998638926863 z_k1_s0062
                                 - 0.0930002833661927 z_k1_s0063
                                 - 0.062000377244439 z_k1_s0064
                                 - 0.0330000659202048 z_k1_s0065
                                 - 0.0680001221385813 z_k1_s0066
                                 - 0.0840003835245135 z_k1_s0067
                                 - 0.206999936922721 z_k1_s0070
                                 - 0.255000439515033 z_k1_s0071
                                 - 0.104000345447801 z_k1_s0072
                                 - 0.0980002826237623 z_k1_s0073
                                 - 0.360000795433508 z_k1_s0074
                                 - 0.114000301454091 z_k1_s0075
                                 - 0.0820002637934333 z_k1_s0076
                                 - 0.267000565163111 z_k1_s0077
                                 - 0.319000306218749 z_k1_s0100
                                 - 0.20099967769576 z_k1_s0101
                                 - 0.143999858759551 z_k1_s0102
                                 - 0.177999868798565 z_k1_s0103
                                 - 0.247998942403446 z_k1_s0104
                                 - 0.307999384177714 z_k1_s0105
                                 - 0.691998366737214 z_k1_s0106
                                 - 0.313000021741584 z_k1_s0107
                                 - 0.139000356378736 z_k1_s0110
                                 - 0.26699963848363 z_k1_s0111
                                 - 0.304000314497705 z_k1_s0112
                                 - 0.0839999472060429 z_k1_s0113
                                 - 0.0829998915083915 z_k1_s0114
                                 - 0.06199999125682 z_k1_s0115
                                 - 0.137000475783271 z_k1_s0116
                                 - 0.0779999592198912 z_k1_s0117
                                 - 0.462999910335128 z_k1_s0120
                                 - 0.225999536033371 z_k1_s0121
                                 - 0.161999603749752 z_k1_s0122
                                 - 0.325998836640586 z_k1_s0123
                                 - 0.253999236595664 z_k1_s0124
                                 - 0.186999572801177 z_k1_s0125
                                 - 0.604997178889946 z_k1_s0126
                                 - 0.654999419386742 z_k1_s0127
                                 - 0.298999587770156 z_k1_s0130
                                 - 0.203999714364923 z_k1_s0131
                                 - 0.120999962863485 z_k1_s0132
                                 - 0.130999977116663 z_k1_s0133
                                 - 0.23300048929901 z_k1_s0134
                                 - 0.110000025784976 z_k1_s0135
                                 - 0.0830000135013921 z_k1_s0136
                                 - 0.129000000466023 z_k1_s0137
                                 - 0.16399953853191 z_k1_s0140
                                 - 0.094999947994585 z_k1_s0141
                                 - 0.238999815159038 z_k1_s0142
                                 - 0.165000004980016 z_k1_s0143
                                 - 0.189000636740398 z_k1_s0144
                                 - 0.247000528745555 z_k1_s0145
                                 - 0.163000581082971 z_k1_s0146
                                 - 0.247000528745555 z_k1_s0147
                                 - 0.13900012321573 z_k1_s0150
                                 - 0.21400069039709 z_k1_s0151
                                 - 0.329999809396852 z_k1_s0152
                                 - 0.297999420879409 z_k1_s0153
                                 - 0.20300008625027 z_k1_s0154
                                 - 0.487000125716132 z_k1_s0155
                                 - 0.259000664405752 z_k1_s0156
                                 - 0.10500014285363 z_k1_s0157
                                 - 0.243999095200893 z_k1_s0160
                                 - 0.457997621912036 z_k1_s0161
                                 - 0.289999168754382 z_k1_s0162
                                 - 0.373998249333674 z_k1_s0163
                                 - 0.747996498667348 z_k1_s0164
                                 - 0.174999422869264 z_k1_s0165
                                 - 0.340999434389186 z_k1_s0166
                                 - 0.289999168754382 z_k1_s0167
                                 - 0.154000043385781 z_k1_s0170
                                 - 0.36300153271736 z_k1_s0171
                                 - 0.629000204919212 z_k1_s0172
                                 - 0.739000470409199 z_k1_s0173
                                 - 0.410000989553588 z_k1_s0174
                                 - 0.415999034233652 z_k1_s0175
                                 - 0.303000074703834 z_k1_s0176
                                 - 0.269000320943494 z_k1_s0177
                                 - 0.745999327881246 z_k1_s1000
                                 - 0.518999600378549 z_k1_s1001
                                 - 0.699001953413081 z_k1_s1002
                                 - 0.464999739668701 z_k1_s1003
                                 - 0.691002130567272 z_k1_s1004
                                 - 0.279000266401477 z_k1_s1005
                                 - 0.264000070315266 z_k1_s1006
                                 - 0.563000739031776 z_k1_s1007
                                 - 0.301999787269407 z_k1_s1010
                                 - 0.503001111645602 z_k1_s1011
                                 - 0.714999778530193 z_k1_s1012
                                 - 0.476001399080101 z_k1_s1013
                                 - 0.619999866096106 z_k1_s1014
                                 - 0.627000253464898 z_k1_s1015
                                 - 0.646001898751566 z_k1_s1016
                                 - 0.268000103035077 z_k1_s1017
                                 - 0.350000111567281 z_k1_s1020
                                 - 0.27900001322532 z_k1_s1021
                                 - 0.200000001695568 z_k1_s1022
                                 - 0.691000320798295 z_k1_s1023
                                 - 0.574998104374094 z_k1_s1024
                                 - 0.551001840011928 z_k1_s1025
                                 - 0.760000440843308 z_k1_s1026
                                 - 0.938000246872295 z_k1_s1027
                                 - 0.884998503295329 z_k1_s1030
                                 - 0.219000072978018 z_k1_s1031
                                 - 0.267999780750372 z_k1_s1032
                                 - 0.65500061640444 z_k1_s1033
                                 - 0.497001499375237 z_k1_s1034
                                 - 0.420999164855785 z_k1_s1035
                                 - 0.494999717046354 z_k1_s1036
                                 - 0.192999790594874 z_k1_s1037
                                 - 0.573999394770487 z_k1_s1040
                                 - 0.441000058214055 z_k1_s1041
                                 - 0.599002316642835 z_k1_s1042
                                 - 0.581002119695269 z_k1_s1043
                                 - 0.924998468246713 z_k1_s1044
                                 - 0.352000143751381 z_k1_s1045
                                 - 0.924002164972867 z_k1_s1046
                                 - 0.37099998087371 z_k1_s1047
                                 - 0.411999883178274 z_k1_s1050
                                 - 0.839999687402306 z_k1_s1051
                                 - 1.45999854417543 z_k1_s1052
                                 - 0.706999673021941 z_k1_s1053
                                 - 1.00300357786848 z_k1_s1054
                                 - 0.517999934981422 z_k1_s1055
                                 - 2.03499618581986 z_k1_s1056
                                 - 0.690000235973324 z_k1_s1057
                                 - 2.63500809581336 z_k1_s1060
                                 - 0.815001085474353 z_k1_s1061
                                 - 1.88399794642852 z_k1_s1062
                                 - 0.570000461250339 z_k1_s1063
                                 - 1.08400198469981 z_k1_s1064
                                 - 2.73400715360376 z_k1_s1065
                                 - 2.06599673060961 z_k1_s1066
                                 - 0.991001398282426 z_k1_s1067
                                 - 1.49300822037812 z_k1_s1070
                                 - 0.626001895641252 z_k1_s1071
                                 - 0.854004066842039 z_k1_s1072
                                 - 0.480001532571796 z_k1_s1073
                                 - 0.667001975004526 z_k1_s1074
                                 - 0.477001265273725 z_k1_s1075
                                 - 1.41000192473468 z_k1_s1076
                                 - 2.24000715200171 z_k1_s1077
                                 - 0.86800111159263 z_k1_s1100
                                 - 0.498999837506478 z_k1_s1101
                                 - 0.540000418294522 z_k1_s1102
                                 - 0.33400056617895 z_k1_s1103
                                 - 0.458999367298783 z_k1_s1104
                                 - 0.348000533101331 z_k1_s1105
                                 - 0.603999589424337 z_k1_s1106
                                 - 0.301000474733069 z_k1_s1107
                                 - 0.124000022427901 z_k1_s1110
                                 - 0.147000085474961 z_k1_s1111
                                 - 0.209000096688911 z_k1_s1112
                                 - 0.131999356488345 z_k1_s1113
                                 - 0.0479998987592829 z_k1_s1114
                                 - 0.178999855714623 z_k1_s1115
                                 - 0.139999907547732 z_k1_s1116
                                 - 0.23799996453104 z_k1_s1117
                                 - 0.411998933249349 z_k1_s1120
                                 - 0.387999021254173 z_k1_s1121
                                 - 0.688997362528494 z_k1_s1122
                                 - 0.518000098557235 z_k1_s1123
                                 - 0.310998970603869 z_k1_s1124
                                 - 0.564997373221265 z_k1_s1125
                                 - 0.509999239894545 z_k1_s1126
                                 - 1.04699804778455 z_k1_s1127
                                 - 0.20599917435121 z_k1_s1130
                                 - 0.899996103923691 z_k1_s1131
                                 - 0.755997124014627 z_k1_s1132
                                 - 0.347998691876524 z_k1_s1133
                                 - 0.620998700222891 z_k1_s1134
                                 - 0.702995502096173 z_k1_s1135
                                 - 0.436998571077018 z_k1_s1136
                                 - 0.987998235525944 z_k1_s1137
                                 - 0.13999949975242 z_k1_s1140
                                 - 0.264999920101604 z_k1_s1141
                                 - 0.11999957121636 z_k1_s1142
                                 - 0.132999929364243 z_k1_s1143
                                 - 0.103999859587194 z_k1_s1144
                                 - 0.0669999339955628 z_k1_s1145
                                 - 0.0909998482388348 z_k1_s1146
                                 - 0.23599938792519 z_k1_s1147
                                 - 0.761996836298415 z_k1_s1150
                                 - 0.679000034247025 z_k1_s1151
                                 - 0.296999045444626 z_k1_s1152
                                 - 0.672999277884767 z_k1_s1153
                                 - 0.300998974487822 z_k1_s1154
                                 - 0.388998707634331 z_k1_s1155
                                 - 0.291999134140631 z_k1_s1156
                                 - 0.920997466955313 z_k1_s1157
                                 - 0.142000168155493 z_k1_s1160
                                 - 0.14300033463521 z_k1_s1161
                                 - 0.186999835754651 z_k1_s1162
                                 - 0.241999864152961 z_k1_s1163
                                 - 0.0799998871977762 z_k1_s1164
                                 - 0.213999114714938 z_k1_s1165
                                 - 0.0879999150374943 z_k1_s1166
                                 - 0.207999419834654 z_k1_s1167
                                 - 0.737998163016379 z_k1_s1170
                                 - 0.24399949439818 z_k1_s1171
                                 - 0.815996503695637 z_k1_s1172
                                 - 0.844000015521564 z_k1_s1173
                                 - 0.190999569343722 z_k1_s1174
                                 - 0.583998871936474 z_k1_s1175
                                 - 0.765000476707665 z_k1_s1176
                                 - 0.823996076791424 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.954418 w_k1_ADM_s000
                                 + 0.908056 w_k1_AXP_s000
                                 + 0.763216 w_k1_DIS_s000
                                 + 0.875009 w_k1_FDX_s000
                                 + 0.832958 w_k1_IBM_s000
                                 + 0.966136 w_k1_INTC_s000
                                 + 0.840796 w_k1_KO_s000
                                 + 0.722245 w_k1_MAT_s000
                                 + 0.809355 w_k1_MSFT_s000
                                 + 0.866874 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.1305 w_k1_ADM_s000
                                 + 1.15008 w_k1_AXP_s000
                                 + 1.01826 w_k1_DIS_s000 + 1.0738 w_k1_FDX_s000
                                 + 1.39092 w_k1_IBM_s000
                                 + 0.878169 w_k1_INTC_s000
                                 + 1.12245 w_k1_KO_s000 + 1.36084 w_k1_MAT_s000
                                 + 0.865237 w_k1_MSFT_s000
                                 + 0.92713 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.83314 w_k1_ADM_s000
                                 + 0.602483 w_k1_AXP_s000
                                 + 0.64936 w_k1_DIS_s000
                                 + 0.697456 w_k1_FDX_s000
                                 + 0.696544 w_k1_IBM_s000
                                 + 0.649554 w_k1_INTC_s000
                                 + 0.785568 w_k1_KO_s000
                                 + 0.822112 w_k1_MAT_s000
                                 + 0.762427 w_k1_MSFT_s000
                                 + 0.785975 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.32917 w_k1_ADM_s000
                                 + 1.16869 w_k1_AXP_s000
                                 + 1.57443 w_k1_DIS_s000
                                 + 1.22485 w_k1_FDX_s000
                                 + 0.946385 w_k1_IBM_s000
                                 + 1.05153 w_k1_INTC_s000
                                 + 1.11379 w_k1_KO_s000 + 0.86796 w_k1_MAT_s000
                                 + 1.17321 w_k1_MSFT_s000
                                 + 1.12256 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.20001 w_k1_ADM_s000
                                 + 1.05128 w_k1_AXP_s000
                                 + 0.750133 w_k1_DIS_s000
                                 + 0.846605 w_k1_FDX_s000
                                 + 1.05912 w_k1_IBM_s000
                                 + 0.77121 w_k1_INTC_s000
                                 + 0.788732 w_k1_KO_s000
                                 + 0.975784 w_k1_MAT_s000
                                 + 0.661509 w_k1_MSFT_s000
                                 + 0.844794 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.908745 w_k1_ADM_s000
                                 + 1.15254 w_k1_AXP_s000
                                 + 0.956963 w_k1_DIS_s000
                                 + 1.04729 w_k1_FDX_s000
                                 + 0.690206 w_k1_IBM_s000
                                 + 0.948361 w_k1_INTC_s000
                                 + 0.773993 w_k1_KO_s000
                                 + 0.848801 w_k1_MAT_s000
                                 + 0.878757 w_k1_MSFT_s000
                                 + 0.903234 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.42709 w_k1_ADM_s000
                                 + 1.48837 w_k1_AXP_s000
                                 + 1.44931 w_k1_DIS_s000
                                 + 1.20871 w_k1_FDX_s000
                                 + 1.58671 w_k1_IBM_s000
                                 + 1.07034 w_k1_INTC_s000 + 1.283 w_k1_KO_s000
                                 + 2.17693 w_k1_MAT_s000
                                 + 1.28157 w_k1_MSFT_s000
                                 + 1.21613 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.02412 w_k1_ADM_s000
                                 + 1.18561 w_k1_AXP_s000
                                 + 1.26413 w_k1_DIS_s000
                                 + 1.09282 w_k1_FDX_s000
                                 + 1.06071 w_k1_IBM_s000
                                 + 1.00986 w_k1_INTC_s000
                                 + 1.09617 w_k1_KO_s000 + 1.46167 w_k1_MAT_s000
                                 + 1.12958 w_k1_MSFT_s000
                                 + 1.14532 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.02633 w_k1_ADM_s001
                                 + 0.930441 w_k1_AXP_s001
                                 + 1.03831 w_k1_DIS_s001
                                 + 1.07791 w_k1_FDX_s001
                                 + 0.757183 w_k1_IBM_s001
                                 + 0.892983 w_k1_INTC_s001
                                 + 0.889899 w_k1_KO_s001
                                 + 0.915725 w_k1_MAT_s001
                                 + 1.11319 w_k1_MSFT_s001
                                 + 0.988009 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.07219 w_k1_ADM_s001
                                 + 1.23317 w_k1_AXP_s001 + 1.0477 w_k1_DIS_s001
                                 + 1.0533 w_k1_FDX_s001
                                 + 0.996616 w_k1_IBM_s001
                                 + 0.70985 w_k1_INTC_s001
                                 + 0.864213 w_k1_KO_s001 + 0.9612 w_k1_MAT_s001
                                 + 1.00574 w_k1_MSFT_s001
                                 + 0.95016 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.37191 w_k1_ADM_s001
                                 + 1.39887 w_k1_AXP_s001
                                 + 1.32437 w_k1_DIS_s001
                                 + 1.33923 w_k1_FDX_s001
                                 + 0.971491 w_k1_IBM_s001
                                 + 0.92796 w_k1_INTC_s001
                                 + 1.05436 w_k1_KO_s001 + 1.17748 w_k1_MAT_s001
                                 + 1.1284 w_k1_MSFT_s001 + 1.10217 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.953115 w_k1_ADM_s001
                                 + 1.1297 w_k1_AXP_s001
                                 + 0.984243 w_k1_DIS_s001
                                 + 0.85376 w_k1_FDX_s001
                                 + 1.40037 w_k1_IBM_s001
                                 + 0.783181 w_k1_INTC_s001
                                 + 1.01775 w_k1_KO_s001 + 1.49789 w_k1_MAT_s001
                                 + 0.81392 w_k1_MSFT_s001
                                 + 1.04387 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.865576 w_k1_ADM_s001
                                 + 0.737288 w_k1_AXP_s001
                                 + 0.614321 w_k1_DIS_s001
                                 + 0.698988 w_k1_FDX_s001
                                 + 0.974167 w_k1_IBM_s001
                                 + 0.64454 w_k1_INTC_s001
                                 + 0.720029 w_k1_KO_s001
                                 + 0.891225 w_k1_MAT_s001
                                 + 0.822808 w_k1_MSFT_s001
                                 + 0.839339 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.783702 w_k1_ADM_s001
                                 + 0.667762 w_k1_AXP_s001
                                 + 0.607212 w_k1_DIS_s001
                                 + 0.787994 w_k1_FDX_s001
                                 + 0.659357 w_k1_IBM_s001
                                 + 0.508437 w_k1_INTC_s001
                                 + 0.685717 w_k1_KO_s001
                                 + 0.843612 w_k1_MAT_s001
                                 + 0.719243 w_k1_MSFT_s001
                                 + 0.852969 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.956112 w_k1_ADM_s001
                                 + 1.27198 w_k1_AXP_s001
                                 + 0.813278 w_k1_DIS_s001
                                 + 1.09468 w_k1_FDX_s001
                                 + 0.819158 w_k1_IBM_s001
                                 + 0.691871 w_k1_INTC_s001
                                 + 0.725715 w_k1_KO_s001
                                 + 1.28115 w_k1_MAT_s001
                                 + 0.821777 w_k1_MSFT_s001
                                 + 1.00962 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.14135 w_k1_ADM_s001
                                 + 1.3963 w_k1_AXP_s001
                                 + 0.960809 w_k1_DIS_s001
                                 + 0.883572 w_k1_FDX_s001
                                 + 1.93776 w_k1_IBM_s001
                                 + 0.782359 w_k1_INTC_s001
                                 + 1.11805 w_k1_KO_s001 + 1.53803 w_k1_MAT_s001
                                 + 0.778766 w_k1_MSFT_s001
                                 + 0.945201 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.39538 w_k1_ADM_s002
                                 + 1.29158 w_k1_AXP_s002 + 1.5925 w_k1_DIS_s002
                                 + 1.3376 w_k1_FDX_s002
                                 + 0.890141 w_k1_IBM_s002
                                 + 1.37535 w_k1_INTC_s002
                                 + 0.943265 w_k1_KO_s002
                                 + 0.568066 w_k1_MAT_s002
                                 + 1.46216 w_k1_MSFT_s002
                                 + 0.940957 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.762005 w_k1_ADM_s002
                                 + 0.951908 w_k1_AXP_s002
                                 + 1.14577 w_k1_DIS_s002
                                 + 0.999024 w_k1_FDX_s002
                                 + 1.22711 w_k1_IBM_s002
                                 + 1.05226 w_k1_INTC_s002
                                 + 1.05151 w_k1_KO_s002 + 1.01229 w_k1_MAT_s002
                                 + 1.06201 w_k1_MSFT_s002
                                 + 0.92384 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.23646 w_k1_ADM_s002
                                 + 1.3967 w_k1_AXP_s002 + 1.38284 w_k1_DIS_s002
                                 + 1.24263 w_k1_FDX_s002
                                 + 1.43929 w_k1_IBM_s002
                                 + 1.56114 w_k1_INTC_s002
                                 + 1.35052 w_k1_KO_s002 + 1.26232 w_k1_MAT_s002
                                 + 1.22779 w_k1_MSFT_s002
                                 + 1.15592 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.14183 w_k1_ADM_s002
                                 + 1.21415 w_k1_AXP_s002
                                 + 0.820706 w_k1_DIS_s002
                                 + 1.02518 w_k1_FDX_s002
                                 + 1.00069 w_k1_IBM_s002
                                 + 1.06673 w_k1_INTC_s002
                                 + 0.834603 w_k1_KO_s002
                                 + 1.04173 w_k1_MAT_s002
                                 + 1.03672 w_k1_MSFT_s002
                                 + 0.828319 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.828754 w_k1_ADM_s002
                                 + 0.7721 w_k1_AXP_s002
                                 + 0.703901 w_k1_DIS_s002
                                 + 0.732253 w_k1_FDX_s002
                                 + 1.19961 w_k1_IBM_s002
                                 + 0.799049 w_k1_INTC_s002
                                 + 1.11342 w_k1_KO_s002 + 1.28612 w_k1_MAT_s002
                                 + 1.00869 w_k1_MSFT_s002
                                 + 0.913966 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.980109 w_k1_ADM_s002
                                 + 0.823517 w_k1_AXP_s002
                                 + 0.776939 w_k1_DIS_s002
                                 + 0.837166 w_k1_FDX_s002
                                 + 0.817163 w_k1_IBM_s002
                                 + 0.840045 w_k1_INTC_s002
                                 + 0.77152 w_k1_KO_s002
                                 + 0.632883 w_k1_MAT_s002
                                 + 0.838115 w_k1_MSFT_s002
                                 + 0.767203 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.10315 w_k1_ADM_s002
                                 + 1.04659 w_k1_AXP_s002
                                 + 1.00354 w_k1_DIS_s002
                                 + 1.05341 w_k1_FDX_s002
                                 + 0.728686 w_k1_IBM_s002
                                 + 0.973814 w_k1_INTC_s002
                                 + 0.875382 w_k1_KO_s002
                                 + 0.835867 w_k1_MAT_s002
                                 + 1.14742 w_k1_MSFT_s002
                                 + 0.912155 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.53983 w_k1_ADM_s002
                                 + 1.53931 w_k1_AXP_s002
                                 + 1.32127 w_k1_DIS_s002
                                 + 1.25947 w_k1_FDX_s002
                                 + 1.05591 w_k1_IBM_s002
                                 + 1.26087 w_k1_INTC_s002
                                 + 1.09828 w_k1_KO_s002 + 1.17526 w_k1_MAT_s002
                                 + 1.32237 w_k1_MSFT_s002
                                 + 0.999637 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.80349 w_k1_ADM_s003
                                 + 0.682689 w_k1_AXP_s003
                                 + 0.7961 w_k1_DIS_s003
                                 + 0.799304 w_k1_FDX_s003
                                 + 0.648141 w_k1_IBM_s003
                                 + 0.964888 w_k1_INTC_s003
                                 + 0.749414 w_k1_KO_s003
                                 + 0.463234 w_k1_MAT_s003
                                 + 0.867953 w_k1_MSFT_s003
                                 + 0.801247 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.58531 w_k1_ADM_s003
                                 + 1.76888 w_k1_AXP_s003
                                 + 1.74749 w_k1_DIS_s003
                                 + 1.44828 w_k1_FDX_s003 + 1.1394 w_k1_IBM_s003
                                 + 1.11614 w_k1_INTC_s003 + 1.0218 w_k1_KO_s003
                                 + 1.65353 w_k1_MAT_s003
                                 + 1.14844 w_k1_MSFT_s003
                                 + 1.00726 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.30126 w_k1_ADM_s003
                                 + 1.54394 w_k1_AXP_s003
                                 + 1.41878 w_k1_DIS_s003
                                 + 1.23611 w_k1_FDX_s003
                                 + 1.32502 w_k1_IBM_s003
                                 + 1.3102 w_k1_INTC_s003 + 1.31101 w_k1_KO_s003
                                 + 1.33975 w_k1_MAT_s003
                                 + 1.15778 w_k1_MSFT_s003
                                 + 1.06103 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.04137 w_k1_ADM_s003
                                 + 1.25891 w_k1_AXP_s003
                                 + 0.849618 w_k1_DIS_s003
                                 + 0.965244 w_k1_FDX_s003
                                 + 0.953408 w_k1_IBM_s003
                                 + 0.904429 w_k1_INTC_s003
                                 + 0.829465 w_k1_KO_s003 + 1.2705 w_k1_MAT_s003
                                 + 0.817675 w_k1_MSFT_s003
                                 + 0.845148 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.11231 w_k1_ADM_s003
                                 + 1.31357 w_k1_AXP_s003
                                 + 1.24357 w_k1_DIS_s003
                                 + 1.06685 w_k1_FDX_s003
                                 + 1.63991 w_k1_IBM_s003
                                 + 0.817224 w_k1_INTC_s003
                                 + 1.11467 w_k1_KO_s003 + 1.65806 w_k1_MAT_s003
                                 + 0.920374 w_k1_MSFT_s003
                                 + 0.98338 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.980285 w_k1_ADM_s003
                                 + 0.864812 w_k1_AXP_s003
                                 + 1.23584 w_k1_DIS_s003
                                 + 0.978283 w_k1_FDX_s003
                                 + 0.989211 w_k1_IBM_s003
                                 + 0.942228 w_k1_INTC_s003
                                 + 1.04378 w_k1_KO_s003 + 1.028 w_k1_MAT_s003
                                 + 1.14284 w_k1_MSFT_s003
                                 + 1.03539 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.96783 w_k1_ADM_s003
                                 + 0.708233 w_k1_AXP_s003
                                 + 0.740103 w_k1_DIS_s003
                                 + 0.737282 w_k1_FDX_s003
                                 + 1.1461 w_k1_IBM_s003
                                 + 0.785814 w_k1_INTC_s003
                                 + 0.972263 w_k1_KO_s003
                                 + 1.14611 w_k1_MAT_s003
                                 + 0.815045 w_k1_MSFT_s003
                                 + 0.883546 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.969094 w_k1_ADM_s003
                                 + 0.847547 w_k1_AXP_s003
                                 + 0.999693 w_k1_DIS_s003
                                 + 0.774525 w_k1_FDX_s003
                                 + 0.973199 w_k1_IBM_s003
                                 + 0.757608 w_k1_INTC_s003
                                 + 0.791748 w_k1_KO_s003
                                 + 1.02913 w_k1_MAT_s003
                                 + 0.77937 w_k1_MSFT_s003
                                 + 0.720553 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.95448 w_k1_ADM_s004
                                 + 0.708646 w_k1_AXP_s004
                                 + 0.745467 w_k1_DIS_s004
                                 + 0.744585 w_k1_FDX_s004
                                 + 0.917926 w_k1_IBM_s004
                                 + 0.677926 w_k1_INTC_s004
                                 + 0.909233 w_k1_KO_s004
                                 + 0.887266 w_k1_MAT_s004
                                 + 0.761226 w_k1_MSFT_s004
                                 + 0.938411 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.44644 w_k1_ADM_s004
                                 + 1.33643 w_k1_AXP_s004
                                 + 1.55994 w_k1_DIS_s004
                                 + 1.38151 w_k1_FDX_s004
                                 + 1.55893 w_k1_IBM_s004
                                 + 1.14918 w_k1_INTC_s004 + 1.5483 w_k1_KO_s004
                                 + 1.44097 w_k1_MAT_s004
                                 + 1.27242 w_k1_MSFT_s004
                                 + 1.33436 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.870048 w_k1_ADM_s004
                                 + 0.692307 w_k1_AXP_s004
                                 + 0.996005 w_k1_DIS_s004
                                 + 0.868737 w_k1_FDX_s004
                                 + 1.11801 w_k1_IBM_s004
                                 + 0.824436 w_k1_INTC_s004
                                 + 1.10132 w_k1_KO_s004 + 0.92181 w_k1_MAT_s004
                                 + 0.964652 w_k1_MSFT_s004
                                 + 0.985643 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.46241 w_k1_ADM_s004
                                 + 1.36706 w_k1_AXP_s004
                                 + 1.49589 w_k1_DIS_s004
                                 + 1.41684 w_k1_FDX_s004
                                 + 1.13466 w_k1_IBM_s004
                                 + 1.00669 w_k1_INTC_s004
                                 + 1.20412 w_k1_KO_s004 + 1.5504 w_k1_MAT_s004
                                 + 1.17506 w_k1_MSFT_s004
                                 + 1.06032 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.867492 w_k1_ADM_s004
                                 + 0.709584 w_k1_AXP_s004
                                 + 0.601579 w_k1_DIS_s004
                                 + 0.695781 w_k1_FDX_s004
                                 + 0.718713 w_k1_IBM_s004
                                 + 0.643789 w_k1_INTC_s004
                                 + 0.715203 w_k1_KO_s004
                                 + 0.659282 w_k1_MAT_s004
                                 + 0.677411 w_k1_MSFT_s004
                                 + 0.755823 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.872764 w_k1_ADM_s004
                                 + 0.917154 w_k1_AXP_s004
                                 + 0.937384 w_k1_DIS_s004
                                 + 0.937507 w_k1_FDX_s004
                                 + 1.27407 w_k1_IBM_s004
                                 + 0.807299 w_k1_INTC_s004 + 1.149 w_k1_KO_s004
                                 + 0.942789 w_k1_MAT_s004
                                 + 0.935227 w_k1_MSFT_s004
                                 + 1.02435 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.973758 w_k1_ADM_s004
                                 + 0.933005 w_k1_AXP_s004
                                 + 0.794023 w_k1_DIS_s004
                                 + 0.93774 w_k1_FDX_s004
                                 + 0.66776 w_k1_IBM_s004
                                 + 0.801918 w_k1_INTC_s004
                                 + 0.80728 w_k1_KO_s004
                                 + 0.906503 w_k1_MAT_s004
                                 + 0.831849 w_k1_MSFT_s004
                                 + 1.0109 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.19541 w_k1_ADM_s004
                                 + 1.28532 w_k1_AXP_s004
                                 + 1.15598 w_k1_DIS_s004
                                 + 1.14874 w_k1_FDX_s004
                                 + 1.00255 w_k1_IBM_s004
                                 + 0.81161 w_k1_INTC_s004
                                 + 1.07153 w_k1_KO_s004 + 1.48555 w_k1_MAT_s004
                                 + 0.991768 w_k1_MSFT_s004
                                 + 1.06737 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.80864 w_k1_ADM_s005
                                 + 1.07557 w_k1_AXP_s005
                                 + 1.35975 w_k1_DIS_s005
                                 + 1.20522 w_k1_FDX_s005 + 0.9651 w_k1_IBM_s005
                                 + 1.06871 w_k1_INTC_s005
                                 + 0.989819 w_k1_KO_s005 + 1.3969 w_k1_MAT_s005
                                 + 1.30237 w_k1_MSFT_s005
                                 + 0.944925 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.68621 w_k1_ADM_s005
                                 + 0.741379 w_k1_AXP_s005
                                 + 0.620249 w_k1_DIS_s005
                                 + 0.741142 w_k1_FDX_s005
                                 + 0.653426 w_k1_IBM_s005
                                 + 0.736836 w_k1_INTC_s005
                                 + 0.768831 w_k1_KO_s005
                                 + 1.34157 w_k1_MAT_s005
                                 + 0.808012 w_k1_MSFT_s005
                                 + 0.93808 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.06224 w_k1_ADM_s005
                                 + 1.23371 w_k1_AXP_s005
                                 + 1.26196 w_k1_DIS_s005
                                 + 1.02746 w_k1_FDX_s005
                                 + 1.55525 w_k1_IBM_s005
                                 + 1.22541 w_k1_INTC_s005
                                 + 1.41877 w_k1_KO_s005 + 2.26735 w_k1_MAT_s005
                                 + 1.16959 w_k1_MSFT_s005
                                 + 1.23636 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.700383 w_k1_ADM_s005
                                 + 1.03284 w_k1_AXP_s005
                                 + 0.94741 w_k1_DIS_s005 + 1.0135 w_k1_FDX_s005
                                 + 1.01707 w_k1_IBM_s005
                                 + 0.961808 w_k1_INTC_s005
                                 + 1.07802 w_k1_KO_s005 + 1.52596 w_k1_MAT_s005
                                 + 1.06969 w_k1_MSFT_s005
                                 + 0.936362 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.848706 w_k1_ADM_s005
                                 + 1.07967 w_k1_AXP_s005
                                 + 0.931213 w_k1_DIS_s005
                                 + 1.03201 w_k1_FDX_s005
                                 + 0.696629 w_k1_IBM_s005
                                 + 0.981403 w_k1_INTC_s005
                                 + 0.910065 w_k1_KO_s005
                                 + 0.852758 w_k1_MAT_s005
                                 + 0.916252 w_k1_MSFT_s005
                                 + 0.931975 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.711038 w_k1_ADM_s005
                                 + 0.753105 w_k1_AXP_s005
                                 + 0.786031 w_k1_DIS_s005
                                 + 0.786932 w_k1_FDX_s005
                                 + 1.06224 w_k1_IBM_s005
                                 + 0.883661 w_k1_INTC_s005
                                 + 0.913242 w_k1_KO_s005
                                 + 1.43444 w_k1_MAT_s005
                                 + 0.889333 w_k1_MSFT_s005
                                 + 0.928474 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.09659 w_k1_ADM_s005
                                 + 1.70484 w_k1_AXP_s005
                                 + 1.24329 w_k1_DIS_s005
                                 + 1.22575 w_k1_FDX_s005
                                 + 1.10996 w_k1_IBM_s005
                                 + 1.16631 w_k1_INTC_s005
                                 + 1.07651 w_k1_KO_s005 + 1.72779 w_k1_MAT_s005
                                 + 1.05364 w_k1_MSFT_s005 + 1.0991 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.895657 w_k1_ADM_s005
                                 + 0.873512 w_k1_AXP_s005
                                 + 1.05732 w_k1_DIS_s005
                                 + 0.902811 w_k1_FDX_s005
                                 + 1.04555 w_k1_IBM_s005
                                 + 0.966636 w_k1_INTC_s005
                                 + 1.14697 w_k1_KO_s005 + 1.44774 w_k1_MAT_s005
                                 + 1.08047 w_k1_MSFT_s005
                                 + 0.944883 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.44309 w_k1_ADM_s006
                                 + 1.40239 w_k1_AXP_s006
                                 + 1.59424 w_k1_DIS_s006 + 1.5006 w_k1_FDX_s006
                                 + 1.20994 w_k1_IBM_s006
                                 + 1.24817 w_k1_INTC_s006
                                 + 1.13092 w_k1_KO_s006
                                 + 0.767014 w_k1_MAT_s006
                                 + 1.31428 w_k1_MSFT_s006
                                 + 1.21521 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.08769 w_k1_ADM_s006
                                 + 0.93127 w_k1_AXP_s006
                                 + 0.982167 w_k1_DIS_s006
                                 + 0.915877 w_k1_FDX_s006
                                 + 1.22274 w_k1_IBM_s006
                                 + 1.06975 w_k1_INTC_s006
                                 + 0.951654 w_k1_KO_s006
                                 + 0.613033 w_k1_MAT_s006
                                 + 0.939263 w_k1_MSFT_s006
                                 + 0.972805 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.965436 w_k1_ADM_s006
                                 + 1.06975 w_k1_AXP_s006
                                 + 1.42919 w_k1_DIS_s006
                                 + 1.24234 w_k1_FDX_s006
                                 + 1.17006 w_k1_IBM_s006
                                 + 0.898216 w_k1_INTC_s006
                                 + 1.0733 w_k1_KO_s006 + 0.912499 w_k1_MAT_s006
                                 + 1.06445 w_k1_MSFT_s006
                                 + 1.06411 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.02094 w_k1_ADM_s006
                                 + 0.904907 w_k1_AXP_s006
                                 + 0.771144 w_k1_DIS_s006
                                 + 0.976975 w_k1_FDX_s006
                                 + 0.821786 w_k1_IBM_s006
                                 + 0.90769 w_k1_INTC_s006
                                 + 0.771584 w_k1_KO_s006
                                 + 0.68929 w_k1_MAT_s006
                                 + 0.82823 w_k1_MSFT_s006
                                 + 0.829721 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.06315 w_k1_ADM_s006
                                 + 1.1889 w_k1_AXP_s006 + 1.20181 w_k1_DIS_s006
                                 + 1.09972 w_k1_FDX_s006
                                 + 1.40717 w_k1_IBM_s006
                                 + 0.981072 w_k1_INTC_s006
                                 + 1.09665 w_k1_KO_s006 + 1.10937 w_k1_MAT_s006
                                 + 1.0501 w_k1_MSFT_s006 + 1.09199 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.13107 w_k1_ADM_s006
                                 + 0.964769 w_k1_AXP_s006
                                 + 1.08579 w_k1_DIS_s006
                                 + 1.16966 w_k1_FDX_s006
                                 + 0.694224 w_k1_IBM_s006
                                 + 0.86967 w_k1_INTC_s006
                                 + 0.739338 w_k1_KO_s006
                                 + 0.579676 w_k1_MAT_s006
                                 + 0.860552 w_k1_MSFT_s006
                                 + 1.01345 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.742101 w_k1_ADM_s006
                                 + 0.641778 w_k1_AXP_s006
                                 + 0.82346 w_k1_DIS_s006
                                 + 0.798174 w_k1_FDX_s006
                                 + 1.11386 w_k1_IBM_s006
                                 + 0.893325 w_k1_INTC_s006
                                 + 0.895854 w_k1_KO_s006
                                 + 0.69014 w_k1_MAT_s006
                                 + 0.824083 w_k1_MSFT_s006
                                 + 0.922535 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.37646 w_k1_ADM_s006
                                 + 1.42713 w_k1_AXP_s006
                                 + 1.00473 w_k1_DIS_s006
                                 + 1.22331 w_k1_FDX_s006
                                 + 0.963409 w_k1_IBM_s006
                                 + 1.05474 w_k1_INTC_s006
                                 + 1.03951 w_k1_KO_s006 + 1.13769 w_k1_MAT_s006
                                 + 0.972676 w_k1_MSFT_s006
                                 + 0.997048 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.92795 w_k1_ADM_s007
                                 + 1.0562 w_k1_AXP_s007 + 1.11863 w_k1_DIS_s007
                                 + 0.980483 w_k1_FDX_s007
                                 + 1.66539 w_k1_IBM_s007
                                 + 1.00333 w_k1_INTC_s007
                                 + 1.22602 w_k1_KO_s007 + 1.96764 w_k1_MAT_s007
                                 + 1.10839 w_k1_MSFT_s007
                                 + 1.21202 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.99836 w_k1_ADM_s007
                                 + 0.719575 w_k1_AXP_s007
                                 + 0.58738 w_k1_DIS_s007
                                 + 0.778905 w_k1_FDX_s007
                                 + 0.750474 w_k1_IBM_s007
                                 + 0.893607 w_k1_INTC_s007
                                 + 0.843903 w_k1_KO_s007
                                 + 0.610312 w_k1_MAT_s007
                                 + 0.848833 w_k1_MSFT_s007
                                 + 0.841884 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.17589 w_k1_ADM_s007
                                 + 1.49603 w_k1_AXP_s007
                                 + 1.39611 w_k1_DIS_s007
                                 + 1.34103 w_k1_FDX_s007
                                 + 0.972867 w_k1_IBM_s007
                                 + 1.26793 w_k1_INTC_s007
                                 + 1.01288 w_k1_KO_s007 + 1.25485 w_k1_MAT_s007
                                 + 1.29917 w_k1_MSFT_s007
                                 + 1.18508 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.25108 w_k1_ADM_s007
                                 + 0.910517 w_k1_AXP_s007
                                 + 1.26482 w_k1_DIS_s007
                                 + 1.16979 w_k1_FDX_s007
                                 + 1.00637 w_k1_IBM_s007 + 1.222 w_k1_INTC_s007
                                 + 1.23308 w_k1_KO_s007
                                 + 0.764039 w_k1_MAT_s007
                                 + 1.23421 w_k1_MSFT_s007
                                 + 1.23425 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.24027 w_k1_ADM_s007
                                 + 0.96127 w_k1_AXP_s007
                                 + 0.795645 w_k1_DIS_s007
                                 + 0.880955 w_k1_FDX_s007
                                 + 1.1668 w_k1_IBM_s007
                                 + 0.935526 w_k1_INTC_s007
                                 + 1.05172 w_k1_KO_s007 + 1.14075 w_k1_MAT_s007
                                 + 0.884252 w_k1_MSFT_s007
                                 + 1.08046 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.03288 w_k1_ADM_s007
                                 + 1.74356 w_k1_AXP_s007
                                 + 1.47236 w_k1_DIS_s007
                                 + 1.33298 w_k1_FDX_s007 + 1.2616 w_k1_IBM_s007
                                 + 1.507 w_k1_INTC_s007 + 1.39663 w_k1_KO_s007
                                 + 1.51348 w_k1_MAT_s007
                                 + 1.33867 w_k1_MSFT_s007
                                 + 1.21805 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.14979 w_k1_ADM_s007
                                 + 1.49007 w_k1_AXP_s007
                                 + 0.750247 w_k1_DIS_s007
                                 + 1.1133 w_k1_FDX_s007
                                 + 0.940339 w_k1_IBM_s007
                                 + 0.950745 w_k1_INTC_s007
                                 + 0.779096 w_k1_KO_s007
                                 + 1.34157 w_k1_MAT_s007
                                 + 0.917097 w_k1_MSFT_s007
                                 + 0.928831 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.9896 w_k1_ADM_s007
                                 + 0.815877 w_k1_AXP_s007
                                 + 0.97994 w_k1_DIS_s007
                                 + 1.07799 w_k1_FDX_s007
                                 + 0.81953 w_k1_IBM_s007
                                 + 1.01794 w_k1_INTC_s007
                                 + 0.853927 w_k1_KO_s007
                                 + 0.898628 w_k1_MAT_s007
                                 + 1.05425 w_k1_MSFT_s007
                                 + 0.993172 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.21814 w_k1_ADM_s010
                                 + 1.02163 w_k1_AXP_s010
                                 + 0.759029 w_k1_DIS_s010
                                 + 0.942426 w_k1_FDX_s010
                                 + 0.909636 w_k1_IBM_s010
                                 + 0.978671 w_k1_INTC_s010
                                 + 0.80362 w_k1_KO_s010
                                 + 0.750319 w_k1_MAT_s010
                                 + 0.817803 w_k1_MSFT_s010
                                 + 0.966144 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.827791 w_k1_ADM_s010
                                 + 0.685272 w_k1_AXP_s010
                                 + 0.60288 w_k1_DIS_s010
                                 + 0.810705 w_k1_FDX_s010
                                 + 0.591152 w_k1_IBM_s010
                                 + 0.740684 w_k1_INTC_s010
                                 + 0.625533 w_k1_KO_s010
                                 + 0.696268 w_k1_MAT_s010
                                 + 0.663638 w_k1_MSFT_s010
                                 + 0.895596 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.899127 w_k1_ADM_s010
                                 + 0.788364 w_k1_AXP_s010
                                 + 0.756057 w_k1_DIS_s010
                                 + 0.816345 w_k1_FDX_s010
                                 + 1.06399 w_k1_IBM_s010
                                 + 0.81247 w_k1_INTC_s010
                                 + 0.95329 w_k1_KO_s010 + 1.06326 w_k1_MAT_s010
                                 + 0.793945 w_k1_MSFT_s010
                                 + 1.04472 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.62442 w_k1_ADM_s010
                                 + 1.38556 w_k1_AXP_s010
                                 + 1.82538 w_k1_DIS_s010
                                 + 1.37986 w_k1_FDX_s010
                                 + 1.47744 w_k1_IBM_s010
                                 + 1.29133 w_k1_INTC_s010 + 1.652 w_k1_KO_s010
                                 + 1.14809 w_k1_MAT_s010
                                 + 1.27806 w_k1_MSFT_s010 + 1.5007 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.864376 w_k1_ADM_s010
                                 + 1.00861 w_k1_AXP_s010
                                 + 1.04752 w_k1_DIS_s010
                                 + 0.949245 w_k1_FDX_s010
                                 + 1.46355 w_k1_IBM_s010
                                 + 1.29283 w_k1_INTC_s010 + 1.2899 w_k1_KO_s010
                                 + 0.977484 w_k1_MAT_s010
                                 + 1.01208 w_k1_MSFT_s010 + 1.0483 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.11751 w_k1_ADM_s010
                                 + 0.874007 w_k1_AXP_s010
                                 + 0.872369 w_k1_DIS_s010
                                 + 1.07745 w_k1_FDX_s010
                                 + 0.654506 w_k1_IBM_s010
                                 + 1.14182 w_k1_INTC_s010
                                 + 0.790459 w_k1_KO_s010
                                 + 0.477786 w_k1_MAT_s010
                                 + 1.04342 w_k1_MSFT_s010
                                 + 0.974158 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.29252 w_k1_ADM_s010
                                 + 1.26202 w_k1_AXP_s010
                                 + 1.18606 w_k1_DIS_s010 + 1.1863 w_k1_FDX_s010
                                 + 1.25676 w_k1_IBM_s010
                                 + 1.07999 w_k1_INTC_s010
                                 + 1.23475 w_k1_KO_s010 + 1.34808 w_k1_MAT_s010
                                 + 1.1933 w_k1_MSFT_s010 + 1.09678 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.02234 w_k1_ADM_s010
                                 + 0.674011 w_k1_AXP_s010
                                 + 0.992234 w_k1_DIS_s010
                                 + 0.873673 w_k1_FDX_s010
                                 + 0.95438 w_k1_IBM_s010
                                 + 1.16477 w_k1_INTC_s010
                                 + 1.05799 w_k1_KO_s010 + 0.58706 w_k1_MAT_s010
                                 + 1.05353 w_k1_MSFT_s010
                                 + 1.02149 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.969777 w_k1_ADM_s011
                                 + 1.09457 w_k1_AXP_s011
                                 + 0.946066 w_k1_DIS_s011
                                 + 1.15545 w_k1_FDX_s011
                                 + 0.782057 w_k1_IBM_s011
                                 + 1.18795 w_k1_INTC_s011
                                 + 0.904163 w_k1_KO_s011
                                 + 0.842107 w_k1_MAT_s011
                                 + 0.955693 w_k1_MSFT_s011
                                 + 1.17744 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.82992 w_k1_ADM_s011
                                 + 0.834308 w_k1_AXP_s011
                                 + 0.806352 w_k1_DIS_s011
                                 + 0.866175 w_k1_FDX_s011
                                 + 0.787824 w_k1_IBM_s011
                                 + 0.850633 w_k1_INTC_s011
                                 + 0.899801 w_k1_KO_s011
                                 + 0.655556 w_k1_MAT_s011
                                 + 0.866573 w_k1_MSFT_s011
                                 + 0.941398 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.784876 w_k1_ADM_s011
                                 + 0.806347 w_k1_AXP_s011
                                 + 0.905292 w_k1_DIS_s011
                                 + 0.92686 w_k1_FDX_s011
                                 + 1.15509 w_k1_IBM_s011
                                 + 0.997945 w_k1_INTC_s011
                                 + 0.837971 w_k1_KO_s011
                                 + 1.13689 w_k1_MAT_s011
                                 + 0.960004 w_k1_MSFT_s011
                                 + 1.01809 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.17964 w_k1_ADM_s011
                                 + 1.13547 w_k1_AXP_s011
                                 + 1.31723 w_k1_DIS_s011
                                 + 1.13497 w_k1_FDX_s011
                                 + 1.72422 w_k1_IBM_s011
                                 + 1.0547 w_k1_INTC_s011 + 1.38344 w_k1_KO_s011
                                 + 1.827 w_k1_MAT_s011
                                 + 0.952657 w_k1_MSFT_s011
                                 + 1.22417 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.686199 w_k1_ADM_s011
                                 + 0.613426 w_k1_AXP_s011
                                 + 0.5746 w_k1_DIS_s011
                                 + 0.733618 w_k1_FDX_s011
                                 + 0.782424 w_k1_IBM_s011
                                 + 1.00957 w_k1_INTC_s011
                                 + 0.698376 w_k1_KO_s011
                                 + 0.537394 w_k1_MAT_s011
                                 + 0.798799 w_k1_MSFT_s011
                                 + 0.757691 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.05461 w_k1_ADM_s011
                                 + 1.15986 w_k1_AXP_s011
                                 + 0.813269 w_k1_DIS_s011
                                 + 0.986364 w_k1_FDX_s011
                                 + 1.03482 w_k1_IBM_s011
                                 + 1.19929 w_k1_INTC_s011
                                 + 0.974692 w_k1_KO_s011
                                 + 1.01222 w_k1_MAT_s011
                                 + 0.975516 w_k1_MSFT_s011
                                 + 1.04139 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.968701 w_k1_ADM_s011
                                 + 1.15276 w_k1_AXP_s011
                                 + 1.01526 w_k1_DIS_s011
                                 + 1.18886 w_k1_FDX_s011
                                 + 1.04189 w_k1_IBM_s011
                                 + 1.08634 w_k1_INTC_s011
                                 + 1.10304 w_k1_KO_s011 + 1.39321 w_k1_MAT_s011
                                 + 1.03093 w_k1_MSFT_s011 + 1.0262 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.11462 w_k1_ADM_s011
                                 + 1.42997 w_k1_AXP_s011
                                 + 1.53519 w_k1_DIS_s011
                                 + 1.42615 w_k1_FDX_s011
                                 + 1.18452 w_k1_IBM_s011 + 1.538 w_k1_INTC_s011
                                 + 1.17294 w_k1_KO_s011 + 1.00041 w_k1_MAT_s011
                                 + 1.38773 w_k1_MSFT_s011
                                 + 1.30034 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.07433 w_k1_ADM_s012
                                 + 1.1998 w_k1_AXP_s012 + 1.17961 w_k1_DIS_s012
                                 + 1.11808 w_k1_FDX_s012
                                 + 0.973141 w_k1_IBM_s012
                                 + 1.00458 w_k1_INTC_s012
                                 + 1.08964 w_k1_KO_s012 + 1.40588 w_k1_MAT_s012
                                 + 1.18839 w_k1_MSFT_s012
                                 + 0.988282 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.00978 w_k1_ADM_s012
                                 + 0.985208 w_k1_AXP_s012
                                 + 1.02491 w_k1_DIS_s012
                                 + 1.00885 w_k1_FDX_s012
                                 + 1.27864 w_k1_IBM_s012
                                 + 0.861355 w_k1_INTC_s012
                                 + 1.09859 w_k1_KO_s012 + 1.59696 w_k1_MAT_s012
                                 + 1.07748 w_k1_MSFT_s012
                                 + 1.01333 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.09071 w_k1_ADM_s012
                                 + 1.56255 w_k1_AXP_s012
                                 + 1.89648 w_k1_DIS_s012
                                 + 1.39019 w_k1_FDX_s012
                                 + 1.75026 w_k1_IBM_s012
                                 + 1.4277 w_k1_INTC_s012 + 1.92474 w_k1_KO_s012
                                 + 1.84143 w_k1_MAT_s012
                                 + 1.63005 w_k1_MSFT_s012
                                 + 1.27003 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.871978 w_k1_ADM_s012
                                 + 1.22779 w_k1_AXP_s012
                                 + 1.19041 w_k1_DIS_s012 + 1.0343 w_k1_FDX_s012
                                 + 1.77222 w_k1_IBM_s012
                                 + 0.844769 w_k1_INTC_s012
                                 + 1.39008 w_k1_KO_s012 + 2.37328 w_k1_MAT_s012
                                 + 1.18964 w_k1_MSFT_s012
                                 + 1.12198 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.706096 w_k1_ADM_s012
                                 + 1.20421 w_k1_AXP_s012
                                 + 0.763842 w_k1_DIS_s012
                                 + 0.859787 w_k1_FDX_s012
                                 + 1.12761 w_k1_IBM_s012
                                 + 1.03343 w_k1_INTC_s012
                                 + 1.04582 w_k1_KO_s012 + 1.64137 w_k1_MAT_s012
                                 + 0.934599 w_k1_MSFT_s012
                                 + 0.913453 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.25844 w_k1_ADM_s012
                                 + 1.49541 w_k1_AXP_s012
                                 + 1.45611 w_k1_DIS_s012 + 1.4007 w_k1_FDX_s012
                                 + 1.0034 w_k1_IBM_s012 + 1.2107 w_k1_INTC_s012
                                 + 1.14156 w_k1_KO_s012 + 1.54545 w_k1_MAT_s012
                                 + 1.48324 w_k1_MSFT_s012
                                 + 0.975207 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.678044 w_k1_ADM_s012
                                 + 0.751798 w_k1_AXP_s012
                                 + 0.636523 w_k1_DIS_s012
                                 + 0.78179 w_k1_FDX_s012
                                 + 0.685487 w_k1_IBM_s012
                                 + 0.891061 w_k1_INTC_s012
                                 + 0.676874 w_k1_KO_s012
                                 + 0.707428 w_k1_MAT_s012
                                 + 0.811246 w_k1_MSFT_s012
                                 + 0.678927 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.07555 w_k1_ADM_s012
                                 + 0.852819 w_k1_AXP_s012
                                 + 0.820379 w_k1_DIS_s012
                                 + 1.04834 w_k1_FDX_s012
                                 + 0.676695 w_k1_IBM_s012
                                 + 0.996048 w_k1_INTC_s012
                                 + 0.797795 w_k1_KO_s012
                                 + 0.665378 w_k1_MAT_s012
                                 + 1.08758 w_k1_MSFT_s012
                                 + 0.872156 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.13267 w_k1_ADM_s013
                                 + 0.853849 w_k1_AXP_s013
                                 + 0.93618 w_k1_DIS_s013
                                 + 0.909841 w_k1_FDX_s013
                                 + 0.920511 w_k1_IBM_s013
                                 + 1.01809 w_k1_INTC_s013
                                 + 0.868908 w_k1_KO_s013
                                 + 0.752321 w_k1_MAT_s013
                                 + 0.959141 w_k1_MSFT_s013
                                 + 0.883041 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.66573 w_k1_ADM_s013
                                 + 1.4513 w_k1_AXP_s013 + 1.43071 w_k1_DIS_s013
                                 + 1.41326 w_k1_FDX_s013
                                 + 1.23781 w_k1_IBM_s013
                                 + 1.00937 w_k1_INTC_s013
                                 + 1.22729 w_k1_KO_s013 + 1.76079 w_k1_MAT_s013
                                 + 1.20905 w_k1_MSFT_s013
                                 + 1.04016 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.8107 w_k1_ADM_s013
                                 + 0.530254 w_k1_AXP_s013
                                 + 0.622288 w_k1_DIS_s013
                                 + 0.727114 w_k1_FDX_s013
                                 + 0.629389 w_k1_IBM_s013
                                 + 0.680622 w_k1_INTC_s013
                                 + 0.622737 w_k1_KO_s013
                                 + 0.658668 w_k1_MAT_s013
                                 + 0.767954 w_k1_MSFT_s013
                                 + 0.737218 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.923206 w_k1_ADM_s013
                                 + 0.714938 w_k1_AXP_s013
                                 + 0.671631 w_k1_DIS_s013
                                 + 0.812207 w_k1_FDX_s013
                                 + 0.770877 w_k1_IBM_s013
                                 + 0.850164 w_k1_INTC_s013
                                 + 0.740278 w_k1_KO_s013
                                 + 0.594903 w_k1_MAT_s013
                                 + 0.755428 w_k1_MSFT_s013
                                 + 0.819804 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.2309 w_k1_ADM_s013
                                 + 1.0425 w_k1_AXP_s013
                                 + 0.824845 w_k1_DIS_s013 + 1.007 w_k1_FDX_s013
                                 + 0.995071 w_k1_IBM_s013
                                 + 0.753083 w_k1_INTC_s013
                                 + 0.854618 w_k1_KO_s013
                                 + 1.44698 w_k1_MAT_s013
                                 + 0.941883 w_k1_MSFT_s013
                                 + 0.926033 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.26493 w_k1_ADM_s013
                                 + 1.24841 w_k1_AXP_s013
                                 + 1.17084 w_k1_DIS_s013
                                 + 1.02368 w_k1_FDX_s013
                                 + 1.47728 w_k1_IBM_s013
                                 + 0.848598 w_k1_INTC_s013
                                 + 1.27928 w_k1_KO_s013 + 2.01286 w_k1_MAT_s013
                                 + 1.11924 w_k1_MSFT_s013
                                 + 1.12551 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.22747 w_k1_ADM_s013
                                 + 0.926497 w_k1_AXP_s013 + 1.242 w_k1_DIS_s013
                                 + 1.16998 w_k1_FDX_s013
                                 + 0.864464 w_k1_IBM_s013
                                 + 1.06438 w_k1_INTC_s013
                                 + 0.967352 w_k1_KO_s013
                                 + 0.746298 w_k1_MAT_s013
                                 + 1.15127 w_k1_MSFT_s013
                                 + 1.13105 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.28842 w_k1_ADM_s013
                                 + 0.998115 w_k1_AXP_s013
                                 + 1.42126 w_k1_DIS_s013
                                 + 0.984166 w_k1_FDX_s013
                                 + 1.23535 w_k1_IBM_s013
                                 + 0.91545 w_k1_INTC_s013
                                 + 1.30791 w_k1_KO_s013 + 1.00262 w_k1_MAT_s013
                                 + 1.21159 w_k1_MSFT_s013
                                 + 0.980018 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.864125 w_k1_ADM_s014
                                 + 0.997458 w_k1_AXP_s014
                                 + 1.13714 w_k1_DIS_s014
                                 + 0.872692 w_k1_FDX_s014
                                 + 1.73214 w_k1_IBM_s014
                                 + 0.761757 w_k1_INTC_s014
                                 + 1.10553 w_k1_KO_s014 + 1.19021 w_k1_MAT_s014
                                 + 0.958368 w_k1_MSFT_s014
                                 + 0.90857 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.77863 w_k1_ADM_s014
                                 + 0.790405 w_k1_AXP_s014
                                 + 0.64081 w_k1_DIS_s014
                                 + 0.704928 w_k1_FDX_s014
                                 + 1.1595 w_k1_IBM_s014
                                 + 0.651274 w_k1_INTC_s014
                                 + 0.847128 w_k1_KO_s014
                                 + 1.34731 w_k1_MAT_s014
                                 + 0.721373 w_k1_MSFT_s014
                                 + 0.891649 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.20911 w_k1_ADM_s014
                                 + 1.3644 w_k1_AXP_s014 + 1.5634 w_k1_DIS_s014
                                 + 1.17582 w_k1_FDX_s014
                                 + 1.26287 w_k1_IBM_s014
                                 + 0.888692 w_k1_INTC_s014
                                 + 1.08487 w_k1_KO_s014 + 1.26325 w_k1_MAT_s014
                                 + 0.983402 w_k1_MSFT_s014
                                 + 1.17356 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.925139 w_k1_ADM_s014
                                 + 0.643407 w_k1_AXP_s014
                                 + 0.67448 w_k1_DIS_s014
                                 + 0.814338 w_k1_FDX_s014
                                 + 0.712291 w_k1_IBM_s014
                                 + 0.707087 w_k1_INTC_s014
                                 + 0.636029 w_k1_KO_s014
                                 + 0.431117 w_k1_MAT_s014
                                 + 0.815287 w_k1_MSFT_s014
                                 + 0.791069 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.2962 w_k1_ADM_s014
                                 + 1.2499 w_k1_AXP_s014 + 1.41082 w_k1_DIS_s014
                                 + 1.39032 w_k1_FDX_s014
                                 + 0.892977 w_k1_IBM_s014
                                 + 0.982378 w_k1_INTC_s014
                                 + 0.902964 w_k1_KO_s014
                                 + 0.972924 w_k1_MAT_s014
                                 + 1.17643 w_k1_MSFT_s014
                                 + 1.11792 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.852065 w_k1_ADM_s014
                                 + 1.1501 w_k1_AXP_s014
                                 + 0.895546 w_k1_DIS_s014
                                 + 0.957948 w_k1_FDX_s014
                                 + 1.25684 w_k1_IBM_s014
                                 + 0.924333 w_k1_INTC_s014
                                 + 0.897951 w_k1_KO_s014
                                 + 1.24337 w_k1_MAT_s014 + 0.978 w_k1_MSFT_s014
                                 + 0.995138 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.08257 w_k1_ADM_s014
                                 + 0.777706 w_k1_AXP_s014
                                 + 0.970378 w_k1_DIS_s014
                                 + 0.970974 w_k1_FDX_s014
                                 + 0.755895 w_k1_IBM_s014
                                 + 0.908151 w_k1_INTC_s014
                                 + 0.78445 w_k1_KO_s014
                                 + 0.496787 w_k1_MAT_s014
                                 + 0.955955 w_k1_MSFT_s014
                                 + 0.867455 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.898752 w_k1_ADM_s014
                                 + 1.00997 w_k1_AXP_s014
                                 + 0.920138 w_k1_DIS_s014
                                 + 1.01814 w_k1_FDX_s014
                                 + 0.873812 w_k1_IBM_s014
                                 + 1.09136 w_k1_INTC_s014
                                 + 0.839875 w_k1_KO_s014
                                 + 0.724958 w_k1_MAT_s014
                                 + 1.13466 w_k1_MSFT_s014
                                 + 0.989347 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.31779 w_k1_ADM_s015
                                 + 1.28896 w_k1_AXP_s015
                                 + 1.75517 w_k1_DIS_s015
                                 + 1.28282 w_k1_FDX_s015
                                 + 0.945801 w_k1_IBM_s015
                                 + 1.30978 w_k1_INTC_s015
                                 + 1.69407 w_k1_KO_s015 + 1.05641 w_k1_MAT_s015
                                 + 1.37572 w_k1_MSFT_s015
                                 + 1.06557 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.255 w_k1_ADM_s015
                                 + 1.27375 w_k1_AXP_s015
                                 + 0.951774 w_k1_DIS_s015
                                 + 0.987055 w_k1_FDX_s015
                                 + 0.984307 w_k1_IBM_s015
                                 + 0.973656 w_k1_INTC_s015
                                 + 1.10525 w_k1_KO_s015 + 1.41889 w_k1_MAT_s015
                                 + 0.999746 w_k1_MSFT_s015
                                 + 0.950032 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.93339 w_k1_ADM_s015
                                 + 0.798343 w_k1_AXP_s015
                                 + 0.703052 w_k1_DIS_s015
                                 + 0.767482 w_k1_FDX_s015
                                 + 0.888571 w_k1_IBM_s015
                                 + 0.768123 w_k1_INTC_s015
                                 + 0.900576 w_k1_KO_s015
                                 + 0.97475 w_k1_MAT_s015
                                 + 0.870503 w_k1_MSFT_s015
                                 + 0.73322 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.984566 w_k1_ADM_s015
                                 + 0.923361 w_k1_AXP_s015
                                 + 0.914554 w_k1_DIS_s015
                                 + 0.877201 w_k1_FDX_s015
                                 + 1.51174 w_k1_IBM_s015
                                 + 1.03315 w_k1_INTC_s015
                                 + 1.29569 w_k1_KO_s015 + 1.4055 w_k1_MAT_s015
                                 + 0.941263 w_k1_MSFT_s015
                                 + 0.948709 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.2171 w_k1_ADM_s015
                                 + 1.43533 w_k1_AXP_s015
                                 + 1.31916 w_k1_DIS_s015
                                 + 1.24997 w_k1_FDX_s015
                                 + 0.997558 w_k1_IBM_s015
                                 + 1.3506 w_k1_INTC_s015 + 1.25057 w_k1_KO_s015
                                 + 1.06715 w_k1_MAT_s015
                                 + 1.23166 w_k1_MSFT_s015
                                 + 1.04335 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.760652 w_k1_ADM_s015
                                 + 0.921577 w_k1_AXP_s015
                                 + 0.950111 w_k1_DIS_s015
                                 + 0.968868 w_k1_FDX_s015
                                 + 0.748882 w_k1_IBM_s015
                                 + 1.11777 w_k1_INTC_s015
                                 + 0.995122 w_k1_KO_s015
                                 + 1.02545 w_k1_MAT_s015
                                 + 1.01644 w_k1_MSFT_s015
                                 + 0.767044 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.01479 w_k1_ADM_s015
                                 + 1.26935 w_k1_AXP_s015
                                 + 1.34108 w_k1_DIS_s015
                                 + 1.15558 w_k1_FDX_s015
                                 + 1.18579 w_k1_IBM_s015
                                 + 1.19078 w_k1_INTC_s015
                                 + 1.29365 w_k1_KO_s015 + 1.26099 w_k1_MAT_s015
                                 + 1.24996 w_k1_MSFT_s015
                                 + 1.06352 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.33927 w_k1_ADM_s015
                                 + 1.17063 w_k1_AXP_s015
                                 + 1.29754 w_k1_DIS_s015
                                 + 1.23646 w_k1_FDX_s015
                                 + 0.647902 w_k1_IBM_s015
                                 + 1.15279 w_k1_INTC_s015
                                 + 0.95477 w_k1_KO_s015
                                 + 0.709828 w_k1_MAT_s015
                                 + 1.12357 w_k1_MSFT_s015
                                 + 1.06877 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.05462 w_k1_ADM_s016
                                 + 1.04195 w_k1_AXP_s016
                                 + 0.972775 w_k1_DIS_s016
                                 + 0.878964 w_k1_FDX_s016
                                 + 1.74555 w_k1_IBM_s016
                                 + 1.0873 w_k1_INTC_s016 + 1.22687 w_k1_KO_s016
                                 + 1.16792 w_k1_MAT_s016
                                 + 0.943168 w_k1_MSFT_s016
                                 + 0.97751 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.33274 w_k1_ADM_s016
                                 + 1.19433 w_k1_AXP_s016
                                 + 1.43487 w_k1_DIS_s016
                                 + 1.13916 w_k1_FDX_s016
                                 + 1.44915 w_k1_IBM_s016
                                 + 1.10311 w_k1_INTC_s016
                                 + 1.15651 w_k1_KO_s016 + 1.12488 w_k1_MAT_s016
                                 + 1.19601 w_k1_MSFT_s016
                                 + 1.14752 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.20865 w_k1_ADM_s016
                                 + 1.18395 w_k1_AXP_s016
                                 + 1.30749 w_k1_DIS_s016
                                 + 1.16082 w_k1_FDX_s016
                                 + 0.899067 w_k1_IBM_s016
                                 + 0.970943 w_k1_INTC_s016
                                 + 1.04629 w_k1_KO_s016 + 1.00768 w_k1_MAT_s016
                                 + 1.18926 w_k1_MSFT_s016
                                 + 1.16235 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.20934 w_k1_ADM_s016
                                 + 1.11583 w_k1_AXP_s016 + 1.5961 w_k1_DIS_s016
                                 + 1.22354 w_k1_FDX_s016
                                 + 1.16061 w_k1_IBM_s016
                                 + 1.40822 w_k1_INTC_s016
                                 + 1.14621 w_k1_KO_s016
                                 + 0.775212 w_k1_MAT_s016
                                 + 1.24372 w_k1_MSFT_s016
                                 + 1.20739 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.962011 w_k1_ADM_s016
                                 + 0.800706 w_k1_AXP_s016
                                 + 0.987247 w_k1_DIS_s016
                                 + 0.858595 w_k1_FDX_s016
                                 + 0.995529 w_k1_IBM_s016
                                 + 0.987771 w_k1_INTC_s016
                                 + 0.934577 w_k1_KO_s016
                                 + 0.790966 w_k1_MAT_s016
                                 + 0.948719 w_k1_MSFT_s016
                                 + 0.923323 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.41479 w_k1_ADM_s016
                                 + 1.01105 w_k1_AXP_s016
                                 + 1.09497 w_k1_DIS_s016
                                 + 1.00949 w_k1_FDX_s016
                                 + 0.796532 w_k1_IBM_s016
                                 + 0.971661 w_k1_INTC_s016
                                 + 0.994067 w_k1_KO_s016
                                 + 0.76667 w_k1_MAT_s016
                                 + 1.07085 w_k1_MSFT_s016
                                 + 1.03252 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.00589 w_k1_ADM_s016
                                 + 0.683846 w_k1_AXP_s016
                                 + 0.653829 w_k1_DIS_s016
                                 + 0.808743 w_k1_FDX_s016
                                 + 0.557386 w_k1_IBM_s016
                                 + 0.844528 w_k1_INTC_s016
                                 + 0.609104 w_k1_KO_s016
                                 + 0.448586 w_k1_MAT_s016
                                 + 0.817177 w_k1_MSFT_s016
                                 + 0.920784 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.00215 w_k1_ADM_s016
                                 + 0.879109 w_k1_AXP_s016
                                 + 0.890672 w_k1_DIS_s016
                                 + 1.10001 w_k1_FDX_s016
                                 + 0.615419 w_k1_IBM_s016
                                 + 1.14528 w_k1_INTC_s016
                                 + 0.781099 w_k1_KO_s016
                                 + 0.596625 w_k1_MAT_s016
                                 + 1.08716 w_k1_MSFT_s016
                                 + 0.926831 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.757977 w_k1_ADM_s017
                                 + 0.61336 w_k1_AXP_s017
                                 + 0.52427 w_k1_DIS_s017
                                 + 0.728224 w_k1_FDX_s017
                                 + 0.600307 w_k1_IBM_s017
                                 + 0.852617 w_k1_INTC_s017
                                 + 0.65316 w_k1_KO_s017
                                 + 0.844022 w_k1_MAT_s017
                                 + 0.7554 w_k1_MSFT_s017
                                 + 0.829318 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.951785 w_k1_ADM_s017
                                 + 1.2734 w_k1_AXP_s017 + 1.38695 w_k1_DIS_s017
                                 + 1.10703 w_k1_FDX_s017 + 1.6243 w_k1_IBM_s017
                                 + 1.34334 w_k1_INTC_s017 + 1.3362 w_k1_KO_s017
                                 + 1.62766 w_k1_MAT_s017
                                 + 1.28995 w_k1_MSFT_s017
                                 + 1.25307 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.29116 w_k1_ADM_s017
                                 + 1.11636 w_k1_AXP_s017
                                 + 0.938295 w_k1_DIS_s017
                                 + 1.09294 w_k1_FDX_s017
                                 + 0.695148 w_k1_IBM_s017
                                 + 0.987506 w_k1_INTC_s017
                                 + 0.757106 w_k1_KO_s017
                                 + 0.678888 w_k1_MAT_s017
                                 + 0.940813 w_k1_MSFT_s017
                                 + 0.950621 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.13063 w_k1_ADM_s017
                                 + 1.27111 w_k1_AXP_s017
                                 + 1.40891 w_k1_DIS_s017
                                 + 1.22577 w_k1_FDX_s017
                                 + 1.23898 w_k1_IBM_s017
                                 + 1.02958 w_k1_INTC_s017
                                 + 1.12278 w_k1_KO_s017 + 1.41486 w_k1_MAT_s017
                                 + 1.14514 w_k1_MSFT_s017
                                 + 1.21062 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.19107 w_k1_ADM_s017
                                 + 1.19742 w_k1_AXP_s017
                                 + 0.935615 w_k1_DIS_s017
                                 + 1.05889 w_k1_FDX_s017
                                 + 1.08887 w_k1_IBM_s017
                                 + 1.20232 w_k1_INTC_s017
                                 + 0.934902 w_k1_KO_s017
                                 + 1.01122 w_k1_MAT_s017
                                 + 1.00917 w_k1_MSFT_s017
                                 + 1.00147 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.995366 w_k1_ADM_s017
                                 + 0.948052 w_k1_AXP_s017
                                 + 0.762296 w_k1_DIS_s017
                                 + 0.835737 w_k1_FDX_s017
                                 + 0.9729 w_k1_IBM_s017
                                 + 1.08179 w_k1_INTC_s017
                                 + 0.893037 w_k1_KO_s017
                                 + 1.18498 w_k1_MAT_s017
                                 + 0.963635 w_k1_MSFT_s017
                                 + 0.946702 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.07783 w_k1_ADM_s017
                                 + 0.836213 w_k1_AXP_s017
                                 + 0.608964 w_k1_DIS_s017
                                 + 0.832826 w_k1_FDX_s017
                                 + 0.57752 w_k1_IBM_s017
                                 + 0.918785 w_k1_INTC_s017
                                 + 0.59207 w_k1_KO_s017
                                 + 0.576344 w_k1_MAT_s017
                                 + 0.803466 w_k1_MSFT_s017
                                 + 0.925119 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.975268 w_k1_ADM_s017
                                 + 0.834379 w_k1_AXP_s017
                                 + 0.899568 w_k1_DIS_s017
                                 + 0.785253 w_k1_FDX_s017
                                 + 1.27725 w_k1_IBM_s017
                                 + 0.981462 w_k1_INTC_s017
                                 + 0.99522 w_k1_KO_s017
                                 + 0.994789 w_k1_MAT_s017
                                 + 0.83566 w_k1_MSFT_s017
                                 + 1.03754 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.572387 w_k1_ADM_s100
                                 + 0.708199 w_k1_AXP_s100
                                 + 0.693321 w_k1_DIS_s100
                                 + 0.738348 w_k1_FDX_s100
                                 + 0.963462 w_k1_IBM_s100
                                 + 0.894759 w_k1_INTC_s100
                                 + 0.85489 w_k1_KO_s100
                                 + 0.691092 w_k1_MAT_s100
                                 + 0.838827 w_k1_MSFT_s100
                                 + 0.85105 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.09425 w_k1_ADM_s100
                                 + 1.30546 w_k1_AXP_s100
                                 + 1.07499 w_k1_DIS_s100
                                 + 1.15132 w_k1_FDX_s100 + 0.9847 w_k1_IBM_s100
                                 + 0.998977 w_k1_INTC_s100
                                 + 0.991322 w_k1_KO_s100
                                 + 0.976994 w_k1_MAT_s100
                                 + 1.05647 w_k1_MSFT_s100
                                 + 1.00153 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.999826 w_k1_ADM_s100
                                 + 0.931017 w_k1_AXP_s100
                                 + 1.09907 w_k1_DIS_s100
                                 + 0.988514 w_k1_FDX_s100
                                 + 1.15533 w_k1_IBM_s100
                                 + 1.05719 w_k1_INTC_s100
                                 + 1.09644 w_k1_KO_s100
                                 + 0.975519 w_k1_MAT_s100
                                 + 1.06022 w_k1_MSFT_s100
                                 + 1.05274 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.857639 w_k1_ADM_s100
                                 + 0.872482 w_k1_AXP_s100
                                 + 0.788752 w_k1_DIS_s100
                                 + 0.84838 w_k1_FDX_s100
                                 + 1.32882 w_k1_IBM_s100
                                 + 0.922643 w_k1_INTC_s100
                                 + 0.970589 w_k1_KO_s100
                                 + 1.04305 w_k1_MAT_s100
                                 + 0.828126 w_k1_MSFT_s100
                                 + 0.940116 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.975606 w_k1_ADM_s100
                                 + 1.34551 w_k1_AXP_s100
                                 + 1.49012 w_k1_DIS_s100
                                 + 1.16236 w_k1_FDX_s100
                                 + 1.51274 w_k1_IBM_s100
                                 + 1.15611 w_k1_INTC_s100
                                 + 1.29441 w_k1_KO_s100 + 1.5745 w_k1_MAT_s100
                                 + 1.19729 w_k1_MSFT_s100
                                 + 1.23262 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.26414 w_k1_ADM_s100
                                 + 1.14518 w_k1_AXP_s100
                                 + 1.48563 w_k1_DIS_s100
                                 + 1.32193 w_k1_FDX_s100
                                 + 0.895395 w_k1_IBM_s100
                                 + 1.17107 w_k1_INTC_s100
                                 + 1.19471 w_k1_KO_s100
                                 + 0.747134 w_k1_MAT_s100
                                 + 1.19427 w_k1_MSFT_s100
                                 + 1.18388 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.33126 w_k1_ADM_s100
                                 + 1.77672 w_k1_AXP_s100
                                 + 1.90158 w_k1_DIS_s100
                                 + 1.53644 w_k1_FDX_s100
                                 + 1.20223 w_k1_IBM_s100
                                 + 1.39683 w_k1_INTC_s100
                                 + 1.48288 w_k1_KO_s100 + 1.484 w_k1_MAT_s100
                                 + 1.40946 w_k1_MSFT_s100
                                 + 1.32545 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.800028 w_k1_ADM_s100
                                 + 0.813966 w_k1_AXP_s100
                                 + 0.827988 w_k1_DIS_s100
                                 + 0.938715 w_k1_FDX_s100
                                 + 0.740548 w_k1_IBM_s100
                                 + 0.880027 w_k1_INTC_s100
                                 + 0.845384 w_k1_KO_s100
                                 + 0.575055 w_k1_MAT_s100
                                 + 0.963641 w_k1_MSFT_s100
                                 + 0.889537 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.1422 w_k1_ADM_s101
                                 + 1.38271 w_k1_AXP_s101
                                 + 1.37476 w_k1_DIS_s101
                                 + 1.36313 w_k1_FDX_s101
                                 + 0.762465 w_k1_IBM_s101
                                 + 1.19369 w_k1_INTC_s101
                                 + 0.958444 w_k1_KO_s101
                                 + 1.10863 w_k1_MAT_s101
                                 + 1.36145 w_k1_MSFT_s101
                                 + 1.12082 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.828112 w_k1_ADM_s101
                                 + 1.05346 w_k1_AXP_s101
                                 + 1.11619 w_k1_DIS_s101
                                 + 0.978321 w_k1_FDX_s101
                                 + 1.20026 w_k1_IBM_s101
                                 + 0.879154 w_k1_INTC_s101
                                 + 1.15146 w_k1_KO_s101 + 1.86001 w_k1_MAT_s101
                                 + 1.04742 w_k1_MSFT_s101
                                 + 1.01593 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.704056 w_k1_ADM_s101
                                 + 0.740412 w_k1_AXP_s101
                                 + 0.848802 w_k1_DIS_s101
                                 + 0.756173 w_k1_FDX_s101
                                 + 0.972572 w_k1_IBM_s101
                                 + 0.814887 w_k1_INTC_s101
                                 + 1.03235 w_k1_KO_s101 + 1.20614 w_k1_MAT_s101
                                 + 0.930011 w_k1_MSFT_s101
                                 + 0.845337 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.19278 w_k1_ADM_s101
                                 + 1.55422 w_k1_AXP_s101
                                 + 1.71719 w_k1_DIS_s101
                                 + 1.22361 w_k1_FDX_s101
                                 + 1.46114 w_k1_IBM_s101
                                 + 1.06577 w_k1_INTC_s101
                                 + 1.53586 w_k1_KO_s101 + 1.64888 w_k1_MAT_s101
                                 + 1.31404 w_k1_MSFT_s101
                                 + 1.26691 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.803401 w_k1_ADM_s101
                                 + 1.17194 w_k1_AXP_s101
                                 + 0.963814 w_k1_DIS_s101
                                 + 1.05787 w_k1_FDX_s101
                                 + 0.977783 w_k1_IBM_s101
                                 + 1.02776 w_k1_INTC_s101
                                 + 0.896832 w_k1_KO_s101
                                 + 1.30625 w_k1_MAT_s101
                                 + 1.00701 w_k1_MSFT_s101
                                 + 1.02633 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.874787 w_k1_ADM_s101
                                 + 0.83739 w_k1_AXP_s101
                                 + 1.25532 w_k1_DIS_s101 + 1.0529 w_k1_FDX_s101
                                 + 0.775439 w_k1_IBM_s101
                                 + 0.976163 w_k1_INTC_s101
                                 + 1.13832 w_k1_KO_s101
                                 + 0.953531 w_k1_MAT_s101
                                 + 1.25622 w_k1_MSFT_s101
                                 + 1.04483 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.836386 w_k1_ADM_s101
                                 + 0.744783 w_k1_AXP_s101
                                 + 0.851681 w_k1_DIS_s101
                                 + 0.862213 w_k1_FDX_s101
                                 + 0.762338 w_k1_IBM_s101
                                 + 0.7663 w_k1_INTC_s101
                                 + 0.788159 w_k1_KO_s101
                                 + 0.783812 w_k1_MAT_s101
                                 + 0.911322 w_k1_MSFT_s101
                                 + 0.836689 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.875391 w_k1_ADM_s101
                                 + 1.03055 w_k1_AXP_s101
                                 + 0.943718 w_k1_DIS_s101
                                 + 0.88391 w_k1_FDX_s101
                                 + 1.54346 w_k1_IBM_s101
                                 + 0.992729 w_k1_INTC_s101
                                 + 1.05714 w_k1_KO_s101 + 1.48349 w_k1_MAT_s101
                                 + 0.979639 w_k1_MSFT_s101
                                 + 1.10215 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.14481 w_k1_ADM_s102
                                 + 1.47641 w_k1_AXP_s102
                                 + 1.62958 w_k1_DIS_s102
                                 + 1.50841 w_k1_FDX_s102
                                 + 1.08226 w_k1_IBM_s102
                                 + 1.43913 w_k1_INTC_s102
                                 + 1.31522 w_k1_KO_s102 + 1.39624 w_k1_MAT_s102
                                 + 1.31296 w_k1_MSFT_s102
                                 + 1.17684 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.732968 w_k1_ADM_s102
                                 + 0.567331 w_k1_AXP_s102
                                 + 0.658876 w_k1_DIS_s102
                                 + 0.660818 w_k1_FDX_s102
                                 + 0.942348 w_k1_IBM_s102
                                 + 0.792568 w_k1_INTC_s102
                                 + 0.852546 w_k1_KO_s102
                                 + 0.832886 w_k1_MAT_s102
                                 + 0.801266 w_k1_MSFT_s102
                                 + 0.750049 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.913685 w_k1_ADM_s102
                                 + 1.09504 w_k1_AXP_s102
                                 + 0.708082 w_k1_DIS_s102
                                 + 0.84693 w_k1_FDX_s102
                                 + 0.924182 w_k1_IBM_s102
                                 + 0.882302 w_k1_INTC_s102
                                 + 0.772257 w_k1_KO_s102
                                 + 0.908134 w_k1_MAT_s102
                                 + 0.806885 w_k1_MSFT_s102
                                 + 0.844251 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.36945 w_k1_ADM_s102
                                 + 1.23692 w_k1_AXP_s102
                                 + 1.27035 w_k1_DIS_s102
                                 + 1.27647 w_k1_FDX_s102
                                 + 0.803419 w_k1_IBM_s102
                                 + 1.07317 w_k1_INTC_s102
                                 + 1.04212 w_k1_KO_s102
                                 + 0.875604 w_k1_MAT_s102
                                 + 1.08063 w_k1_MSFT_s102
                                 + 1.11463 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.707856 w_k1_ADM_s102
                                 + 0.821213 w_k1_AXP_s102
                                 + 0.796866 w_k1_DIS_s102
                                 + 0.837713 w_k1_FDX_s102
                                 + 1.27233 w_k1_IBM_s102
                                 + 0.860401 w_k1_INTC_s102
                                 + 1.05712 w_k1_KO_s102 + 1.29895 w_k1_MAT_s102
                                 + 0.85504 w_k1_MSFT_s102
                                 + 0.893871 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.13308 w_k1_ADM_s102
                                 + 1.30885 w_k1_AXP_s102 + 1.701 w_k1_DIS_s102
                                 + 1.34472 w_k1_FDX_s102
                                 + 1.48929 w_k1_IBM_s102
                                 + 1.02564 w_k1_INTC_s102 + 1.4355 w_k1_KO_s102
                                 + 1.5931 w_k1_MAT_s102
                                 + 1.13035 w_k1_MSFT_s102
                                 + 1.14606 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.06907 w_k1_ADM_s102
                                 + 0.961688 w_k1_AXP_s102
                                 + 1.13468 w_k1_DIS_s102
                                 + 0.99636 w_k1_FDX_s102
                                 + 0.998845 w_k1_IBM_s102
                                 + 0.991405 w_k1_INTC_s102
                                 + 1.10577 w_k1_KO_s102 + 1.13026 w_k1_MAT_s102
                                 + 1.07117 w_k1_MSFT_s102
                                 + 0.984537 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.853834 w_k1_ADM_s102
                                 + 0.748358 w_k1_AXP_s102
                                 + 0.813425 w_k1_DIS_s102
                                 + 0.92808 w_k1_FDX_s102
                                 + 0.732761 w_k1_IBM_s102
                                 + 1.00654 w_k1_INTC_s102
                                 + 0.845163 w_k1_KO_s102
                                 + 0.585316 w_k1_MAT_s102
                                 + 0.880607 w_k1_MSFT_s102
                                 + 0.803392 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.12289 w_k1_ADM_s103
                                 + 0.93575 w_k1_AXP_s103
                                 + 0.908404 w_k1_DIS_s103
                                 + 0.971303 w_k1_FDX_s103
                                 + 0.92856 w_k1_IBM_s103
                                 + 0.808863 w_k1_INTC_s103
                                 + 0.840218 w_k1_KO_s103
                                 + 0.894378 w_k1_MAT_s103
                                 + 0.924333 w_k1_MSFT_s103
                                 + 0.980139 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.839157 w_k1_ADM_s103
                                 + 0.791128 w_k1_AXP_s103
                                 + 0.687775 w_k1_DIS_s103
                                 + 0.843768 w_k1_FDX_s103
                                 + 0.686665 w_k1_IBM_s103
                                 + 0.697064 w_k1_INTC_s103
                                 + 0.720178 w_k1_KO_s103
                                 + 0.65296 w_k1_MAT_s103
                                 + 0.748467 w_k1_MSFT_s103
                                 + 0.830666 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.729571 w_k1_ADM_s103
                                 + 0.622389 w_k1_AXP_s103
                                 + 0.642874 w_k1_DIS_s103
                                 + 0.691223 w_k1_FDX_s103
                                 + 0.921102 w_k1_IBM_s103
                                 + 0.767741 w_k1_INTC_s103
                                 + 0.840769 w_k1_KO_s103
                                 + 0.636112 w_k1_MAT_s103
                                 + 0.721981 w_k1_MSFT_s103
                                 + 0.909412 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.31539 w_k1_ADM_s103
                                 + 1.20147 w_k1_AXP_s103
                                 + 1.22548 w_k1_DIS_s103
                                 + 1.20437 w_k1_FDX_s103
                                 + 0.884595 w_k1_IBM_s103
                                 + 0.994589 w_k1_INTC_s103
                                 + 1.07052 w_k1_KO_s103
                                 + 0.712691 w_k1_MAT_s103
                                 + 1.10847 w_k1_MSFT_s103
                                 + 1.07638 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.984945 w_k1_ADM_s103
                                 + 1.18288 w_k1_AXP_s103
                                 + 1.11152 w_k1_DIS_s103
                                 + 1.10112 w_k1_FDX_s103 + 1.4057 w_k1_IBM_s103
                                 + 1.02679 w_k1_INTC_s103 + 1.262 w_k1_KO_s103
                                 + 1.39909 w_k1_MAT_s103
                                 + 1.15553 w_k1_MSFT_s103
                                 + 1.17694 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.03817 w_k1_ADM_s103
                                 + 1.00692 w_k1_AXP_s103
                                 + 0.978816 w_k1_DIS_s103
                                 + 0.878421 w_k1_FDX_s103
                                 + 1.2227 w_k1_IBM_s103
                                 + 0.875184 w_k1_INTC_s103
                                 + 1.13634 w_k1_KO_s103 + 1.11438 w_k1_MAT_s103
                                 + 0.926777 w_k1_MSFT_s103
                                 + 1.01028 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.17616 w_k1_ADM_s103
                                 + 1.46104 w_k1_AXP_s103
                                 + 1.67787 w_k1_DIS_s103
                                 + 1.31269 w_k1_FDX_s103
                                 + 1.34198 w_k1_IBM_s103
                                 + 1.17344 w_k1_INTC_s103
                                 + 1.22781 w_k1_KO_s103 + 1.17019 w_k1_MAT_s103
                                 + 1.27955 w_k1_MSFT_s103 + 1.2481 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.928174 w_k1_ADM_s103
                                 + 0.74697 w_k1_AXP_s103
                                 + 1.02569 w_k1_DIS_s103
                                 + 0.904945 w_k1_FDX_s103
                                 + 0.751178 w_k1_IBM_s103
                                 + 0.887495 w_k1_INTC_s103
                                 + 0.835819 w_k1_KO_s103
                                 + 0.791546 w_k1_MAT_s103
                                 + 1.14526 w_k1_MSFT_s103
                                 + 0.875833 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.82832 w_k1_ADM_s104
                                 + 0.774991 w_k1_AXP_s104
                                 + 0.834972 w_k1_DIS_s104
                                 + 0.83822 w_k1_FDX_s104
                                 + 0.942288 w_k1_IBM_s104
                                 + 0.852154 w_k1_INTC_s104
                                 + 0.834106 w_k1_KO_s104
                                 + 0.830163 w_k1_MAT_s104
                                 + 0.81819 w_k1_MSFT_s104
                                 + 0.893449 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.855669 w_k1_ADM_s104
                                 + 0.830323 w_k1_AXP_s104
                                 + 0.911659 w_k1_DIS_s104
                                 + 0.920008 w_k1_FDX_s104
                                 + 0.891098 w_k1_IBM_s104
                                 + 0.956202 w_k1_INTC_s104
                                 + 0.881689 w_k1_KO_s104
                                 + 0.939254 w_k1_MAT_s104
                                 + 0.885965 w_k1_MSFT_s104
                                 + 1.01539 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.882304 w_k1_ADM_s104
                                 + 1.03833 w_k1_AXP_s104
                                 + 0.77024 w_k1_DIS_s104
                                 + 0.866614 w_k1_FDX_s104
                                 + 0.995698 w_k1_IBM_s104
                                 + 0.820745 w_k1_INTC_s104 + 0.943 w_k1_KO_s104
                                 + 1.40223 w_k1_MAT_s104
                                 + 0.872322 w_k1_MSFT_s104
                                 + 0.999998 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.826089 w_k1_ADM_s104
                                 + 0.761092 w_k1_AXP_s104
                                 + 0.775715 w_k1_DIS_s104
                                 + 0.886588 w_k1_FDX_s104
                                 + 0.670653 w_k1_IBM_s104
                                 + 0.853058 w_k1_INTC_s104
                                 + 0.78572 w_k1_KO_s104
                                 + 0.687872 w_k1_MAT_s104
                                 + 0.814899 w_k1_MSFT_s104
                                 + 0.856552 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.05594 w_k1_ADM_s104
                                 + 1.09631 w_k1_AXP_s104
                                 + 1.20975 w_k1_DIS_s104
                                 + 1.19108 w_k1_FDX_s104
                                 + 1.11041 w_k1_IBM_s104
                                 + 1.17766 w_k1_INTC_s104
                                 + 1.12489 w_k1_KO_s104 + 1.09217 w_k1_MAT_s104
                                 + 1.18863 w_k1_MSFT_s104
                                 + 1.16811 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.1463 w_k1_ADM_s104
                                 + 1.07671 w_k1_AXP_s104
                                 + 1.20646 w_k1_DIS_s104
                                 + 1.29188 w_k1_FDX_s104
                                 + 0.782916 w_k1_IBM_s104
                                 + 1.07461 w_k1_INTC_s104
                                 + 0.947373 w_k1_KO_s104
                                 + 0.808096 w_k1_MAT_s104
                                 + 1.09304 w_k1_MSFT_s104 + 1.1038 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.16364 w_k1_ADM_s104
                                 + 1.24367 w_k1_AXP_s104
                                 + 1.11175 w_k1_DIS_s104
                                 + 1.00078 w_k1_FDX_s104
                                 + 1.52182 w_k1_IBM_s104
                                 + 1.00299 w_k1_INTC_s104
                                 + 1.08813 w_k1_KO_s104 + 1.74206 w_k1_MAT_s104
                                 + 0.959382 w_k1_MSFT_s104
                                 + 1.21593 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.895541 w_k1_ADM_s104
                                 + 0.683314 w_k1_AXP_s104
                                 + 0.543896 w_k1_DIS_s104
                                 + 0.721538 w_k1_FDX_s104
                                 + 0.728601 w_k1_IBM_s104
                                 + 0.835633 w_k1_INTC_s104
                                 + 0.640066 w_k1_KO_s104
                                 + 0.620103 w_k1_MAT_s104
                                 + 0.829849 w_k1_MSFT_s104
                                 + 0.744241 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.05758 w_k1_ADM_s105
                                 + 0.921388 w_k1_AXP_s105
                                 + 1.06475 w_k1_DIS_s105
                                 + 1.01936 w_k1_FDX_s105
                                 + 0.749533 w_k1_IBM_s105
                                 + 0.903914 w_k1_INTC_s105
                                 + 0.757497 w_k1_KO_s105
                                 + 1.08176 w_k1_MAT_s105
                                 + 0.932355 w_k1_MSFT_s105
                                 + 0.900002 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.1947 w_k1_ADM_s105
                                 + 1.27873 w_k1_AXP_s105
                                 + 1.40452 w_k1_DIS_s105
                                 + 1.30564 w_k1_FDX_s105
                                 + 0.891841 w_k1_IBM_s105
                                 + 1.21451 w_k1_INTC_s105
                                 + 1.04656 w_k1_KO_s105 + 1.47882 w_k1_MAT_s105
                                 + 1.22573 w_k1_MSFT_s105
                                 + 1.23629 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.832785 w_k1_ADM_s105
                                 + 0.719295 w_k1_AXP_s105
                                 + 0.648608 w_k1_DIS_s105
                                 + 0.765769 w_k1_FDX_s105
                                 + 0.712916 w_k1_IBM_s105
                                 + 0.780766 w_k1_INTC_s105
                                 + 0.641908 w_k1_KO_s105
                                 + 0.97494 w_k1_MAT_s105
                                 + 0.823049 w_k1_MSFT_s105
                                 + 0.83408 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.990451 w_k1_ADM_s105
                                 + 1.08844 w_k1_AXP_s105
                                 + 1.35429 w_k1_DIS_s105
                                 + 1.04916 w_k1_FDX_s105
                                 + 1.11709 w_k1_IBM_s105
                                 + 1.15361 w_k1_INTC_s105
                                 + 1.07413 w_k1_KO_s105 + 1.14073 w_k1_MAT_s105
                                 + 1.07746 w_k1_MSFT_s105
                                 + 1.14143 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.134 w_k1_ADM_s105
                                 + 1.25791 w_k1_AXP_s105
                                 + 1.46836 w_k1_DIS_s105
                                 + 1.13305 w_k1_FDX_s105
                                 + 1.47749 w_k1_IBM_s105
                                 + 1.22963 w_k1_INTC_s105
                                 + 1.39914 w_k1_KO_s105 + 1.96057 w_k1_MAT_s105
                                 + 1.29202 w_k1_MSFT_s105
                                 + 1.28365 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.50806 w_k1_ADM_s105
                                 + 1.58385 w_k1_AXP_s105
                                 + 1.81841 w_k1_DIS_s105
                                 + 1.57868 w_k1_FDX_s105
                                 + 1.25076 w_k1_IBM_s105
                                 + 1.47128 w_k1_INTC_s105
                                 + 1.38011 w_k1_KO_s105 + 2.07616 w_k1_MAT_s105
                                 + 1.59563 w_k1_MSFT_s105
                                 + 1.24261 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.929077 w_k1_ADM_s105
                                 + 0.798265 w_k1_AXP_s105
                                 + 0.819928 w_k1_DIS_s105
                                 + 0.776608 w_k1_FDX_s105
                                 + 1.13365 w_k1_IBM_s105
                                 + 0.851039 w_k1_INTC_s105
                                 + 0.957554 w_k1_KO_s105
                                 + 1.46472 w_k1_MAT_s105
                                 + 0.88702 w_k1_MSFT_s105
                                 + 0.99111 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.1252 w_k1_ADM_s105
                                 + 0.971778 w_k1_AXP_s105
                                 + 1.02761 w_k1_DIS_s105
                                 + 1.10417 w_k1_FDX_s105
                                 + 0.906378 w_k1_IBM_s105
                                 + 1.20948 w_k1_INTC_s105
                                 + 0.931527 w_k1_KO_s105
                                 + 1.12454 w_k1_MAT_s105
                                 + 1.08775 w_k1_MSFT_s105
                                 + 1.04288 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.99107 w_k1_ADM_s106
                                 + 0.88135 w_k1_AXP_s106
                                 + 0.991264 w_k1_DIS_s106
                                 + 1.02877 w_k1_FDX_s106
                                 + 0.767693 w_k1_IBM_s106
                                 + 0.897717 w_k1_INTC_s106
                                 + 0.864725 w_k1_KO_s106
                                 + 0.727207 w_k1_MAT_s106
                                 + 1.02948 w_k1_MSFT_s106
                                 + 1.07509 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.727292 w_k1_ADM_s106
                                 + 0.809584 w_k1_AXP_s106
                                 + 0.868924 w_k1_DIS_s106
                                 + 0.827111 w_k1_FDX_s106
                                 + 1.4505 w_k1_IBM_s106
                                 + 0.888929 w_k1_INTC_s106
                                 + 1.09149 w_k1_KO_s106 + 1.28472 w_k1_MAT_s106
                                 + 0.829806 w_k1_MSFT_s106
                                 + 0.96642 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.868963 w_k1_ADM_s106
                                 + 0.977345 w_k1_AXP_s106
                                 + 1.13438 w_k1_DIS_s106
                                 + 0.973849 w_k1_FDX_s106
                                 + 1.24691 w_k1_IBM_s106
                                 + 0.919726 w_k1_INTC_s106
                                 + 1.25155 w_k1_KO_s106
                                 + 0.951164 w_k1_MAT_s106
                                 + 1.11795 w_k1_MSFT_s106
                                 + 1.10381 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.802318 w_k1_ADM_s106
                                 + 1.10692 w_k1_AXP_s106
                                 + 0.814873 w_k1_DIS_s106
                                 + 0.929936 w_k1_FDX_s106
                                 + 1.09596 w_k1_IBM_s106
                                 + 0.877223 w_k1_INTC_s106
                                 + 1.01214 w_k1_KO_s106 + 1.57985 w_k1_MAT_s106
                                 + 0.917654 w_k1_MSFT_s106
                                 + 0.991002 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.17408 w_k1_ADM_s106
                                 + 1.31956 w_k1_AXP_s106
                                 + 1.74889 w_k1_DIS_s106 + 1.4084 w_k1_FDX_s106
                                 + 1.3193 w_k1_IBM_s106
                                 + 1.06212 w_k1_INTC_s106
                                 + 1.58523 w_k1_KO_s106 + 1.12829 w_k1_MAT_s106
                                 + 1.36624 w_k1_MSFT_s106
                                 + 1.37804 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.763868 w_k1_ADM_s106
                                 + 0.702562 w_k1_AXP_s106
                                 + 0.714425 w_k1_DIS_s106
                                 + 0.798858 w_k1_FDX_s106
                                 + 0.816646 w_k1_IBM_s106
                                 + 0.76805 w_k1_INTC_s106
                                 + 0.841816 w_k1_KO_s106
                                 + 0.860927 w_k1_MAT_s106
                                 + 0.859476 w_k1_MSFT_s106
                                 + 0.902885 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.12891 w_k1_ADM_s106
                                 + 1.16739 w_k1_AXP_s106
                                 + 1.34832 w_k1_DIS_s106
                                 + 1.24589 w_k1_FDX_s106 + 1.0243 w_k1_IBM_s106
                                 + 1.04354 w_k1_INTC_s106
                                 + 1.20877 w_k1_KO_s106 + 1.00702 w_k1_MAT_s106
                                 + 1.25519 w_k1_MSFT_s106
                                 + 1.16287 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.11824 w_k1_ADM_s106
                                 + 1.39768 w_k1_AXP_s106
                                 + 1.11384 w_k1_DIS_s106
                                 + 1.14063 w_k1_FDX_s106
                                 + 1.48877 w_k1_IBM_s106
                                 + 1.1508 w_k1_INTC_s106 + 1.40184 w_k1_KO_s106
                                 + 1.86836 w_k1_MAT_s106
                                 + 1.06324 w_k1_MSFT_s106
                                 + 1.20272 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.22392 w_k1_ADM_s107
                                 + 1.1818 w_k1_AXP_s107 + 1.2211 w_k1_DIS_s107
                                 + 1.05886 w_k1_FDX_s107
                                 + 1.15487 w_k1_IBM_s107
                                 + 1.05758 w_k1_INTC_s107 + 1.1211 w_k1_KO_s107
                                 + 1.20345 w_k1_MAT_s107
                                 + 1.07626 w_k1_MSFT_s107
                                 + 1.09837 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.858893 w_k1_ADM_s107
                                 + 0.948753 w_k1_AXP_s107
                                 + 0.551009 w_k1_DIS_s107
                                 + 0.777589 w_k1_FDX_s107
                                 + 0.712855 w_k1_IBM_s107
                                 + 0.73684 w_k1_INTC_s107
                                 + 0.711565 w_k1_KO_s107 + 1.0469 w_k1_MAT_s107
                                 + 0.747544 w_k1_MSFT_s107
                                 + 0.904329 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.12571 w_k1_ADM_s107
                                 + 1.33495 w_k1_AXP_s107
                                 + 1.36114 w_k1_DIS_s107
                                 + 1.05047 w_k1_FDX_s107
                                 + 1.47061 w_k1_IBM_s107
                                 + 1.07217 w_k1_INTC_s107
                                 + 1.46798 w_k1_KO_s107 + 2.04829 w_k1_MAT_s107
                                 + 1.23994 w_k1_MSFT_s107
                                 + 1.22133 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.04434 w_k1_ADM_s107
                                 + 1.66403 w_k1_AXP_s107
                                 + 1.83286 w_k1_DIS_s107
                                 + 1.40705 w_k1_FDX_s107
                                 + 1.68678 w_k1_IBM_s107
                                 + 1.29939 w_k1_INTC_s107 + 1.5757 w_k1_KO_s107
                                 + 2.37615 w_k1_MAT_s107
                                 + 1.54012 w_k1_MSFT_s107
                                 + 1.31214 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.05653 w_k1_ADM_s107
                                 + 0.858154 w_k1_AXP_s107
                                 + 0.801381 w_k1_DIS_s107
                                 + 0.751262 w_k1_FDX_s107
                                 + 1.2701 w_k1_IBM_s107
                                 + 0.89661 w_k1_INTC_s107
                                 + 1.02138 w_k1_KO_s107 + 1.01354 w_k1_MAT_s107
                                 + 0.777027 w_k1_MSFT_s107
                                 + 0.927709 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.36745 w_k1_ADM_s107
                                 + 1.47068 w_k1_AXP_s107
                                 + 1.77903 w_k1_DIS_s107
                                 + 1.34116 w_k1_FDX_s107
                                 + 1.02154 w_k1_IBM_s107
                                 + 1.34281 w_k1_INTC_s107
                                 + 1.24723 w_k1_KO_s107 + 1.32244 w_k1_MAT_s107
                                 + 1.32813 w_k1_MSFT_s107
                                 + 1.27284 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.794355 w_k1_ADM_s107
                                 + 0.690328 w_k1_AXP_s107
                                 + 0.625683 w_k1_DIS_s107
                                 + 0.758943 w_k1_FDX_s107
                                 + 0.748199 w_k1_IBM_s107
                                 + 0.879272 w_k1_INTC_s107
                                 + 0.819886 w_k1_KO_s107
                                 + 0.91835 w_k1_MAT_s107
                                 + 0.890643 w_k1_MSFT_s107
                                 + 0.930934 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.92268 w_k1_ADM_s107
                                 + 1.07362 w_k1_AXP_s107
                                 + 0.969926 w_k1_DIS_s107
                                 + 1.04082 w_k1_FDX_s107
                                 + 0.82987 w_k1_IBM_s107
                                 + 0.932643 w_k1_INTC_s107
                                 + 0.895503 w_k1_KO_s107
                                 + 1.03819 w_k1_MAT_s107
                                 + 1.07044 w_k1_MSFT_s107
                                 + 0.995857 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.912536 w_k1_ADM_s110
                                 + 0.96652 w_k1_AXP_s110
                                 + 0.901999 w_k1_DIS_s110
                                 + 0.880173 w_k1_FDX_s110
                                 + 1.17838 w_k1_IBM_s110
                                 + 0.751735 w_k1_INTC_s110
                                 + 0.85603 w_k1_KO_s110 + 1.07538 w_k1_MAT_s110
                                 + 0.775662 w_k1_MSFT_s110
                                 + 0.93412 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.11175 w_k1_ADM_s110
                                 + 1.02437 w_k1_AXP_s110 + 1.1501 w_k1_DIS_s110
                                 + 1.09573 w_k1_FDX_s110
                                 + 0.823306 w_k1_IBM_s110
                                 + 0.965187 w_k1_INTC_s110
                                 + 0.80484 w_k1_KO_s110
                                 + 0.926373 w_k1_MAT_s110
                                 + 1.07584 w_k1_MSFT_s110
                                 + 0.984341 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.31426 w_k1_ADM_s110
                                 + 1.26475 w_k1_AXP_s110
                                 + 1.12495 w_k1_DIS_s110
                                 + 1.00279 w_k1_FDX_s110
                                 + 1.00937 w_k1_IBM_s110
                                 + 1.05887 w_k1_INTC_s110
                                 + 0.879123 w_k1_KO_s110
                                 + 1.04196 w_k1_MAT_s110
                                 + 0.897061 w_k1_MSFT_s110
                                 + 1.0188 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.33862 w_k1_ADM_s110
                                 + 1.37708 w_k1_AXP_s110
                                 + 1.46671 w_k1_DIS_s110
                                 + 1.32188 w_k1_FDX_s110
                                 + 0.829498 w_k1_IBM_s110
                                 + 1.29008 w_k1_INTC_s110
                                 + 1.01898 w_k1_KO_s110
                                 + 0.816982 w_k1_MAT_s110
                                 + 1.33648 w_k1_MSFT_s110
                                 + 1.21531 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.870997 w_k1_ADM_s110
                                 + 0.66273 w_k1_AXP_s110
                                 + 0.966633 w_k1_DIS_s110
                                 + 0.817063 w_k1_FDX_s110
                                 + 0.937412 w_k1_IBM_s110
                                 + 1.07535 w_k1_INTC_s110
                                 + 0.853106 w_k1_KO_s110
                                 + 0.751477 w_k1_MAT_s110
                                 + 1.01052 w_k1_MSFT_s110
                                 + 0.975739 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.988728 w_k1_ADM_s110
                                 + 0.840207 w_k1_AXP_s110
                                 + 0.757751 w_k1_DIS_s110
                                 + 0.910686 w_k1_FDX_s110
                                 + 0.686047 w_k1_IBM_s110
                                 + 1.02884 w_k1_INTC_s110
                                 + 0.672234 w_k1_KO_s110
                                 + 0.803507 w_k1_MAT_s110
                                 + 0.940896 w_k1_MSFT_s110
                                 + 0.976282 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.13667 w_k1_ADM_s110
                                 + 1.0753 w_k1_AXP_s110 + 1.42607 w_k1_DIS_s110
                                 + 1.14759 w_k1_FDX_s110 + 1.5833 w_k1_IBM_s110
                                 + 1.07275 w_k1_INTC_s110
                                 + 1.18812 w_k1_KO_s110 + 1.85955 w_k1_MAT_s110
                                 + 1.18212 w_k1_MSFT_s110
                                 + 1.07326 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.869499 w_k1_ADM_s110
                                 + 0.553691 w_k1_AXP_s110
                                 + 0.5963 w_k1_DIS_s110
                                 + 0.727169 w_k1_FDX_s110
                                 + 0.694262 w_k1_IBM_s110
                                 + 0.826745 w_k1_INTC_s110
                                 + 0.639786 w_k1_KO_s110
                                 + 0.584872 w_k1_MAT_s110
                                 + 0.808208 w_k1_MSFT_s110
                                 + 0.776536 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.835714 w_k1_ADM_s111
                                 + 0.741073 w_k1_AXP_s111
                                 + 0.648917 w_k1_DIS_s111
                                 + 0.749996 w_k1_FDX_s111
                                 + 1.07749 w_k1_IBM_s111
                                 + 1.04874 w_k1_INTC_s111
                                 + 1.06557 w_k1_KO_s111 + 1.1811 w_k1_MAT_s111
                                 + 0.907722 w_k1_MSFT_s111
                                 + 0.903821 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.71314 w_k1_ADM_s111
                                 + 0.60602 w_k1_AXP_s111
                                 + 0.557313 w_k1_DIS_s111
                                 + 0.774316 w_k1_FDX_s111
                                 + 0.809476 w_k1_IBM_s111
                                 + 0.730748 w_k1_INTC_s111
                                 + 0.756363 w_k1_KO_s111
                                 + 0.795424 w_k1_MAT_s111
                                 + 0.79615 w_k1_MSFT_s111
                                 + 0.84489 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.19316 w_k1_ADM_s111
                                 + 1.76986 w_k1_AXP_s111
                                 + 1.52817 w_k1_DIS_s111
                                 + 1.58101 w_k1_FDX_s111
                                 + 1.47549 w_k1_IBM_s111
                                 + 1.30752 w_k1_INTC_s111
                                 + 1.53318 w_k1_KO_s111 + 2.07481 w_k1_MAT_s111
                                 + 1.2522 w_k1_MSFT_s111 + 1.33952 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.23296 w_k1_ADM_s111
                                 + 1.01686 w_k1_AXP_s111
                                 + 0.998748 w_k1_DIS_s111
                                 + 1.21577 w_k1_FDX_s111
                                 + 0.763401 w_k1_IBM_s111
                                 + 1.14794 w_k1_INTC_s111
                                 + 1.04954 w_k1_KO_s111
                                 + 0.821184 w_k1_MAT_s111
                                 + 1.06361 w_k1_MSFT_s111
                                 + 1.11976 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.78611 w_k1_ADM_s111
                                 + 1.65695 w_k1_AXP_s111
                                 + 1.51598 w_k1_DIS_s111
                                 + 1.49602 w_k1_FDX_s111
                                 + 0.921087 w_k1_IBM_s111
                                 + 1.29626 w_k1_INTC_s111
                                 + 1.19625 w_k1_KO_s111 + 1.22076 w_k1_MAT_s111
                                 + 1.46461 w_k1_MSFT_s111
                                 + 1.34736 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.815829 w_k1_ADM_s111
                                 + 0.721842 w_k1_AXP_s111
                                 + 0.63333 w_k1_DIS_s111
                                 + 0.907449 w_k1_FDX_s111
                                 + 0.654014 w_k1_IBM_s111
                                 + 1.0611 w_k1_INTC_s111
                                 + 0.767172 w_k1_KO_s111
                                 + 0.668971 w_k1_MAT_s111
                                 + 0.91915 w_k1_MSFT_s111
                                 + 0.93104 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.31205 w_k1_ADM_s111
                                 + 1.31529 w_k1_AXP_s111
                                 + 1.43874 w_k1_DIS_s111
                                 + 1.38922 w_k1_FDX_s111
                                 + 1.08253 w_k1_IBM_s111
                                 + 1.45147 w_k1_INTC_s111
                                 + 1.20367 w_k1_KO_s111 + 1.14984 w_k1_MAT_s111
                                 + 1.38542 w_k1_MSFT_s111
                                 + 1.26699 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.859256 w_k1_ADM_s111
                                 + 0.960994 w_k1_AXP_s111
                                 + 0.971166 w_k1_DIS_s111
                                 + 0.993988 w_k1_FDX_s111
                                 + 1.10198 w_k1_IBM_s111
                                 + 1.00848 w_k1_INTC_s111
                                 + 1.10836 w_k1_KO_s111 + 1.23832 w_k1_MAT_s111
                                 + 0.96322 w_k1_MSFT_s111
                                 + 1.07232 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.82039 w_k1_ADM_s112
                                 + 0.730332 w_k1_AXP_s112
                                 + 0.787035 w_k1_DIS_s112
                                 + 0.882866 w_k1_FDX_s112
                                 + 0.656005 w_k1_IBM_s112
                                 + 1.073 w_k1_INTC_s112 + 0.80536 w_k1_KO_s112
                                 + 0.633386 w_k1_MAT_s112
                                 + 1.07518 w_k1_MSFT_s112
                                 + 0.92211 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.10474 w_k1_ADM_s112
                                 + 1.44169 w_k1_AXP_s112
                                 + 1.29828 w_k1_DIS_s112
                                 + 1.27043 w_k1_FDX_s112
                                 + 1.39182 w_k1_IBM_s112
                                 + 1.3805 w_k1_INTC_s112 + 1.06238 w_k1_KO_s112
                                 + 1.449 w_k1_MAT_s112 + 1.10177 w_k1_MSFT_s112
                                 + 1.09101 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.806977 w_k1_ADM_s112
                                 + 0.626101 w_k1_AXP_s112
                                 + 0.67392 w_k1_DIS_s112
                                 + 0.777954 w_k1_FDX_s112
                                 + 0.780404 w_k1_IBM_s112
                                 + 0.977051 w_k1_INTC_s112
                                 + 0.842823 w_k1_KO_s112
                                 + 0.718268 w_k1_MAT_s112
                                 + 0.904162 w_k1_MSFT_s112
                                 + 0.781361 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.236 w_k1_ADM_s112
                                 + 1.04953 w_k1_AXP_s112
                                 + 1.34594 w_k1_DIS_s112
                                 + 1.29576 w_k1_FDX_s112
                                 + 0.977868 w_k1_IBM_s112
                                 + 1.56808 w_k1_INTC_s112
                                 + 1.24094 w_k1_KO_s112
                                 + 0.874203 w_k1_MAT_s112
                                 + 1.3493 w_k1_MSFT_s112 + 1.05825 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.23592 w_k1_ADM_s112
                                 + 1.31354 w_k1_AXP_s112
                                 + 1.77189 w_k1_DIS_s112
                                 + 1.34572 w_k1_FDX_s112
                                 + 1.51554 w_k1_IBM_s112
                                 + 1.46928 w_k1_INTC_s112
                                 + 1.61092 w_k1_KO_s112 + 1.58289 w_k1_MAT_s112
                                 + 1.53202 w_k1_MSFT_s112
                                 + 1.42115 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.05316 w_k1_ADM_s112
                                 + 0.960503 w_k1_AXP_s112
                                 + 0.943758 w_k1_DIS_s112
                                 + 1.16947 w_k1_FDX_s112
                                 + 0.789052 w_k1_IBM_s112
                                 + 1.15869 w_k1_INTC_s112
                                 + 0.880036 w_k1_KO_s112
                                 + 0.56744 w_k1_MAT_s112
                                 + 1.07799 w_k1_MSFT_s112
                                 + 0.915581 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.842107 w_k1_ADM_s112
                                 + 1.07439 w_k1_AXP_s112
                                 + 0.854054 w_k1_DIS_s112
                                 + 0.91016 w_k1_FDX_s112
                                 + 1.44041 w_k1_IBM_s112
                                 + 0.955818 w_k1_INTC_s112
                                 + 0.879188 w_k1_KO_s112
                                 + 1.51289 w_k1_MAT_s112
                                 + 0.948216 w_k1_MSFT_s112
                                 + 0.996533 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.998594 w_k1_ADM_s112
                                 + 0.856216 w_k1_AXP_s112
                                 + 1.01821 w_k1_DIS_s112
                                 + 1.01684 w_k1_FDX_s112
                                 + 1.09162 w_k1_IBM_s112
                                 + 1.03909 w_k1_INTC_s112 + 1.0204 w_k1_KO_s112
                                 + 1.03304 w_k1_MAT_s112
                                 + 1.03069 w_k1_MSFT_s112
                                 + 1.00415 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.759957 w_k1_ADM_s113
                                 + 1.23946 w_k1_AXP_s113
                                 + 1.46468 w_k1_DIS_s113
                                 + 1.09294 w_k1_FDX_s113
                                 + 1.75354 w_k1_IBM_s113
                                 + 1.06112 w_k1_INTC_s113
                                 + 1.62979 w_k1_KO_s113 + 1.60744 w_k1_MAT_s113
                                 + 1.13065 w_k1_MSFT_s113
                                 + 1.10343 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.795393 w_k1_ADM_s113
                                 + 1.19514 w_k1_AXP_s113 + 1.0095 w_k1_DIS_s113
                                 + 0.981948 w_k1_FDX_s113
                                 + 1.35278 w_k1_IBM_s113
                                 + 0.942724 w_k1_INTC_s113
                                 + 1.03108 w_k1_KO_s113 + 1.27782 w_k1_MAT_s113
                                 + 0.921884 w_k1_MSFT_s113
                                 + 1.03465 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.73645 w_k1_ADM_s113
                                 + 0.904001 w_k1_AXP_s113
                                 + 0.620216 w_k1_DIS_s113
                                 + 0.743631 w_k1_FDX_s113
                                 + 0.898743 w_k1_IBM_s113
                                 + 0.647845 w_k1_INTC_s113
                                 + 0.710441 w_k1_KO_s113
                                 + 0.829078 w_k1_MAT_s113
                                 + 0.81008 w_k1_MSFT_s113
                                 + 0.718499 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.05172 w_k1_ADM_s113
                                 + 1.51174 w_k1_AXP_s113
                                 + 1.71771 w_k1_DIS_s113
                                 + 1.45715 w_k1_FDX_s113
                                 + 1.19862 w_k1_IBM_s113
                                 + 1.2968 w_k1_INTC_s113 + 1.35167 w_k1_KO_s113
                                 + 1.26754 w_k1_MAT_s113
                                 + 1.42328 w_k1_MSFT_s113
                                 + 1.13783 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.844228 w_k1_ADM_s113
                                 + 1.25114 w_k1_AXP_s113
                                 + 1.05053 w_k1_DIS_s113
                                 + 1.07071 w_k1_FDX_s113
                                 + 0.793374 w_k1_IBM_s113
                                 + 0.79012 w_k1_INTC_s113
                                 + 0.865011 w_k1_KO_s113
                                 + 0.849062 w_k1_MAT_s113
                                 + 1.01392 w_k1_MSFT_s113
                                 + 0.958744 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.06535 w_k1_ADM_s113
                                 + 1.14459 w_k1_AXP_s113
                                 + 1.26338 w_k1_DIS_s113
                                 + 1.29494 w_k1_FDX_s113
                                 + 0.948762 w_k1_IBM_s113
                                 + 1.20369 w_k1_INTC_s113
                                 + 0.998153 w_k1_KO_s113
                                 + 0.959338 w_k1_MAT_s113
                                 + 1.40056 w_k1_MSFT_s113 + 1.0961 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.86046 w_k1_ADM_s113
                                 + 0.87031 w_k1_AXP_s113
                                 + 0.747557 w_k1_DIS_s113
                                 + 0.928993 w_k1_FDX_s113
                                 + 0.653023 w_k1_IBM_s113
                                 + 0.694684 w_k1_INTC_s113
                                 + 0.680869 w_k1_KO_s113
                                 + 0.587015 w_k1_MAT_s113
                                 + 0.891518 w_k1_MSFT_s113
                                 + 0.822379 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.778892 w_k1_ADM_s113
                                 + 0.910661 w_k1_AXP_s113
                                 + 1.03406 w_k1_DIS_s113
                                 + 0.899556 w_k1_FDX_s113
                                 + 1.01933 w_k1_IBM_s113
                                 + 0.933776 w_k1_INTC_s113
                                 + 1.01439 w_k1_KO_s113
                                 + 0.816994 w_k1_MAT_s113
                                 + 1.03979 w_k1_MSFT_s113
                                 + 0.876166 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.93339 w_k1_ADM_s114
                                 + 0.687089 w_k1_AXP_s114
                                 + 0.636023 w_k1_DIS_s114
                                 + 0.757464 w_k1_FDX_s114
                                 + 0.803399 w_k1_IBM_s114
                                 + 0.816688 w_k1_INTC_s114
                                 + 0.745899 w_k1_KO_s114
                                 + 0.779177 w_k1_MAT_s114
                                 + 0.793204 w_k1_MSFT_s114
                                 + 0.83429 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.15806 w_k1_ADM_s114
                                 + 1.06166 w_k1_AXP_s114
                                 + 1.17823 w_k1_DIS_s114 + 1.1308 w_k1_FDX_s114
                                 + 0.903949 w_k1_IBM_s114
                                 + 0.986851 w_k1_INTC_s114
                                 + 1.10365 w_k1_KO_s114
                                 + 0.952061 w_k1_MAT_s114
                                 + 1.18141 w_k1_MSFT_s114 + 1.1338 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.26393 w_k1_ADM_s114
                                 + 1.28084 w_k1_AXP_s114
                                 + 1.46591 w_k1_DIS_s114
                                 + 1.16236 w_k1_FDX_s114 + 1.3718 w_k1_IBM_s114
                                 + 1.02626 w_k1_INTC_s114
                                 + 1.39007 w_k1_KO_s114 + 1.69881 w_k1_MAT_s114
                                 + 1.0854 w_k1_MSFT_s114 + 1.22177 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.15632 w_k1_ADM_s114
                                 + 0.862865 w_k1_AXP_s114
                                 + 0.918536 w_k1_DIS_s114
                                 + 0.912108 w_k1_FDX_s114
                                 + 1.29547 w_k1_IBM_s114
                                 + 0.873812 w_k1_INTC_s114 + 1.003 w_k1_KO_s114
                                 + 0.772111 w_k1_MAT_s114
                                 + 0.985001 w_k1_MSFT_s114
                                 + 1.00893 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.74101 w_k1_ADM_s114
                                 + 1.48699 w_k1_AXP_s114
                                 + 1.72059 w_k1_DIS_s114
                                 + 1.27493 w_k1_FDX_s114
                                 + 1.31022 w_k1_IBM_s114
                                 + 1.18107 w_k1_INTC_s114
                                 + 1.57071 w_k1_KO_s114
                                 + 0.955885 w_k1_MAT_s114
                                 + 1.3277 w_k1_MSFT_s114 + 1.12445 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.09746 w_k1_ADM_s114
                                 + 0.956098 w_k1_AXP_s114
                                 + 0.776276 w_k1_DIS_s114
                                 + 0.948797 w_k1_FDX_s114
                                 + 0.64232 w_k1_IBM_s114
                                 + 0.659006 w_k1_INTC_s114
                                 + 0.666225 w_k1_KO_s114
                                 + 0.649769 w_k1_MAT_s114
                                 + 0.764189 w_k1_MSFT_s114
                                 + 0.861881 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.04411 w_k1_ADM_s114
                                 + 0.899219 w_k1_AXP_s114
                                 + 1.00473 w_k1_DIS_s114
                                 + 0.879665 w_k1_FDX_s114
                                 + 0.841134 w_k1_IBM_s114
                                 + 0.802376 w_k1_INTC_s114
                                 + 0.969618 w_k1_KO_s114
                                 + 0.753771 w_k1_MAT_s114
                                 + 0.922182 w_k1_MSFT_s114
                                 + 0.980341 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.28557 w_k1_ADM_s114
                                 + 1.07465 w_k1_AXP_s114
                                 + 0.908364 w_k1_DIS_s114
                                 + 0.971279 w_k1_FDX_s114
                                 + 1.05274 w_k1_IBM_s114
                                 + 0.75794 w_k1_INTC_s114
                                 + 0.950323 w_k1_KO_s114
                                 + 1.09796 w_k1_MAT_s114
                                 + 0.979619 w_k1_MSFT_s114
                                 + 0.977973 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.809549 w_k1_ADM_s115
                                 + 0.799659 w_k1_AXP_s115
                                 + 0.741389 w_k1_DIS_s115
                                 + 0.802197 w_k1_FDX_s115
                                 + 0.98788 w_k1_IBM_s115
                                 + 0.808228 w_k1_INTC_s115
                                 + 1.01864 w_k1_KO_s115 + 1.12711 w_k1_MAT_s115
                                 + 0.840744 w_k1_MSFT_s115
                                 + 1.00368 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.05132 w_k1_ADM_s115
                                 + 1.13215 w_k1_AXP_s115
                                 + 1.25458 w_k1_DIS_s115
                                 + 1.03524 w_k1_FDX_s115
                                 + 1.05266 w_k1_IBM_s115
                                 + 1.12739 w_k1_INTC_s115
                                 + 1.24703 w_k1_KO_s115 + 1.17322 w_k1_MAT_s115
                                 + 1.22543 w_k1_MSFT_s115
                                 + 1.16839 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.775241 w_k1_ADM_s115
                                 + 0.560883 w_k1_AXP_s115
                                 + 0.64185 w_k1_DIS_s115
                                 + 0.665069 w_k1_FDX_s115
                                 + 0.648402 w_k1_IBM_s115
                                 + 0.752372 w_k1_INTC_s115
                                 + 0.768965 w_k1_KO_s115
                                 + 0.408864 w_k1_MAT_s115
                                 + 0.773911 w_k1_MSFT_s115
                                 + 0.89132 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.04978 w_k1_ADM_s115
                                 + 1.22116 w_k1_AXP_s115
                                 + 1.24551 w_k1_DIS_s115
                                 + 1.07224 w_k1_FDX_s115 + 1.7899 w_k1_IBM_s115
                                 + 1.40009 w_k1_INTC_s115
                                 + 1.61887 w_k1_KO_s115 + 2.10959 w_k1_MAT_s115
                                 + 1.2202 w_k1_MSFT_s115 + 1.28008 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.843792 w_k1_ADM_s115
                                 + 0.895694 w_k1_AXP_s115
                                 + 0.635398 w_k1_DIS_s115
                                 + 0.873844 w_k1_FDX_s115
                                 + 0.628918 w_k1_IBM_s115
                                 + 1.15522 w_k1_INTC_s115
                                 + 0.795699 w_k1_KO_s115
                                 + 0.812326 w_k1_MAT_s115
                                 + 0.896 w_k1_MSFT_s115 + 0.8547 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.35462 w_k1_ADM_s115
                                 + 1.48478 w_k1_AXP_s115
                                 + 1.41137 w_k1_DIS_s115 + 1.4397 w_k1_FDX_s115
                                 + 0.916515 w_k1_IBM_s115
                                 + 1.20129 w_k1_INTC_s115
                                 + 1.29403 w_k1_KO_s115
                                 + 0.850847 w_k1_MAT_s115
                                 + 1.17832 w_k1_MSFT_s115
                                 + 1.27677 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.976971 w_k1_ADM_s115
                                 + 0.725043 w_k1_AXP_s115
                                 + 0.93127 w_k1_DIS_s115
                                 + 1.00497 w_k1_FDX_s115
                                 + 0.582458 w_k1_IBM_s115
                                 + 0.932497 w_k1_INTC_s115
                                 + 0.88114 w_k1_KO_s115
                                 + 0.544354 w_k1_MAT_s115
                                 + 0.888509 w_k1_MSFT_s115
                                 + 1.09907 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.933353 w_k1_ADM_s115
                                 + 1.01634 w_k1_AXP_s115
                                 + 0.902984 w_k1_DIS_s115
                                 + 1.00379 w_k1_FDX_s115
                                 + 1.05403 w_k1_IBM_s115
                                 + 1.04644 w_k1_INTC_s115
                                 + 1.05419 w_k1_KO_s115 + 1.11548 w_k1_MAT_s115
                                 + 1.05716 w_k1_MSFT_s115
                                 + 1.15141 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.36299 w_k1_ADM_s116
                                 + 1.56938 w_k1_AXP_s116
                                 + 1.47513 w_k1_DIS_s116
                                 + 1.36538 w_k1_FDX_s116
                                 + 1.12424 w_k1_IBM_s116
                                 + 1.0661 w_k1_INTC_s116 + 1.27182 w_k1_KO_s116
                                 + 1.27158 w_k1_MAT_s116
                                 + 1.43089 w_k1_MSFT_s116
                                 + 1.06929 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.759995 w_k1_ADM_s116
                                 + 0.753745 w_k1_AXP_s116
                                 + 0.559838 w_k1_DIS_s116
                                 + 0.807997 w_k1_FDX_s116
                                 + 0.650273 w_k1_IBM_s116
                                 + 0.813111 w_k1_INTC_s116
                                 + 0.644918 w_k1_KO_s116
                                 + 0.527531 w_k1_MAT_s116
                                 + 0.754792 w_k1_MSFT_s116
                                 + 0.758243 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.01173 w_k1_ADM_s116
                                 + 0.982129 w_k1_AXP_s116
                                 + 0.962501 w_k1_DIS_s116
                                 + 1.00421 w_k1_FDX_s116
                                 + 1.11444 w_k1_IBM_s116
                                 + 0.750457 w_k1_INTC_s116
                                 + 0.943024 w_k1_KO_s116
                                 + 1.23454 w_k1_MAT_s116
                                 + 0.936263 w_k1_MSFT_s116
                                 + 0.962429 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.28603 w_k1_ADM_s116
                                 + 1.15525 w_k1_AXP_s116
                                 + 1.12719 w_k1_DIS_s116
                                 + 1.12952 w_k1_FDX_s116
                                 + 1.51893 w_k1_IBM_s116
                                 + 0.699845 w_k1_INTC_s116
                                 + 1.13594 w_k1_KO_s116 + 1.43377 w_k1_MAT_s116
                                 + 1.03555 w_k1_MSFT_s116
                                 + 1.09295 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.907836 w_k1_ADM_s116
                                 + 0.870157 w_k1_AXP_s116
                                 + 0.904942 w_k1_DIS_s116
                                 + 0.994225 w_k1_FDX_s116
                                 + 0.902184 w_k1_IBM_s116
                                 + 0.878601 w_k1_INTC_s116
                                 + 0.902587 w_k1_KO_s116
                                 + 1.00359 w_k1_MAT_s116
                                 + 0.940852 w_k1_MSFT_s116
                                 + 0.995299 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.06922 w_k1_ADM_s116
                                 + 1.12366 w_k1_AXP_s116
                                 + 0.99415 w_k1_DIS_s116
                                 + 1.21616 w_k1_FDX_s116
                                 + 0.96874 w_k1_IBM_s116
                                 + 1.0412 w_k1_INTC_s116 + 1.01345 w_k1_KO_s116
                                 + 1.16497 w_k1_MAT_s116
                                 + 1.27189 w_k1_MSFT_s116
                                 + 1.14188 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.724369 w_k1_ADM_s116
                                 + 0.792949 w_k1_AXP_s116
                                 + 0.725495 w_k1_DIS_s116
                                 + 0.784947 w_k1_FDX_s116
                                 + 0.993592 w_k1_IBM_s116
                                 + 0.706132 w_k1_INTC_s116
                                 + 0.902707 w_k1_KO_s116
                                 + 0.963402 w_k1_MAT_s116
                                 + 0.883668 w_k1_MSFT_s116
                                 + 0.870571 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.983319 w_k1_ADM_s116
                                 + 1.02105 w_k1_AXP_s116
                                 + 0.813352 w_k1_DIS_s116
                                 + 1.02329 w_k1_FDX_s116
                                 + 0.71497 w_k1_IBM_s116
                                 + 0.858534 w_k1_INTC_s116
                                 + 0.661331 w_k1_KO_s116
                                 + 0.61639 w_k1_MAT_s116
                                 + 0.959849 w_k1_MSFT_s116
                                 + 0.904216 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.921246 w_k1_ADM_s117
                                 + 0.896314 w_k1_AXP_s117
                                 + 0.723845 w_k1_DIS_s117
                                 + 0.84392 w_k1_FDX_s117
                                 + 1.08521 w_k1_IBM_s117
                                 + 0.852808 w_k1_INTC_s117
                                 + 0.878822 w_k1_KO_s117
                                 + 0.815094 w_k1_MAT_s117
                                 + 0.880227 w_k1_MSFT_s117
                                 + 0.957047 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.58028 w_k1_ADM_s117
                                 + 1.64641 w_k1_AXP_s117
                                 + 1.69917 w_k1_DIS_s117
                                 + 1.54104 w_k1_FDX_s117
                                 + 1.21712 w_k1_IBM_s117
                                 + 1.64485 w_k1_INTC_s117
                                 + 1.23195 w_k1_KO_s117
                                 + 0.793321 w_k1_MAT_s117
                                 + 1.36949 w_k1_MSFT_s117
                                 + 1.26212 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.08053 w_k1_ADM_s117
                                 + 1.01657 w_k1_AXP_s117
                                 + 0.871585 w_k1_DIS_s117
                                 + 0.943041 w_k1_FDX_s117
                                 + 0.803539 w_k1_IBM_s117
                                 + 1.01674 w_k1_INTC_s117
                                 + 0.894181 w_k1_KO_s117
                                 + 1.01847 w_k1_MAT_s117
                                 + 1.00755 w_k1_MSFT_s117
                                 + 0.937848 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.03079 w_k1_ADM_s117
                                 + 0.60684 w_k1_AXP_s117
                                 + 0.722065 w_k1_DIS_s117
                                 + 0.775594 w_k1_FDX_s117
                                 + 0.746515 w_k1_IBM_s117
                                 + 0.758634 w_k1_INTC_s117
                                 + 0.810784 w_k1_KO_s117
                                 + 0.535482 w_k1_MAT_s117
                                 + 0.77634 w_k1_MSFT_s117
                                 + 0.841504 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.14412 w_k1_ADM_s117
                                 + 1.11114 w_k1_AXP_s117
                                 + 1.75402 w_k1_DIS_s117
                                 + 1.22535 w_k1_FDX_s117
                                 + 1.30792 w_k1_IBM_s117
                                 + 1.1676 w_k1_INTC_s117 + 1.46129 w_k1_KO_s117
                                 + 1.2676 w_k1_MAT_s117
                                 + 1.58062 w_k1_MSFT_s117
                                 + 1.41441 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.14464 w_k1_ADM_s117
                                 + 0.946541 w_k1_AXP_s117
                                 + 1.12985 w_k1_DIS_s117
                                 + 1.07274 w_k1_FDX_s117
                                 + 0.783494 w_k1_IBM_s117
                                 + 1.1737 w_k1_INTC_s117 + 1.0123 w_k1_KO_s117
                                 + 0.603243 w_k1_MAT_s117
                                 + 1.2175 w_k1_MSFT_s117 + 1.10655 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.06498 w_k1_ADM_s117
                                 + 1.10987 w_k1_AXP_s117
                                 + 1.08278 w_k1_DIS_s117
                                 + 0.959004 w_k1_FDX_s117
                                 + 1.47023 w_k1_IBM_s117
                                 + 1.01691 w_k1_INTC_s117
                                 + 1.14005 w_k1_KO_s117 + 1.22359 w_k1_MAT_s117
                                 + 1.11966 w_k1_MSFT_s117
                                 + 1.15908 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.22606 w_k1_ADM_s117
                                 + 1.28077 w_k1_AXP_s117
                                 + 1.21893 w_k1_DIS_s117
                                 + 1.17722 w_k1_FDX_s117 + 1.1799 w_k1_IBM_s117
                                 + 1.04851 w_k1_INTC_s117
                                 + 1.13068 w_k1_KO_s117
                                 + 0.958062 w_k1_MAT_s117
                                 + 1.05431 w_k1_MSFT_s117
                                 + 1.04063 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00985511581428061 w_k1_ADM_s000
                                 + 0.00418266519859269 w_k1_ADM_s001
                                 + 0.0114722130315621 w_k1_ADM_s002
                                 + 0.00631018222399564 w_k1_ADM_s003
                                 + 0.00424648483548023 w_k1_ADM_s004
                                 + 0.00777642548819582 w_k1_ADM_s005
                                 + 0.00616229933121806 w_k1_ADM_s006
                                 + 0.0161432065056591 w_k1_ADM_s007
                                 + 0.0274656436195126 w_k1_ADM_s010
                                 + 0.01036073002912 w_k1_ADM_s011
                                 + 0.0271595782659203 w_k1_ADM_s012
                                 + 0.0159153978412343 w_k1_ADM_s013
                                 + 0.0151120580599766 w_k1_ADM_s014
                                 + 0.0207412630888461 w_k1_ADM_s015
                                 + 0.0327131549575761 w_k1_ADM_s016
                                 + 0.0360124506000562 w_k1_ADM_s017
                                 + 0.039213081790221 w_k1_ADM_s100
                                 + 0.0365416225188828 w_k1_ADM_s101
                                 + 0.0437894429741438 w_k1_ADM_s102
                                 + 0.0392256164207669 w_k1_ADM_s103
                                 + 0.0464895907799173 w_k1_ADM_s104
                                 + 0.079410213398477 w_k1_ADM_s105
                                 + 0.121002947799396 w_k1_ADM_s106
                                 + 0.0837147184063062 w_k1_ADM_s107
                                 + 0.0419577888746459 w_k1_ADM_s110
                                 + 0.0124053682632197 w_k1_ADM_s111
                                 + 0.0441729336979635 w_k1_ADM_s112
                                 + 0.0421391684281568 w_k1_ADM_s113
                                 + 0.0139602360469904 w_k1_ADM_s114
                                 + 0.0419328350263284 w_k1_ADM_s115
                                 + 0.013723498806378 w_k1_ADM_s116
                                 + 0.0552913458847739 w_k1_ADM_s117
                                 + 0.00978920204824812 w_k1_AXP_s000
                                 + 0.00438874999060765 w_k1_AXP_s001
                                 + 0.0113915398923269 w_k1_AXP_s002
                                 + 0.00631651722236096 w_k1_AXP_s003
                                 + 0.00408502307895635 w_k1_AXP_s004
                                 + 0.00977313742861075 w_k1_AXP_s005
                                 + 0.00591273725170386 w_k1_AXP_s006
                                 + 0.0153179391817975 w_k1_AXP_s007
                                 + 0.0242740260434816 w_k1_AXP_s010
                                 + 0.0110770658757095 w_k1_AXP_s011
                                 + 0.0303049550659513 w_k1_AXP_s012
                                 + 0.012950671016824 w_k1_AXP_s013
                                 + 0.0156746144950774 w_k1_AXP_s014
                                 + 0.0218222209855627 w_k1_AXP_s015
                                 + 0.0282588856883123 w_k1_AXP_s016
                                 + 0.0356139809665021 w_k1_AXP_s017
                                 + 0.0443890926709726 w_k1_AXP_s100
                                 + 0.0422564011109116 w_k1_AXP_s101
                                 + 0.0437496710917028 w_k1_AXP_s102
                                 + 0.0383434319815807 w_k1_AXP_s103
                                 + 0.0467092539595656 w_k1_AXP_s104
                                 + 0.0765055663281454 w_k1_AXP_s105
                                 + 0.126025790580256 w_k1_AXP_s106
                                 + 0.0894933489151554 w_k1_AXP_s107
                                 + 0.0390573907360354 w_k1_AXP_s110
                                 + 0.0130670389959577 w_k1_AXP_s111
                                 + 0.0423089471863537 w_k1_AXP_s112
                                 + 0.0540210840194594 w_k1_AXP_s113
                                 + 0.0120014395517623 w_k1_AXP_s114
                                 + 0.0436142800103449 w_k1_AXP_s115
                                 + 0.0138609827521825 w_k1_AXP_s116
                                 + 0.0507429850031685 w_k1_AXP_s117
                                 + 0.00931308221196713 w_k1_DIS_s000
                                 + 0.00392848696533746 w_k1_DIS_s001
                                 + 0.0109619082064506 w_k1_DIS_s002
                                 + 0.00639177730242539 w_k1_DIS_s003
                                 + 0.00407908489605665 w_k1_DIS_s004
                                 + 0.00931252307758852 w_k1_DIS_s005
                                 + 0.00597310515621361 w_k1_DIS_s006
                                 + 0.0142793443982133 w_k1_DIS_s007
                                 + 0.0245689248418709 w_k1_DIS_s010
                                 + 0.010896057758075 w_k1_DIS_s011
                                 + 0.028618394131579 w_k1_DIS_s012
                                 + 0.0134247261215593 w_k1_DIS_s013
                                 + 0.0160560183053021 w_k1_DIS_s014
                                 + 0.0216626899432733 w_k1_DIS_s015
                                 + 0.0328195876270217 w_k1_DIS_s016
                                 + 0.033427931724309 w_k1_DIS_s017
                                 + 0.0462250030221225 w_k1_DIS_s100
                                 + 0.0458865469634434 w_k1_DIS_s101
                                 + 0.0479440897357701 w_k1_DIS_s102
                                 + 0.0392255015761024 w_k1_DIS_s103
                                 + 0.0456611850126969 w_k1_DIS_s104
                                 + 0.0831523064998577 w_k1_DIS_s105
                                 + 0.136602952456266 w_k1_DIS_s106
                                 + 0.0864821740282723 w_k1_DIS_s107
                                 + 0.0420240525369428 w_k1_DIS_s110
                                 + 0.0123230635792108 w_k1_DIS_s111
                                 + 0.0457542263499455 w_k1_DIS_s112
                                 + 0.0516577030110078 w_k1_DIS_s113
                                 + 0.0123610273695639 w_k1_DIS_s114
                                 + 0.0428950336608677 w_k1_DIS_s115
                                 + 0.0126045580547685 w_k1_DIS_s116
                                 + 0.0509699502921689 w_k1_DIS_s117
                                 + 0.00902530792742017 w_k1_FDX_s000
                                 + 0.00403272744520595 w_k1_FDX_s001
                                 + 0.0108764742440895 w_k1_FDX_s002
                                 + 0.00586862499509821 w_k1_FDX_s003
                                 + 0.00405625393708692 w_k1_FDX_s004
                                 + 0.00893867943586641 w_k1_FDX_s005
                                 + 0.00605873095895654 w_k1_FDX_s006
                                 + 0.0150390782585539 w_k1_FDX_s007
                                 + 0.0248839711297374 w_k1_FDX_s010
                                 + 0.0116513832648295 w_k1_FDX_s011
                                 + 0.0294801977661075 w_k1_FDX_s012
                                 + 0.0132602867504594 w_k1_FDX_s013
                                 + 0.0153461121767325 w_k1_FDX_s014
                                 + 0.0205892421325324 w_k1_FDX_s015
                                 + 0.0294410675710797 w_k1_FDX_s016
                                 + 0.0335269558687542 w_k1_FDX_s017
                                 + 0.0433994099319229 w_k1_FDX_s100
                                 + 0.04136792151448 w_k1_FDX_s101
                                 + 0.0461413896962042 w_k1_FDX_s102
                                 + 0.0376000583210471 w_k1_FDX_s103
                                 + 0.0466997724730345 w_k1_FDX_s104
                                 + 0.0767297201459115 w_k1_FDX_s105
                                 + 0.131648695801894 w_k1_FDX_s106
                                 + 0.0818253427015076 w_k1_FDX_s107
                                 + 0.038977486387968 w_k1_FDX_s110
                                 + 0.0136303796842742 w_k1_FDX_s111
                                 + 0.0467195292452285 w_k1_FDX_s112
                                 + 0.0504813817116447 w_k1_FDX_s113
                                 + 0.0117193200026727 w_k1_FDX_s114
                                 + 0.0427434717985182 w_k1_FDX_s115
                                 + 0.0139227286943409 w_k1_FDX_s116
                                 + 0.0498712734832165 w_k1_FDX_s117
                                 + 0.00913625876216337 w_k1_IBM_s000
                                 + 0.00410988860978271 w_k1_IBM_s001
                                 + 0.0107377666943704 w_k1_IBM_s002
                                 + 0.00618858767564008 w_k1_IBM_s003
                                 + 0.00430509269970224 w_k1_IBM_s004
                                 + 0.00898790661765144 w_k1_IBM_s005
                                 + 0.00614510156137143 w_k1_IBM_s006
                                 + 0.0159570460094485 w_k1_IBM_s007
                                 + 0.0255813718512171 w_k1_IBM_s010
                                 + 0.0117926891261689 w_k1_IBM_s011
                                 + 0.0293281931102135 w_k1_IBM_s012
                                 + 0.0133034975318844 w_k1_IBM_s013
                                 + 0.0163183303189452 w_k1_IBM_s014
                                 + 0.020281949268431 w_k1_IBM_s015
                                 + 0.0303699882102621 w_k1_IBM_s016
                                 + 0.0340465823188354 w_k1_IBM_s017
                                 + 0.0468471967418769 w_k1_IBM_s100
                                 + 0.042234348440593 w_k1_IBM_s101
                                 + 0.043803469211923 w_k1_IBM_s102
                                 + 0.0382106690317066 w_k1_IBM_s103
                                 + 0.0489911326026522 w_k1_IBM_s104
                                 + 0.0795078807138426 w_k1_IBM_s105
                                 + 0.134333233567336 w_k1_IBM_s106
                                 + 0.0848435473693265 w_k1_IBM_s107
                                 + 0.0409007971211042 w_k1_IBM_s110
                                 + 0.0123685361397725 w_k1_IBM_s111
                                 + 0.0464920587735392 w_k1_IBM_s112
                                 + 0.0512741583759253 w_k1_IBM_s113
                                 + 0.0119322185679087 w_k1_IBM_s114
                                 + 0.0455135680857589 w_k1_IBM_s115
                                 + 0.0134424595366935 w_k1_IBM_s116
                                 + 0.0518793472806024 w_k1_IBM_s117
                                 + 0.00827896312237605 w_k1_INTC_s000
                                 + 0.00309574146675202 w_k1_INTC_s001
                                 + 0.0112897823914408 w_k1_INTC_s002
                                 + 0.00582753016728316 w_k1_INTC_s003
                                 + 0.00330656019750234 w_k1_INTC_s004
                                 + 0.00900309484647422 w_k1_INTC_s005
                                 + 0.00560666908022114 w_k1_INTC_s006
                                 + 0.0154552138237872 w_k1_INTC_s007
                                 + 0.0259215041074955 w_k1_INTC_s010
                                 + 0.012111577191391 w_k1_INTC_s011
                                 + 0.0284685789255366 w_k1_INTC_s012
                                 + 0.01179297497432 w_k1_INTC_s013
                                 + 0.0134744181987785 w_k1_INTC_s014
                                 + 0.0219976315572232 w_k1_INTC_s015
                                 + 0.0310776963253044 w_k1_INTC_s016
                                 + 0.0348632061626915 w_k1_INTC_s017
                                 + 0.0434378729107028 w_k1_INTC_s100
                                 + 0.0390300984109926 w_k1_INTC_s101
                                 + 0.0440030483554141 w_k1_INTC_s102
                                 + 0.0335661715874698 w_k1_INTC_s103
                                 + 0.0460245900956735 w_k1_INTC_s104
                                 + 0.0790996843706956 w_k1_INTC_s105
                                 + 0.118703380922737 w_k1_INTC_s106
                                 + 0.0814705051470696 w_k1_INTC_s107
                                 + 0.0388522570026098 w_k1_INTC_s110
                                 + 0.0135764391539468 w_k1_INTC_s111
                                 + 0.0515015168967597 w_k1_INTC_s112
                                 + 0.0457108853606044 w_k1_INTC_s113
                                 + 0.0103409298481995 w_k1_INTC_s114
                                 + 0.0459815555112182 w_k1_INTC_s115
                                 + 0.0111117724408183 w_k1_INTC_s116
                                 + 0.0505101203311215 w_k1_INTC_s117
                                 + 0.00867217646326667 w_k1_KO_s000
                                 + 0.00363653158374691 w_k1_KO_s001
                                 + 0.0102603006445726 w_k1_KO_s002
                                 + 0.00572484340389843 w_k1_KO_s003
                                 + 0.00429891532831072 w_k1_KO_s004
                                 + 0.00923175395705254 w_k1_KO_s005
                                 + 0.00541700813887143 w_k1_KO_s006
                                 + 0.0152488510369053 w_k1_KO_s007
                                 + 0.0256236696343879 w_k1_KO_s010
                                 + 0.0109788016011988 w_k1_KO_s011
                                 + 0.029289264689441 w_k1_KO_s012
                                 + 0.0127135536300686 w_k1_KO_s013
                                 + 0.0135378256106893 w_k1_KO_s014
                                 + 0.0232910253264983 w_k1_KO_s015
                                 + 0.0286838141851887 w_k1_KO_s016
                                 + 0.0309350912197623 w_k1_KO_s017
                                 + 0.0446517713851924 w_k1_KO_s100
                                 + 0.044000655621574 w_k1_KO_s101
                                 + 0.0460469807518215 w_k1_KO_s102
                                 + 0.0370252045229962 w_k1_KO_s103
                                 + 0.0450586131824435 w_k1_KO_s104
                                 + 0.0759740495637147 w_k1_KO_s105
                                 + 0.140094245075901 w_k1_KO_s106
                                 + 0.0856741858767352 w_k1_KO_s107
                                 + 0.0349542949195477 w_k1_KO_s110
                                 + 0.0132933638800357 w_k1_KO_s111
                                 + 0.0448247423721277 w_k1_KO_s112
                                 + 0.0480979620813842 w_k1_KO_s113
                                 + 0.012176007694583 w_k1_KO_s114
                                 + 0.0491189309939107 w_k1_KO_s115
                                 + 0.0123014366709085 w_k1_KO_s116
                                 + 0.0503721697332261 w_k1_KO_s117
                                 + 0.0103135831753953 w_k1_MAT_s000
                                 + 0.00454366817013103 w_k1_MAT_s001
                                 + 0.00986261126240206 w_k1_MAT_s002
                                 + 0.00648935678767499 w_k1_MAT_s003
                                 + 0.00450080219005456 w_k1_MAT_s004
                                 + 0.0131623422844015 w_k1_MAT_s005
                                 + 0.00457776781387814 w_k1_MAT_s006
                                 + 0.0170146341695294 w_k1_MAT_s007
                                 + 0.0224296253652498 w_k1_MAT_s010
                                 + 0.0116743285612596 w_k1_MAT_s011
                                 + 0.0365355134705815 w_k1_MAT_s012
                                 + 0.0149161783330418 w_k1_MAT_s013
                                 + 0.0144727697625443 w_k1_MAT_s014
                                 + 0.0230814517018755 w_k1_MAT_s015
                                 + 0.024341130138654 w_k1_MAT_s016
                                 + 0.0354320796761038 w_k1_MAT_s017
                                 + 0.0420148726207691 w_k1_MAT_s100
                                 + 0.0522931065851447 w_k1_MAT_s101
                                 + 0.0454044827698756 w_k1_MAT_s102
                                 + 0.0346832720810015 w_k1_MAT_s103
                                 + 0.0526474913466604 w_k1_MAT_s104
                                 + 0.107163419973441 w_k1_MAT_s105
                                 + 0.131646473544869 w_k1_MAT_s106
                                 + 0.102691927867647 w_k1_MAT_s107
                                 + 0.0415498481632139 w_k1_MAT_s110
                                 + 0.0143945475130749 w_k1_MAT_s111
                                 + 0.0443692981924726 w_k1_MAT_s112
                                 + 0.048144372238545 w_k1_MAT_s113
                                 + 0.0113858497965691 w_k1_MAT_s114
                                 + 0.0495844826613885 w_k1_MAT_s115
                                 + 0.0137641029781386 w_k1_MAT_s116
                                 + 0.043980131212329 w_k1_MAT_s117
                                 + 0.00846592381577561 w_k1_MSFT_s000
                                 + 0.00374715583938246 w_k1_MSFT_s001
                                 + 0.0114879978724125 w_k1_MSFT_s002
                                 + 0.00576585449695878 w_k1_MSFT_s003
                                 + 0.00379292201113381 w_k1_MSFT_s004
                                 + 0.00917371339633291 w_k1_MSFT_s005
                                 + 0.00543245007828299 w_k1_MSFT_s006
                                 + 0.0152958425443173 w_k1_MSFT_s007
                                 + 0.0246397696271214 w_k1_MSFT_s010
                                 + 0.0111234399864823 w_k1_MSFT_s011
                                 + 0.0316354164244887 w_k1_MSFT_s012
                                 + 0.0131973823929089 w_k1_MSFT_s013
                                 + 0.0149525293914488 w_k1_MSFT_s014
                                 + 0.0215968221170726 w_k1_MSFT_s015
                                 + 0.0307890440179717 w_k1_MSFT_s016
                                 + 0.033054910225806 w_k1_MSFT_s017
                                 + 0.0437540583481113 w_k1_MSFT_s100
                                 + 0.04491708991092 w_k1_MSFT_s101
                                 + 0.0434575261374988 w_k1_MSFT_s102
                                 + 0.0372037124584385 w_k1_MSFT_s103
                                 + 0.0453490830118112 w_k1_MSFT_s104
                                 + 0.0805522416975754 w_k1_MSFT_s105
                                 + 0.134959707996198 w_k1_MSFT_s106
                                 + 0.0867840258544401 w_k1_MSFT_s107
                                 + 0.0388944919189472 w_k1_MSFT_s110
                                 + 0.0128973126988991 w_k1_MSFT_s111
                                 + 0.0484059947941025 w_k1_MSFT_s112
                                 + 0.0520145586309835 w_k1_MSFT_s113
                                 + 0.0118976344941333 w_k1_MSFT_s114
                                 + 0.0448490140640171 w_k1_MSFT_s115
                                 + 0.0136166824767356 w_k1_MSFT_s116
                                 + 0.0519935312932054 w_k1_MSFT_s117
                                 + 0.00881598388468912 w_k1_PG_s000
                                 + 0.00395145102908669 w_k1_PG_s001
                                 + 0.00964073955253672 w_k1_PG_s002
                                 + 0.00553136554250045 w_k1_PG_s003
                                 + 0.00414495349148021 w_k1_PG_s004
                                 + 0.00887995309293622 w_k1_PG_s005
                                 + 0.00559611188174138 w_k1_PG_s006
                                 + 0.0157894083072535 w_k1_PG_s007
                                 + 0.0254449239103848 w_k1_PG_s010
                                 + 0.0119681819775459 w_k1_PG_s011
                                 + 0.0265448536906444 w_k1_PG_s012
                                 + 0.0123269065259614 w_k1_PG_s013
                                 + 0.0148757078654672 w_k1_PG_s014
                                 + 0.0184912311172886 w_k1_PG_s015
                                 + 0.0300681508472138 w_k1_PG_s016
                                 + 0.0343901141968699 w_k1_PG_s017
                                 + 0.0436047217261762 w_k1_PG_s100
                                 + 0.0418428015483617 w_k1_PG_s101
                                 + 0.0420750344890019 w_k1_PG_s102
                                 + 0.0379520195612574 w_k1_PG_s103
                                 + 0.0492595894927303 w_k1_PG_s104
                                 + 0.0810170011266002 w_k1_PG_s105
                                 + 0.138216265368429 w_k1_PG_s106
                                 + 0.0864811266509743 w_k1_PG_s107
                                 + 0.0392766639575214 w_k1_PG_s110
                                 + 0.0132795990951254 w_k1_PG_s111
                                 + 0.0440859963025557 w_k1_PG_s112
                                 + 0.0468859422466994 w_k1_PG_s113
                                 + 0.0119275711077264 w_k1_PG_s114
                                 + 0.048196508247183 w_k1_PG_s115
                                 + 0.0129340787327136 w_k1_PG_s116
                                 + 0.0515031309040011 w_k1_PG_s117 >= 1.11
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
