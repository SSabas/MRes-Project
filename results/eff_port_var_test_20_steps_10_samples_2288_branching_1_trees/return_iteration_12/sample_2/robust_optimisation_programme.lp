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
 w_k1_ADM_s0_return_constraint:  1.11336 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.946328 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.20825 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.06807 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.853761 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.08001 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.767223 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.887754 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.02107 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.991916 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.09944 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.652254 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.645238 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.35823 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.959531 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.08173 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.46839 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.02222 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.826434 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.38495 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.796694 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.17071 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.20998 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.911082 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.05287 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.804031 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.13685 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.41246 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.875328 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.0794 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.752382 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.54994 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.0829 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.947763 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.836504 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.14738 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.871471 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.15636 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.22328 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.890092 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.05027 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.05124 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.833585 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.11312 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.791907 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.06452 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.15884 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.950431 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.52564 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.79647 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.559925 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.70209 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.75387 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.28056 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.72851 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.921034 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.92299 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.5643 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.732082 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.14104 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.993638 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.806761 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.20603 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.603858 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.949482 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.47337 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.70552 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.831477 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.774487 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.81577 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.26929 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.955893 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.01697 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.38466 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.781035 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.08833 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.3395 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.899781 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.13929 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.01308 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.741482 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.18986 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.879776 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.978747 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.711741 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.00901 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.31312 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.627761 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.524152 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.92721 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.656291 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.3618 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.7657 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.949545 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.800377 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.68746 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.658588 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.18473 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.19305 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.701756 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.9733 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.727765 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.08329 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.70315 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.733752 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.02106 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.643821 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.89948 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.16845 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.11734 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.72052 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.24037 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.758848 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.1714 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.23172 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.939794 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.09677 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.909941 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.845333 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.08356 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.852838 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.924757 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.984579 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.08086 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.18847 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.749179 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.711731 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.37153 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.853302 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.16575 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.60451 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.00533 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.872005 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.30073 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.767737 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.20091 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.22492 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.826931 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.0653 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.817724 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.01682 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.43276 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.712146 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.12404 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.71767 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.49114 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.16784 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.977865 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.795182 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.03273 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.880172 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.00252 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18571 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.976035 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.890395 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.24225 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.688674 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.15773 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.741828 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.19745 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.70519 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.636305 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.20922 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.852236 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.513769 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.35218 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.637008 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.39501 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.16835 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.846306 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.1821 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.94616 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.913448 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.828916 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.816506 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.740724 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.973792 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.729359 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.28842 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.24725 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.29217 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.705987 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.871462 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.27722 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.893909 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.935371 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.28439 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 2.02443 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.589835 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.36508 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.17539 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.898412 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.03802 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.04949 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.972763 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.15738 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.934293 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.894757 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.0364 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.16461 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.00463 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.74671 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.759546 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.2969 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.793891 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.10105 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.37285 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.869991 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.854133 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.13717 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.791406 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.0044 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.07386 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.807763 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.9267 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.883162 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.09322 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.33887 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.793292 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.935318 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.825257 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.15514 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.12025 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.977954 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.89006 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.21143 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.753782 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.04959 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.25889 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.958475 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.979775 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.04365 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.746486 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.09146 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.838821 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.02554 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.778297 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.898295 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.11267 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.795181 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.54028 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.51893 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.711587 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.38152 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.37004 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.876888 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.978195 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.66102 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.787274 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.01253 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.978697 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.744182 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.934306 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.833793 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.14768 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.33226 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.03015 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.893784 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.804718 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.51674 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.08347 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.03352 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.946359 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.53909 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.758708 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.20672 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.19564 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.924066 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.875378 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.26627 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.671122 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.26108 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.967723 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.40021 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.826125 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.663586 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.31733 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.00453 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.475949 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.51634 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.555988 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.62589 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.26477 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.872624 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.25418 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 2.21608 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.780405 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.921456 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.811098 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.75624 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.26047 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.761128 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.48972 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.14757 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.21918 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.671464 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.873798 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.46004 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.788636 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.915939 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.2141 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.84871 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.629686 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.27511 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19015 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.912604 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.07024 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.967371 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.916025 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.07346 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.957188 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.03989 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.818263 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.07625 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.16208 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.730675 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.774776 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.41117 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.850953 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.1605 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.41812 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.07103 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.962832 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.21692 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.795284 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.16739 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.05228 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.799747 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.907102 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.826738 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.08602 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.32271 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.792058 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.04645 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.811746 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.45585 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.1915 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.09214 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.8396 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.22166 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.973007 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.01971 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14815 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.957932 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.01944 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.93265 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.870464 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.12083 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.916193 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.02503 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.992001 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.908899 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.06067 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.851281 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.73881 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.31078 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.892652 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.27941 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.29823 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.02606 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.03928 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.41655 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.90869 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.2186 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.05352 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.82536 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.936 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.86488 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.08842 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.17569 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.918847 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.922861 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.797604 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.2316 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.04585 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.957458 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.846652 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.15292 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.754132 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.01628 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.179000516370085 z_k1_s0000
                                 - 0.225999407540118 z_k1_s0001
                                 - 0.114999998394251 z_k1_s0002
                                 - 0.264000550400612 z_k1_s0003
                                 - 0.160000619939711 z_k1_s0004
                                 - 0.152000521442738 z_k1_s0005
                                 - 0.144999422758077 z_k1_s0006
                                 - 0.108999924521522 z_k1_s0007
                                 - 0.790999166443218 z_k1_s0010
                                 - 0.368999915401311 z_k1_s0011
                                 - 0.215999959722714 z_k1_s0012
                                 - 0.400000206153045 z_k1_s0013
                                 - 0.313999839680289 z_k1_s0014
                                 - 0.943000836580644 z_k1_s0015
                                 - 0.442000966848776 z_k1_s0016
                                 - 0.315000020220589 z_k1_s0017
                                 - 0.144999697131524 z_k1_s0020
                                 - 0.124000331554124 z_k1_s0021
                                 - 0.0949996393274762 z_k1_s0022
                                 - 0.113000036537105 z_k1_s0023
                                 - 0.174000389358172 z_k1_s0024
                                 - 0.118000512817725 z_k1_s0025
                                 - 0.067000077457424 z_k1_s0026
                                 - 0.104999650888286 z_k1_s0027
                                 - 0.317000741786562 z_k1_s0030
                                 - 0.139999987528357 z_k1_s0031
                                 - 0.292999544215658 z_k1_s0032
                                 - 0.153000747887627 z_k1_s0033
                                 - 0.139999987528357 z_k1_s0034
                                 - 0.137000098407065 z_k1_s0035
                                 - 0.197999829207206 z_k1_s0036
                                 - 0.111000066688933 z_k1_s0037
                                 - 0.277000595419455 z_k1_s0040
                                 - 0.361001866207814 z_k1_s0041
                                 - 0.439000281755422 z_k1_s0042
                                 - 0.477002262412663 z_k1_s0043
                                 - 0.169000333694479 z_k1_s0044
                                 - 0.357001657717578 z_k1_s0045
                                 - 0.229000353541421 z_k1_s0046
                                 - 0.516001470186467 z_k1_s0047
                                 - 0.55199942621345 z_k1_s0050
                                 - 0.212999763286736 z_k1_s0051
                                 - 0.575999809439462 z_k1_s0052
                                 - 0.164999700933574 z_k1_s0053
                                 - 0.181999698569109 z_k1_s0054
                                 - 0.238999317882973 z_k1_s0055
                                 - 0.154999854188897 z_k1_s0056
                                 - 0.264999341878452 z_k1_s0057
                                 - 0.185000111164388 z_k1_s0060
                                 - 0.124000023731794 z_k1_s0061
                                 - 0.0880000601322511 z_k1_s0062
                                 - 0.115000032831908 z_k1_s0063
                                 - 0.340999595562324 z_k1_s0064
                                 - 0.208000654530895 z_k1_s0065
                                 - 0.0940000093321645 z_k1_s0066
                                 - 0.186999691630932 z_k1_s0067
                                 - 0.172000405874102 z_k1_s0070
                                 - 0.182000497290208 z_k1_s0071
                                 - 0.204000573425328 z_k1_s0072
                                 - 0.485000684271754 z_k1_s0073
                                 - 0.459000613230295 z_k1_s0074
                                 - 0.0980001876560655 z_k1_s0075
                                 - 0.385001853115906 z_k1_s0076
                                 - 0.0980001876560655 z_k1_s0077
                                 - 0.295999845817725 z_k1_s0100
                                 - 0.243999789006569 z_k1_s0101
                                 - 0.337999638873088 z_k1_s0102
                                 - 0.692999324411339 z_k1_s0103
                                 - 0.433999687570651 z_k1_s0104
                                 - 0.435000334785743 z_k1_s0105
                                 - 0.374998021851556 z_k1_s0106
                                 - 0.836997571459115 z_k1_s0107
                                 - 0.127000412269118 z_k1_s0110
                                 - 0.351002140871441 z_k1_s0111
                                 - 0.2990001761444 z_k1_s0112
                                 - 0.103000407598026 z_k1_s0113
                                 - 0.19400061065985 z_k1_s0114
                                 - 0.512002391257392 z_k1_s0115
                                 - 0.223001037555453 z_k1_s0116
                                 - 0.213000193106431 z_k1_s0117
                                 - 0.105999998372116 z_k1_s0120
                                 - 0.191999843089165 z_k1_s0121
                                 - 0.0629999944136176 z_k1_s0122
                                 - 0.116999674882533 z_k1_s0123
                                 - 0.0959999215445824 z_k1_s0124
                                 - 0.0859998447170487 z_k1_s0125
                                 - 0.0440000116412518 z_k1_s0126
                                 - 0.112000044468636 z_k1_s0127
                                 - 0.280000149987708 z_k1_s0130
                                 - 0.393001106977929 z_k1_s0131
                                 - 0.284000600360123 z_k1_s0132
                                 - 0.299000328251596 z_k1_s0133
                                 - 0.131000434359479 z_k1_s0134
                                 - 0.0880002992682192 z_k1_s0135
                                 - 0.205999662118366 z_k1_s0136
                                 - 0.280000149987708 z_k1_s0137
                                 - 0.262999908056144 z_k1_s0140
                                 - 0.223999170845966 z_k1_s0141
                                 - 0.347999299564528 z_k1_s0142
                                 - 0.317999998326489 z_k1_s0143
                                 - 0.156999977180678 z_k1_s0144
                                 - 0.283000127881806 z_k1_s0145
                                 - 0.311000024237552 z_k1_s0146
                                 - 0.153000138915636 z_k1_s0147
                                 - 0.123999577731613 z_k1_s0150
                                 - 0.279000015895769 z_k1_s0151
                                 - 0.11299973983602 z_k1_s0152
                                 - 0.0569999679314069 z_k1_s0153
                                 - 0.100000282686552 z_k1_s0154
                                 - 0.0579999707582724 z_k1_s0155
                                 - 0.0629999848926001 z_k1_s0156
                                 - 0.172000099821014 z_k1_s0157
                                 - 0.0749999398274731 z_k1_s0160
                                 - 0.36199947310864 z_k1_s0161
                                 - 0.317999911780461 z_k1_s0162
                                 - 0.119999929803953 z_k1_s0163
                                 - 0.141000194619603 z_k1_s0164
                                 - 0.0729999953307323 z_k1_s0165
                                 - 0.129000006039237 z_k1_s0166
                                 - 0.085999960559498 z_k1_s0167
                                 - 0.409997917378213 z_k1_s0170
                                 - 0.0939997225226334 z_k1_s0171
                                 - 0.151999409303028 z_k1_s0172
                                 - 0.435997869031187 z_k1_s0173
                                 - 0.164999385129515 z_k1_s0174
                                 - 0.229998692163851 z_k1_s0175
                                 - 0.166999440087241 z_k1_s0176
                                 - 0.25299894277897 z_k1_s0177
                                 - 0.178999937509259 z_k1_s1000
                                 - 0.0999998558746054 z_k1_s1001
                                 - 0.258999715089099 z_k1_s1002
                                 - 0.0659998834532708 z_k1_s1003
                                 - 0.183999756233243 z_k1_s1004
                                 - 0.101999917264004 z_k1_s1005
                                 - 0.222999413478756 z_k1_s1006
                                 - 0.226000108111977 z_k1_s1007
                                 - 0.400999554745707 z_k1_s1010
                                 - 0.329000464190732 z_k1_s1011
                                 - 0.550000918048398 z_k1_s1012
                                 - 0.272000199109278 z_k1_s1013
                                 - 0.248000044090752 z_k1_s1014
                                 - 0.329000464190732 z_k1_s1015
                                 - 0.205999978983422 z_k1_s1016
                                 - 0.413999581443217 z_k1_s1017
                                 - 0.16799966683994 z_k1_s1020
                                 - 0.257999416682885 z_k1_s1021
                                 - 0.270998733850174 z_k1_s1022
                                 - 0.213000545259727 z_k1_s1023
                                 - 0.12999979476177 z_k1_s1024
                                 - 0.309998692348671 z_k1_s1025
                                 - 0.375998776499715 z_k1_s1026
                                 - 0.280999598053448 z_k1_s1027
                                 - 0.090999790286146 z_k1_s1030
                                 - 0.295999047471808 z_k1_s1031
                                 - 0.344999809193863 z_k1_s1032
                                 - 0.251999390577061 z_k1_s1033
                                 - 0.137999737277812 z_k1_s1034
                                 - 0.209999802813772 z_k1_s1035
                                 - 0.262998838801416 z_k1_s1036
                                 - 0.26899904619579 z_k1_s1037
                                 - 0.452999839250101 z_k1_s1040
                                 - 0.725999837127983 z_k1_s1041
                                 - 0.901000752934386 z_k1_s1042
                                 - 0.307000285865991 z_k1_s1043
                                 - 0.891001801202672 z_k1_s1044
                                 - 0.994997502812979 z_k1_s1045
                                 - 0.852002037348706 z_k1_s1046
                                 - 0.377999997061683 z_k1_s1047
                                 - 0.246999430779472 z_k1_s1050
                                 - 0.420999167438775 z_k1_s1051
                                 - 0.222999350919842 z_k1_s1052
                                 - 0.489999205598162 z_k1_s1053
                                 - 0.29399921705962 z_k1_s1054
                                 - 0.302999208719572 z_k1_s1055
                                 - 0.848997273360003 z_k1_s1056
                                 - 0.235999304839853 z_k1_s1057
                                 - 0.347999865505665 z_k1_s1060
                                 - 0.302999163176935 z_k1_s1061
                                 - 0.243999366346981 z_k1_s1062
                                 - 0.166999471404983 z_k1_s1063
                                 - 0.709997606716236 z_k1_s1064
                                 - 0.42099967093408 z_k1_s1065
                                 - 0.609998415867453 z_k1_s1066
                                 - 0.243999366346981 z_k1_s1067
                                 - 0.119999815131455 z_k1_s1070
                                 - 0.127999888486767 z_k1_s1071
                                 - 0.0819999307930821 z_k1_s1072
                                 - 0.320000078216425 z_k1_s1073
                                 - 0.374999422285797 z_k1_s1074
                                 - 0.18100017811598 z_k1_s1075
                                 - 0.0819999307930821 z_k1_s1076
                                 - 0.139999784320031 z_k1_s1077
                                 - 1.50700349611597 z_k1_s1100
                                 - 1.27200103621568 z_k1_s1101
                                 - 1.51700406707038 z_k1_s1102
                                 - 0.945999955147277 z_k1_s1103
                                 - 0.692001523946869 z_k1_s1104
                                 - 0.476000879361859 z_k1_s1105
                                 - 0.521001257151034 z_k1_s1106
                                 - 0.374000899642002 z_k1_s1107
                                 - 0.412000128352017 z_k1_s1110
                                 - 1.07000200899223 z_k1_s1111
                                 - 0.623998462544451 z_k1_s1112
                                 - 0.386000263879346 z_k1_s1113
                                 - 0.574999476607648 z_k1_s1114
                                 - 0.315999936083553 z_k1_s1115
                                 - 0.270999939757729 z_k1_s1116
                                 - 0.633998080572334 z_k1_s1117
                                 - 0.918999956772466 z_k1_s1120
                                 - 0.60099936251197 z_k1_s1121
                                 - 2.59399955195746 z_k1_s1122
                                 - 1.44300383136081 z_k1_s1123
                                 - 0.531000084491986 z_k1_s1124
                                 - 1.34399838517827 z_k1_s1125
                                 - 1.50899652715401 z_k1_s1126
                                 - 1.99399636750461 z_k1_s1127
                                 - 0.555999183126377 z_k1_s1130
                                 - 0.423999409584084 z_k1_s1131
                                 - 0.484999018102635 z_k1_s1132
                                 - 0.755000481367456 z_k1_s1133
                                 - 1.54899844278369 z_k1_s1134
                                 - 0.513999332240053 z_k1_s1135
                                 - 0.619998336987616 z_k1_s1136
                                 - 0.747998905105982 z_k1_s1137
                                 - 0.527002277503487 z_k1_s1140
                                 - 0.637000652673458 z_k1_s1141
                                 - 0.425000869029388 z_k1_s1142
                                 - 1.50400590108623 z_k1_s1143
                                 - 0.429000903981262 z_k1_s1144
                                 - 0.441001008836884 z_k1_s1145
                                 - 0.648999723126454 z_k1_s1146
                                 - 0.559999721249228 z_k1_s1147
                                 - 0.538998968394224 z_k1_s1150
                                 - 0.440000158791553 z_k1_s1151
                                 - 0.180000117414745 z_k1_s1152
                                 - 0.365999882926523 z_k1_s1153
                                 - 0.211000030250024 z_k1_s1154
                                 - 0.465001577530203 z_k1_s1155
                                 - 0.435000452044025 z_k1_s1156
                                 - 0.249000109532045 z_k1_s1157
                                 - 1.05400057037239 z_k1_s1160
                                 - 0.606001079738226 z_k1_s1161
                                 - 1.30500051720082 z_k1_s1162
                                 - 1.00799866947659 z_k1_s1163
                                 - 0.344000517956206 z_k1_s1164
                                 - 1.28900498854932 z_k1_s1165
                                 - 0.921004537405629 z_k1_s1166
                                 - 0.851001965521204 z_k1_s1167
                                 - 0.352000089431558 z_k1_s1170
                                 - 0.393000177132454 z_k1_s1171
                                 - 1.0240001777281 z_k1_s1172
                                 - 1.00500270266875 z_k1_s1173
                                 - 0.669998779110101 z_k1_s1174
                                 - 0.379000080834535 z_k1_s1175
                                 - 0.379000080834535 z_k1_s1176
                                 - 0.332000146177541 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.23724 w_k1_ADM_s000
                                 + 1.01067 w_k1_AXP_s000
                                 + 1.54818 w_k1_DIS_s000
                                 + 1.22006 w_k1_FDX_s000
                                 + 1.50679 w_k1_IBM_s000
                                 + 1.02363 w_k1_INTC_s000
                                 + 1.35248 w_k1_KO_s000 + 1.21855 w_k1_MAT_s000
                                 + 1.24396 w_k1_MSFT_s000
                                 + 1.19767 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.81155 w_k1_ADM_s000
                                 + 0.653994 w_k1_AXP_s000
                                 + 0.665852 w_k1_DIS_s000
                                 + 0.792626 w_k1_FDX_s000
                                 + 0.758976 w_k1_IBM_s000
                                 + 0.862136 w_k1_INTC_s000
                                 + 0.693925 w_k1_KO_s000
                                 + 0.420401 w_k1_MAT_s000
                                 + 0.841886 w_k1_MSFT_s000
                                 + 0.7539 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.888694 w_k1_ADM_s000
                                 + 0.990069 w_k1_AXP_s000
                                 + 1.28392 w_k1_DIS_s000
                                 + 1.15191 w_k1_FDX_s000
                                 + 0.979528 w_k1_IBM_s000
                                 + 1.16014 w_k1_INTC_s000
                                 + 1.15119 w_k1_KO_s000
                                 + 0.858612 w_k1_MAT_s000
                                 + 1.17988 w_k1_MSFT_s000
                                 + 0.936841 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.06216 w_k1_ADM_s000
                                 + 0.862181 w_k1_AXP_s000
                                 + 0.917961 w_k1_DIS_s000
                                 + 0.968658 w_k1_FDX_s000
                                 + 0.96527 w_k1_IBM_s000
                                 + 0.944453 w_k1_INTC_s000
                                 + 0.932468 w_k1_KO_s000
                                 + 0.63546 w_k1_MAT_s000
                                 + 0.931969 w_k1_MSFT_s000
                                 + 0.987501 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.35655 w_k1_ADM_s000
                                 + 1.00367 w_k1_AXP_s000
                                 + 1.01908 w_k1_DIS_s000
                                 + 0.953883 w_k1_FDX_s000
                                 + 1.22781 w_k1_IBM_s000
                                 + 0.935383 w_k1_INTC_s000
                                 + 1.13574 w_k1_KO_s000
                                 + 0.751843 w_k1_MAT_s000
                                 + 0.892911 w_k1_MSFT_s000
                                 + 1.07482 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.32324 w_k1_ADM_s000
                                 + 0.974661 w_k1_AXP_s000
                                 + 0.958725 w_k1_DIS_s000
                                 + 1.11713 w_k1_FDX_s000
                                 + 0.821391 w_k1_IBM_s000
                                 + 0.976341 w_k1_INTC_s000
                                 + 0.968222 w_k1_KO_s000
                                 + 0.69568 w_k1_MAT_s000
                                 + 1.0454 w_k1_MSFT_s000
                                 + 0.952089 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.942137 w_k1_ADM_s000
                                 + 0.825986 w_k1_AXP_s000
                                 + 0.802242 w_k1_DIS_s000
                                 + 0.845149 w_k1_FDX_s000
                                 + 0.916777 w_k1_IBM_s000
                                 + 0.801415 w_k1_INTC_s000
                                 + 0.920667 w_k1_KO_s000
                                 + 0.716753 w_k1_MAT_s000
                                 + 0.874977 w_k1_MSFT_s000
                                 + 0.92474 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.16206 w_k1_ADM_s000
                                 + 1.54923 w_k1_AXP_s000
                                 + 1.17964 w_k1_DIS_s000
                                 + 1.25868 w_k1_FDX_s000
                                 + 0.979569 w_k1_IBM_s000
                                 + 0.930717 w_k1_INTC_s000
                                 + 1.00847 w_k1_KO_s000 + 1.04058 w_k1_MAT_s000
                                 + 0.993231 w_k1_MSFT_s000
                                 + 1.14424 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.06803 w_k1_ADM_s001
                                 + 1.18154 w_k1_AXP_s001
                                 + 1.24212 w_k1_DIS_s001
                                 + 1.12757 w_k1_FDX_s001
                                 + 0.99644 w_k1_IBM_s001
                                 + 1.33168 w_k1_INTC_s001
                                 + 1.09712 w_k1_KO_s001
                                 + 0.990733 w_k1_MAT_s001
                                 + 1.06952 w_k1_MSFT_s001
                                 + 1.00249 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.32261 w_k1_ADM_s001
                                 + 1.41501 w_k1_AXP_s001
                                 + 1.58289 w_k1_DIS_s001
                                 + 1.34438 w_k1_FDX_s001
                                 + 1.47862 w_k1_IBM_s001
                                 + 1.37413 w_k1_INTC_s001
                                 + 1.46547 w_k1_KO_s001 + 1.33806 w_k1_MAT_s001
                                 + 1.25844 w_k1_MSFT_s001
                                 + 1.14693 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.04595 w_k1_ADM_s001
                                 + 1.04849 w_k1_AXP_s001
                                 + 1.16555 w_k1_DIS_s001
                                 + 0.967531 w_k1_FDX_s001
                                 + 1.26388 w_k1_IBM_s001
                                 + 1.24674 w_k1_INTC_s001
                                 + 1.05884 w_k1_KO_s001
                                 + 0.938329 w_k1_MAT_s001
                                 + 0.989504 w_k1_MSFT_s001
                                 + 1.00051 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.860055 w_k1_ADM_s001
                                 + 1.13812 w_k1_AXP_s001
                                 + 0.958773 w_k1_DIS_s001
                                 + 0.899195 w_k1_FDX_s001
                                 + 1.42349 w_k1_IBM_s001
                                 + 1.11711 w_k1_INTC_s001
                                 + 1.14114 w_k1_KO_s001 + 1.15481 w_k1_MAT_s001
                                 + 0.817354 w_k1_MSFT_s001
                                 + 0.930898 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.711831 w_k1_ADM_s001
                                 + 0.574043 w_k1_AXP_s001
                                 + 0.587785 w_k1_DIS_s001
                                 + 0.800489 w_k1_FDX_s001
                                 + 0.685078 w_k1_IBM_s001
                                 + 0.8396 w_k1_INTC_s001
                                 + 0.652246 w_k1_KO_s001
                                 + 0.587123 w_k1_MAT_s001
                                 + 0.711021 w_k1_MSFT_s001
                                 + 0.803399 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.9368 w_k1_ADM_s001
                                 + 0.866976 w_k1_AXP_s001
                                 + 0.852121 w_k1_DIS_s001
                                 + 0.906525 w_k1_FDX_s001
                                 + 0.901788 w_k1_IBM_s001
                                 + 1.00658 w_k1_INTC_s001
                                 + 0.846267 w_k1_KO_s001
                                 + 0.78248 w_k1_MAT_s001
                                 + 0.733778 w_k1_MSFT_s001
                                 + 0.87474 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.04884 w_k1_ADM_s001
                                 + 1.1136 w_k1_AXP_s001
                                 + 0.837131 w_k1_DIS_s001
                                 + 1.00015 w_k1_FDX_s001
                                 + 0.819947 w_k1_IBM_s001
                                 + 1.17951 w_k1_INTC_s001
                                 + 0.889749 w_k1_KO_s001
                                 + 0.85047 w_k1_MAT_s001
                                 + 0.86806 w_k1_MSFT_s001
                                 + 0.898836 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.17945 w_k1_ADM_s001
                                 + 1.27574 w_k1_AXP_s001
                                 + 1.53215 w_k1_DIS_s001
                                 + 1.46867 w_k1_FDX_s001
                                 + 0.882201 w_k1_IBM_s001
                                 + 1.28449 w_k1_INTC_s001
                                 + 1.26463 w_k1_KO_s001
                                 + 0.933139 w_k1_MAT_s001
                                 + 1.21441 w_k1_MSFT_s001
                                 + 1.25317 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.20488 w_k1_ADM_s002
                                 + 1.26866 w_k1_AXP_s002
                                 + 1.42345 w_k1_DIS_s002
                                 + 1.31925 w_k1_FDX_s002
                                 + 1.34383 w_k1_IBM_s002
                                 + 1.28645 w_k1_INTC_s002
                                 + 1.21692 w_k1_KO_s002 + 1.11954 w_k1_MAT_s002
                                 + 1.41102 w_k1_MSFT_s002
                                 + 1.23605 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.992712 w_k1_ADM_s002
                                 + 0.988938 w_k1_AXP_s002
                                 + 1.37044 w_k1_DIS_s002
                                 + 1.09866 w_k1_FDX_s002
                                 + 1.32435 w_k1_IBM_s002
                                 + 1.02043 w_k1_INTC_s002
                                 + 1.19669 w_k1_KO_s002 + 1.48667 w_k1_MAT_s002
                                 + 1.23864 w_k1_MSFT_s002
                                 + 0.943523 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.979691 w_k1_ADM_s002
                                 + 1.08971 w_k1_AXP_s002
                                 + 0.917326 w_k1_DIS_s002
                                 + 0.927977 w_k1_FDX_s002
                                 + 1.02792 w_k1_IBM_s002
                                 + 0.78743 w_k1_INTC_s002
                                 + 0.862806 w_k1_KO_s002
                                 + 1.32599 w_k1_MAT_s002
                                 + 0.865452 w_k1_MSFT_s002
                                 + 0.765216 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.17594 w_k1_ADM_s002
                                 + 1.40869 w_k1_AXP_s002
                                 + 1.20797 w_k1_DIS_s002
                                 + 1.13927 w_k1_FDX_s002
                                 + 1.16199 w_k1_IBM_s002
                                 + 0.979246 w_k1_INTC_s002
                                 + 1.17307 w_k1_KO_s002 + 1.85172 w_k1_MAT_s002
                                 + 1.28059 w_k1_MSFT_s002
                                 + 1.14013 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.866505 w_k1_ADM_s002
                                 + 0.842843 w_k1_AXP_s002
                                 + 1.00902 w_k1_DIS_s002
                                 + 0.955833 w_k1_FDX_s002
                                 + 0.945797 w_k1_IBM_s002
                                 + 0.854561 w_k1_INTC_s002
                                 + 0.992322 w_k1_KO_s002
                                 + 1.16831 w_k1_MAT_s002
                                 + 1.15594 w_k1_MSFT_s002
                                 + 0.881139 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.750054 w_k1_ADM_s002
                                 + 0.614858 w_k1_AXP_s002
                                 + 0.623052 w_k1_DIS_s002
                                 + 0.771821 w_k1_FDX_s002
                                 + 0.748353 w_k1_IBM_s002
                                 + 0.68552 w_k1_INTC_s002
                                 + 0.705884 w_k1_KO_s002
                                 + 0.741715 w_k1_MAT_s002
                                 + 0.820887 w_k1_MSFT_s002
                                 + 0.783766 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.02343 w_k1_ADM_s002
                                 + 1.0107 w_k1_AXP_s002 + 1.00092 w_k1_DIS_s002
                                 + 1.14076 w_k1_FDX_s002
                                 + 0.721754 w_k1_IBM_s002
                                 + 0.984373 w_k1_INTC_s002
                                 + 0.777635 w_k1_KO_s002
                                 + 0.696688 w_k1_MAT_s002
                                 + 1.27333 w_k1_MSFT_s002
                                 + 0.873077 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.911204 w_k1_ADM_s002
                                 + 0.661939 w_k1_AXP_s002
                                 + 0.886075 w_k1_DIS_s002
                                 + 0.843774 w_k1_FDX_s002
                                 + 0.954226 w_k1_IBM_s002
                                 + 0.699369 w_k1_INTC_s002
                                 + 0.809412 w_k1_KO_s002
                                 + 0.712681 w_k1_MAT_s002
                                 + 0.909134 w_k1_MSFT_s002
                                 + 0.852621 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.10481 w_k1_ADM_s003
                                 + 1.10683 w_k1_AXP_s003
                                 + 0.964734 w_k1_DIS_s003
                                 + 1.07017 w_k1_FDX_s003
                                 + 1.06505 w_k1_IBM_s003
                                 + 0.985432 w_k1_INTC_s003
                                 + 0.954365 w_k1_KO_s003
                                 + 1.33871 w_k1_MAT_s003
                                 + 1.00621 w_k1_MSFT_s003
                                 + 1.10246 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.31105 w_k1_ADM_s003
                                 + 1.3112 w_k1_AXP_s003 + 1.60755 w_k1_DIS_s003
                                 + 1.40385 w_k1_FDX_s003
                                 + 1.27064 w_k1_IBM_s003
                                 + 1.03981 w_k1_INTC_s003
                                 + 1.24564 w_k1_KO_s003 + 1.23791 w_k1_MAT_s003
                                 + 1.51482 w_k1_MSFT_s003
                                 + 1.51149 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.898322 w_k1_ADM_s003
                                 + 0.961013 w_k1_AXP_s003
                                 + 0.895552 w_k1_DIS_s003
                                 + 0.940289 w_k1_FDX_s003
                                 + 1.17926 w_k1_IBM_s003
                                 + 0.957639 w_k1_INTC_s003
                                 + 1.01493 w_k1_KO_s003 + 1.25112 w_k1_MAT_s003
                                 + 0.996643 w_k1_MSFT_s003
                                 + 1.13027 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.786343 w_k1_ADM_s003
                                 + 0.665483 w_k1_AXP_s003
                                 + 0.695267 w_k1_DIS_s003
                                 + 0.845619 w_k1_FDX_s003
                                 + 0.716284 w_k1_IBM_s003
                                 + 0.783621 w_k1_INTC_s003
                                 + 0.716837 w_k1_KO_s003
                                 + 0.62592 w_k1_MAT_s003
                                 + 0.799501 w_k1_MSFT_s003
                                 + 0.980701 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.06296 w_k1_ADM_s003
                                 + 1.11764 w_k1_AXP_s003
                                 + 1.38918 w_k1_DIS_s003
                                 + 1.10947 w_k1_FDX_s003 + 1.5562 w_k1_IBM_s003
                                 + 1.03077 w_k1_INTC_s003 + 1.2523 w_k1_KO_s003
                                 + 1.39146 w_k1_MAT_s003
                                 + 1.15665 w_k1_MSFT_s003 + 1.0511 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.687705 w_k1_ADM_s003
                                 + 0.677383 w_k1_AXP_s003
                                 + 0.580739 w_k1_DIS_s003
                                 + 0.692242 w_k1_FDX_s003
                                 + 0.943373 w_k1_IBM_s003
                                 + 0.595013 w_k1_INTC_s003
                                 + 0.643919 w_k1_KO_s003
                                 + 0.640884 w_k1_MAT_s003
                                 + 0.669558 w_k1_MSFT_s003
                                 + 0.895803 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.927762 w_k1_ADM_s003
                                 + 0.951721 w_k1_AXP_s003
                                 + 0.803067 w_k1_DIS_s003
                                 + 0.943779 w_k1_FDX_s003
                                 + 0.865197 w_k1_IBM_s003
                                 + 0.941916 w_k1_INTC_s003
                                 + 0.757208 w_k1_KO_s003
                                 + 1.03032 w_k1_MAT_s003
                                 + 0.932714 w_k1_MSFT_s003
                                 + 0.878049 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.26761 w_k1_ADM_s003
                                 + 1.16403 w_k1_AXP_s003
                                 + 1.22645 w_k1_DIS_s003
                                 + 1.27223 w_k1_FDX_s003
                                 + 1.07104 w_k1_IBM_s003
                                 + 1.05977 w_k1_INTC_s003
                                 + 0.917892 w_k1_KO_s003
                                 + 1.06821 w_k1_MAT_s003
                                 + 1.1984 w_k1_MSFT_s003 + 1.08936 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.23952 w_k1_ADM_s004
                                 + 0.781367 w_k1_AXP_s004
                                 + 1.18951 w_k1_DIS_s004
                                 + 1.08301 w_k1_FDX_s004
                                 + 0.992692 w_k1_IBM_s004
                                 + 1.01902 w_k1_INTC_s004 + 1.1472 w_k1_KO_s004
                                 + 0.691787 w_k1_MAT_s004
                                 + 1.19696 w_k1_MSFT_s004
                                 + 1.21895 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.980031 w_k1_ADM_s004
                                 + 0.677147 w_k1_AXP_s004
                                 + 0.658708 w_k1_DIS_s004
                                 + 0.799688 w_k1_FDX_s004
                                 + 0.907541 w_k1_IBM_s004
                                 + 0.958415 w_k1_INTC_s004
                                 + 0.955277 w_k1_KO_s004
                                 + 0.727375 w_k1_MAT_s004
                                 + 0.807631 w_k1_MSFT_s004
                                 + 1.01404 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.933122 w_k1_ADM_s004
                                 + 0.673596 w_k1_AXP_s004
                                 + 0.683111 w_k1_DIS_s004
                                 + 0.900683 w_k1_FDX_s004
                                 + 0.61721 w_k1_IBM_s004
                                 + 0.954245 w_k1_INTC_s004
                                 + 0.80255 w_k1_KO_s004
                                 + 0.433043 w_k1_MAT_s004
                                 + 0.848064 w_k1_MSFT_s004
                                 + 0.955118 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.761928 w_k1_ADM_s004
                                 + 0.910482 w_k1_AXP_s004
                                 + 0.905309 w_k1_DIS_s004
                                 + 0.883908 w_k1_FDX_s004
                                 + 1.25686 w_k1_IBM_s004
                                 + 1.16179 w_k1_INTC_s004
                                 + 1.13544 w_k1_KO_s004 + 1.43627 w_k1_MAT_s004
                                 + 0.942645 w_k1_MSFT_s004
                                 + 0.934902 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.15189 w_k1_ADM_s004
                                 + 1.35699 w_k1_AXP_s004
                                 + 1.59805 w_k1_DIS_s004
                                 + 1.23254 w_k1_FDX_s004
                                 + 1.72666 w_k1_IBM_s004
                                 + 1.14134 w_k1_INTC_s004
                                 + 1.51601 w_k1_KO_s004 + 2.4025 w_k1_MAT_s004
                                 + 1.30637 w_k1_MSFT_s004
                                 + 1.36223 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.18023 w_k1_ADM_s004
                                 + 1.27546 w_k1_AXP_s004
                                 + 1.01657 w_k1_DIS_s004
                                 + 1.11689 w_k1_FDX_s004
                                 + 1.21796 w_k1_IBM_s004
                                 + 0.913017 w_k1_INTC_s004
                                 + 1.14669 w_k1_KO_s004 + 1.72892 w_k1_MAT_s004
                                 + 0.93316 w_k1_MSFT_s004
                                 + 1.07336 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.14545 w_k1_ADM_s004
                                 + 1.38 w_k1_AXP_s004 + 1.395 w_k1_DIS_s004
                                 + 1.26545 w_k1_FDX_s004
                                 + 1.09865 w_k1_IBM_s004
                                 + 1.25144 w_k1_INTC_s004
                                 + 1.27453 w_k1_KO_s004 + 1.4486 w_k1_MAT_s004
                                 + 1.23308 w_k1_MSFT_s004
                                 + 1.13005 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.00322 w_k1_ADM_s004
                                 + 1.0546 w_k1_AXP_s004 + 1.04932 w_k1_DIS_s004
                                 + 1.15321 w_k1_FDX_s004
                                 + 0.732767 w_k1_IBM_s004
                                 + 1.48344 w_k1_INTC_s004
                                 + 0.885777 w_k1_KO_s004
                                 + 0.762732 w_k1_MAT_s004
                                 + 1.05232 w_k1_MSFT_s004 + 1.0337 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.03359 w_k1_ADM_s005
                                 + 0.96622 w_k1_AXP_s005
                                 + 0.955451 w_k1_DIS_s005
                                 + 0.950819 w_k1_FDX_s005
                                 + 0.843183 w_k1_IBM_s005
                                 + 0.948011 w_k1_INTC_s005
                                 + 0.95205 w_k1_KO_s005
                                 + 0.964022 w_k1_MAT_s005
                                 + 0.977955 w_k1_MSFT_s005
                                 + 0.998142 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.31828 w_k1_ADM_s005
                                 + 1.18739 w_k1_AXP_s005
                                 + 1.75458 w_k1_DIS_s005
                                 + 1.36684 w_k1_FDX_s005
                                 + 1.27568 w_k1_IBM_s005
                                 + 1.31471 w_k1_INTC_s005
                                 + 1.43227 w_k1_KO_s005 + 1.0936 w_k1_MAT_s005
                                 + 1.50218 w_k1_MSFT_s005
                                 + 1.18949 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.5022 w_k1_ADM_s005
                                 + 1.26493 w_k1_AXP_s005
                                 + 1.15726 w_k1_DIS_s005
                                 + 1.15072 w_k1_FDX_s005
                                 + 0.991942 w_k1_IBM_s005
                                 + 1.28284 w_k1_INTC_s005
                                 + 1.02259 w_k1_KO_s005
                                 + 0.780457 w_k1_MAT_s005
                                 + 1.10762 w_k1_MSFT_s005
                                 + 0.966014 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.18477 w_k1_ADM_s005
                                 + 0.570215 w_k1_AXP_s005
                                 + 0.813908 w_k1_DIS_s005
                                 + 0.777732 w_k1_FDX_s005
                                 + 0.771174 w_k1_IBM_s005
                                 + 1.12445 w_k1_INTC_s005
                                 + 0.784148 w_k1_KO_s005
                                 + 0.416382 w_k1_MAT_s005
                                 + 0.980722 w_k1_MSFT_s005
                                 + 0.889804 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.897966 w_k1_ADM_s005
                                 + 0.688053 w_k1_AXP_s005
                                 + 0.612969 w_k1_DIS_s005
                                 + 0.76017 w_k1_FDX_s005
                                 + 0.628257 w_k1_IBM_s005
                                 + 0.831763 w_k1_INTC_s005
                                 + 0.755422 w_k1_KO_s005
                                 + 0.628013 w_k1_MAT_s005
                                 + 0.916334 w_k1_MSFT_s005
                                 + 0.826226 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.84618 w_k1_ADM_s005
                                 + 0.722482 w_k1_AXP_s005
                                 + 0.864441 w_k1_DIS_s005
                                 + 0.806191 w_k1_FDX_s005
                                 + 1.21631 w_k1_IBM_s005
                                 + 1.15996 w_k1_INTC_s005
                                 + 1.11761 w_k1_KO_s005
                                 + 0.775853 w_k1_MAT_s005
                                 + 1.02349 w_k1_MSFT_s005
                                 + 0.954205 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.877496 w_k1_ADM_s005
                                 + 0.733194 w_k1_AXP_s005
                                 + 0.584251 w_k1_DIS_s005
                                 + 0.681239 w_k1_FDX_s005
                                 + 0.887883 w_k1_IBM_s005
                                 + 0.73281 w_k1_INTC_s005
                                 + 0.736642 w_k1_KO_s005
                                 + 0.639075 w_k1_MAT_s005
                                 + 0.759591 w_k1_MSFT_s005
                                 + 0.747578 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.07299 w_k1_ADM_s005
                                 + 1.27842 w_k1_AXP_s005
                                 + 1.39936 w_k1_DIS_s005
                                 + 1.06165 w_k1_FDX_s005
                                 + 1.56866 w_k1_IBM_s005
                                 + 1.22267 w_k1_INTC_s005
                                 + 1.26641 w_k1_KO_s005 + 1.26291 w_k1_MAT_s005
                                 + 1.20411 w_k1_MSFT_s005 + 1.2136 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.23732 w_k1_ADM_s006
                                 + 1.84073 w_k1_AXP_s006
                                 + 1.14017 w_k1_DIS_s006
                                 + 1.27245 w_k1_FDX_s006
                                 + 1.17188 w_k1_IBM_s006
                                 + 1.19191 w_k1_INTC_s006
                                 + 1.18846 w_k1_KO_s006 + 1.44037 w_k1_MAT_s006
                                 + 1.05484 w_k1_MSFT_s006 + 1.0693 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.10628 w_k1_ADM_s006
                                 + 1.12075 w_k1_AXP_s006
                                 + 1.45476 w_k1_DIS_s006
                                 + 1.30158 w_k1_FDX_s006
                                 + 0.825997 w_k1_IBM_s006
                                 + 1.22045 w_k1_INTC_s006 + 1.0676 w_k1_KO_s006
                                 + 0.659242 w_k1_MAT_s006
                                 + 1.1332 w_k1_MSFT_s006
                                 + 0.945372 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.35225 w_k1_ADM_s006
                                 + 1.52335 w_k1_AXP_s006
                                 + 1.57175 w_k1_DIS_s006
                                 + 1.16631 w_k1_FDX_s006
                                 + 1.46663 w_k1_IBM_s006
                                 + 1.23551 w_k1_INTC_s006
                                 + 1.54757 w_k1_KO_s006 + 1.41315 w_k1_MAT_s006
                                 + 1.2438 w_k1_MSFT_s006 + 1.09694 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.813794 w_k1_ADM_s006
                                 + 0.574961 w_k1_AXP_s006
                                 + 0.529893 w_k1_DIS_s006
                                 + 0.634116 w_k1_FDX_s006
                                 + 0.889434 w_k1_IBM_s006
                                 + 0.787886 w_k1_INTC_s006
                                 + 0.746825 w_k1_KO_s006
                                 + 0.626451 w_k1_MAT_s006
                                 + 0.63827 w_k1_MSFT_s006
                                 + 0.787522 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.01828 w_k1_ADM_s006
                                 + 0.895793 w_k1_AXP_s006
                                 + 0.777535 w_k1_DIS_s006
                                 + 0.832255 w_k1_FDX_s006
                                 + 0.931094 w_k1_IBM_s006
                                 + 0.870463 w_k1_INTC_s006
                                 + 0.925473 w_k1_KO_s006
                                 + 0.939078 w_k1_MAT_s006
                                 + 0.764238 w_k1_MSFT_s006
                                 + 0.869388 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.881229 w_k1_ADM_s006
                                 + 1.19287 w_k1_AXP_s006
                                 + 0.979906 w_k1_DIS_s006
                                 + 0.915295 w_k1_FDX_s006
                                 + 1.32633 w_k1_IBM_s006
                                 + 1.08954 w_k1_INTC_s006
                                 + 1.22048 w_k1_KO_s006 + 1.72958 w_k1_MAT_s006
                                 + 0.928299 w_k1_MSFT_s006
                                 + 0.987383 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.917483 w_k1_ADM_s006
                                 + 0.859846 w_k1_AXP_s006
                                 + 0.659458 w_k1_DIS_s006
                                 + 0.812569 w_k1_FDX_s006
                                 + 0.68005 w_k1_IBM_s006
                                 + 0.732885 w_k1_INTC_s006
                                 + 0.703795 w_k1_KO_s006
                                 + 0.720387 w_k1_MAT_s006
                                 + 0.759548 w_k1_MSFT_s006
                                 + 0.792031 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.22992 w_k1_ADM_s006
                                 + 1.02253 w_k1_AXP_s006
                                 + 0.848278 w_k1_DIS_s006
                                 + 1.0386 w_k1_FDX_s006
                                 + 0.810947 w_k1_IBM_s006
                                 + 0.967487 w_k1_INTC_s006
                                 + 0.886449 w_k1_KO_s006
                                 + 0.996488 w_k1_MAT_s006
                                 + 0.999978 w_k1_MSFT_s006
                                 + 0.986327 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.892378 w_k1_ADM_s007
                                 + 0.877249 w_k1_AXP_s007
                                 + 0.847411 w_k1_DIS_s007
                                 + 1.12395 w_k1_FDX_s007
                                 + 0.567561 w_k1_IBM_s007
                                 + 1.04357 w_k1_INTC_s007
                                 + 0.729777 w_k1_KO_s007
                                 + 0.972141 w_k1_MAT_s007
                                 + 1.04257 w_k1_MSFT_s007
                                 + 0.917544 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.42485 w_k1_ADM_s007
                                 + 1.65304 w_k1_AXP_s007 + 1.5967 w_k1_DIS_s007
                                 + 1.40122 w_k1_FDX_s007
                                 + 1.97016 w_k1_IBM_s007
                                 + 1.43888 w_k1_INTC_s007
                                 + 1.66566 w_k1_KO_s007 + 2.56077 w_k1_MAT_s007
                                 + 1.38538 w_k1_MSFT_s007
                                 + 1.37555 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.86596 w_k1_ADM_s007
                                 + 0.68174 w_k1_AXP_s007
                                 + 0.689913 w_k1_DIS_s007
                                 + 0.886355 w_k1_FDX_s007
                                 + 0.687472 w_k1_IBM_s007
                                 + 0.870306 w_k1_INTC_s007
                                 + 0.725164 w_k1_KO_s007
                                 + 0.732715 w_k1_MAT_s007
                                 + 0.829051 w_k1_MSFT_s007
                                 + 0.801841 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.752514 w_k1_ADM_s007
                                 + 1.06948 w_k1_AXP_s007
                                 + 0.959346 w_k1_DIS_s007
                                 + 1.01099 w_k1_FDX_s007
                                 + 1.35551 w_k1_IBM_s007
                                 + 1.03256 w_k1_INTC_s007
                                 + 1.02639 w_k1_KO_s007 + 1.54176 w_k1_MAT_s007
                                 + 0.931645 w_k1_MSFT_s007
                                 + 1.02901 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.24895 w_k1_ADM_s007
                                 + 1.09324 w_k1_AXP_s007
                                 + 1.30004 w_k1_DIS_s007
                                 + 1.35039 w_k1_FDX_s007
                                 + 0.939046 w_k1_IBM_s007
                                 + 0.972052 w_k1_INTC_s007
                                 + 0.91134 w_k1_KO_s007 + 1.16554 w_k1_MAT_s007
                                 + 1.09573 w_k1_MSFT_s007
                                 + 1.11993 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.857214 w_k1_ADM_s007
                                 + 0.590111 w_k1_AXP_s007
                                 + 0.643336 w_k1_DIS_s007
                                 + 0.681423 w_k1_FDX_s007
                                 + 1.12713 w_k1_IBM_s007
                                 + 0.825099 w_k1_INTC_s007
                                 + 0.906542 w_k1_KO_s007 + 0.9577 w_k1_MAT_s007
                                 + 0.84788 w_k1_MSFT_s007
                                 + 0.873202 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.892981 w_k1_ADM_s007
                                 + 0.815644 w_k1_AXP_s007
                                 + 0.85767 w_k1_DIS_s007
                                 + 0.947007 w_k1_FDX_s007
                                 + 0.872196 w_k1_IBM_s007
                                 + 1.00823 w_k1_INTC_s007
                                 + 0.817973 w_k1_KO_s007
                                 + 0.752101 w_k1_MAT_s007
                                 + 0.934006 w_k1_MSFT_s007
                                 + 0.868275 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.625526 w_k1_ADM_s007
                                 + 0.539995 w_k1_AXP_s007
                                 + 0.682344 w_k1_DIS_s007
                                 + 0.759806 w_k1_FDX_s007
                                 + 0.520511 w_k1_IBM_s007
                                 + 0.800407 w_k1_INTC_s007
                                 + 0.6081 w_k1_KO_s007 + 0.366232 w_k1_MAT_s007
                                 + 0.885264 w_k1_MSFT_s007
                                 + 0.729299 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.15795 w_k1_ADM_s010
                                 + 1.32652 w_k1_AXP_s010
                                 + 1.65202 w_k1_DIS_s010
                                 + 1.28859 w_k1_FDX_s010
                                 + 1.62887 w_k1_IBM_s010
                                 + 1.11417 w_k1_INTC_s010
                                 + 1.72163 w_k1_KO_s010 + 1.90539 w_k1_MAT_s010
                                 + 1.30055 w_k1_MSFT_s010
                                 + 1.09286 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.23159 w_k1_ADM_s010
                                 + 1.4424 w_k1_AXP_s010 + 1.23581 w_k1_DIS_s010
                                 + 1.42449 w_k1_FDX_s010
                                 + 0.818797 w_k1_IBM_s010
                                 + 1.23456 w_k1_INTC_s010
                                 + 1.06139 w_k1_KO_s010 + 1.02711 w_k1_MAT_s010
                                 + 1.02465 w_k1_MSFT_s010
                                 + 1.11675 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.721883 w_k1_ADM_s010
                                 + 0.912444 w_k1_AXP_s010
                                 + 0.887079 w_k1_DIS_s010
                                 + 0.824668 w_k1_FDX_s010
                                 + 1.28499 w_k1_IBM_s010
                                 + 0.813792 w_k1_INTC_s010
                                 + 1.04003 w_k1_KO_s010 + 1.76146 w_k1_MAT_s010
                                 + 0.79235 w_k1_MSFT_s010
                                 + 1.01511 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.975681 w_k1_ADM_s010
                                 + 0.886011 w_k1_AXP_s010
                                 + 0.962892 w_k1_DIS_s010
                                 + 0.896828 w_k1_FDX_s010
                                 + 0.931971 w_k1_IBM_s010
                                 + 0.944764 w_k1_INTC_s010
                                 + 0.945519 w_k1_KO_s010 + 0.9078 w_k1_MAT_s010
                                 + 0.927517 w_k1_MSFT_s010
                                 + 0.930569 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.899461 w_k1_ADM_s010
                                 + 1.02003 w_k1_AXP_s010
                                 + 1.38309 w_k1_DIS_s010
                                 + 1.09856 w_k1_FDX_s010
                                 + 1.16031 w_k1_IBM_s010
                                 + 1.1048 w_k1_INTC_s010 + 1.2164 w_k1_KO_s010
                                 + 1.23436 w_k1_MAT_s010
                                 + 1.11887 w_k1_MSFT_s010
                                 + 1.04047 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.10409 w_k1_ADM_s010
                                 + 1.47498 w_k1_AXP_s010
                                 + 1.11874 w_k1_DIS_s010
                                 + 1.05882 w_k1_FDX_s010
                                 + 1.26534 w_k1_IBM_s010
                                 + 1.13333 w_k1_INTC_s010
                                 + 1.08446 w_k1_KO_s010 + 1.35148 w_k1_MAT_s010
                                 + 0.998084 w_k1_MSFT_s010
                                 + 1.05426 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.940016 w_k1_ADM_s010
                                 + 0.981814 w_k1_AXP_s010
                                 + 0.973411 w_k1_DIS_s010
                                 + 1.1388 w_k1_FDX_s010
                                 + 0.708223 w_k1_IBM_s010
                                 + 1.09771 w_k1_INTC_s010
                                 + 0.917456 w_k1_KO_s010
                                 + 0.875611 w_k1_MAT_s010
                                 + 1.06089 w_k1_MSFT_s010
                                 + 0.973588 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.582963 w_k1_ADM_s010
                                 + 0.604414 w_k1_AXP_s010
                                 + 0.667718 w_k1_DIS_s010
                                 + 0.730694 w_k1_FDX_s010
                                 + 0.830181 w_k1_IBM_s010
                                 + 0.814139 w_k1_INTC_s010
                                 + 0.769505 w_k1_KO_s010
                                 + 0.788348 w_k1_MAT_s010
                                 + 0.775898 w_k1_MSFT_s010
                                 + 0.812643 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.10951 w_k1_ADM_s011
                                 + 1.36427 w_k1_AXP_s011
                                 + 1.54743 w_k1_DIS_s011
                                 + 1.25689 w_k1_FDX_s011 + 1.4478 w_k1_IBM_s011
                                 + 1.48811 w_k1_INTC_s011
                                 + 1.37261 w_k1_KO_s011 + 1.16939 w_k1_MAT_s011
                                 + 1.28499 w_k1_MSFT_s011
                                 + 1.09113 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.24601 w_k1_ADM_s011
                                 + 1.09159 w_k1_AXP_s011
                                 + 0.967714 w_k1_DIS_s011
                                 + 0.961903 w_k1_FDX_s011
                                 + 1.10815 w_k1_IBM_s011
                                 + 0.976331 w_k1_INTC_s011
                                 + 0.923063 w_k1_KO_s011
                                 + 1.27812 w_k1_MAT_s011
                                 + 1.05523 w_k1_MSFT_s011
                                 + 1.01063 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.01525 w_k1_ADM_s011
                                 + 0.668553 w_k1_AXP_s011
                                 + 0.810946 w_k1_DIS_s011
                                 + 0.832099 w_k1_FDX_s011
                                 + 0.955207 w_k1_IBM_s011
                                 + 0.730048 w_k1_INTC_s011
                                 + 0.856207 w_k1_KO_s011
                                 + 0.606474 w_k1_MAT_s011
                                 + 0.820315 w_k1_MSFT_s011
                                 + 0.765555 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.19404 w_k1_ADM_s011
                                 + 1.38781 w_k1_AXP_s011
                                 + 2.00103 w_k1_DIS_s011
                                 + 1.51564 w_k1_FDX_s011
                                 + 1.24775 w_k1_IBM_s011
                                 + 1.40575 w_k1_INTC_s011
                                 + 1.52102 w_k1_KO_s011 + 1.109 w_k1_MAT_s011
                                 + 1.58524 w_k1_MSFT_s011 + 1.3032 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.17944 w_k1_ADM_s011
                                 + 1.14965 w_k1_AXP_s011
                                 + 1.24527 w_k1_DIS_s011
                                 + 1.00402 w_k1_FDX_s011
                                 + 1.36561 w_k1_IBM_s011
                                 + 0.874237 w_k1_INTC_s011
                                 + 1.14988 w_k1_KO_s011 + 1.04939 w_k1_MAT_s011
                                 + 1.02367 w_k1_MSFT_s011
                                 + 1.06882 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.970399 w_k1_ADM_s011
                                 + 1.01163 w_k1_AXP_s011
                                 + 1.18851 w_k1_DIS_s011
                                 + 1.09057 w_k1_FDX_s011
                                 + 0.848734 w_k1_IBM_s011
                                 + 0.973398 w_k1_INTC_s011
                                 + 0.841431 w_k1_KO_s011
                                 + 0.802528 w_k1_MAT_s011
                                 + 1.13341 w_k1_MSFT_s011
                                 + 1.01513 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.962593 w_k1_ADM_s011
                                 + 1.12141 w_k1_AXP_s011
                                 + 0.695432 w_k1_DIS_s011
                                 + 0.89993 w_k1_FDX_s011
                                 + 0.888548 w_k1_IBM_s011
                                 + 0.754583 w_k1_INTC_s011
                                 + 0.7456 w_k1_KO_s011 + 0.803689 w_k1_MAT_s011
                                 + 0.777884 w_k1_MSFT_s011
                                 + 0.834434 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.765734 w_k1_ADM_s011
                                 + 0.868883 w_k1_AXP_s011
                                 + 0.656712 w_k1_DIS_s011
                                 + 0.859155 w_k1_FDX_s011
                                 + 0.663231 w_k1_IBM_s011
                                 + 1.02272 w_k1_INTC_s011
                                 + 0.595772 w_k1_KO_s011
                                 + 0.525088 w_k1_MAT_s011
                                 + 0.876775 w_k1_MSFT_s011
                                 + 0.77462 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.737426 w_k1_ADM_s012
                                 + 0.65785 w_k1_AXP_s012
                                 + 0.724798 w_k1_DIS_s012
                                 + 0.804879 w_k1_FDX_s012
                                 + 0.867988 w_k1_IBM_s012
                                 + 0.760216 w_k1_INTC_s012
                                 + 0.833825 w_k1_KO_s012
                                 + 0.982799 w_k1_MAT_s012
                                 + 0.747452 w_k1_MSFT_s012
                                 + 0.883544 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.967562 w_k1_ADM_s012
                                 + 1.21123 w_k1_AXP_s012
                                 + 1.17097 w_k1_DIS_s012
                                 + 1.04234 w_k1_FDX_s012
                                 + 1.46967 w_k1_IBM_s012
                                 + 0.894663 w_k1_INTC_s012
                                 + 1.44531 w_k1_KO_s012 + 1.99009 w_k1_MAT_s012
                                 + 1.14784 w_k1_MSFT_s012
                                 + 1.27632 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.828374 w_k1_ADM_s012
                                 + 1.50478 w_k1_AXP_s012
                                 + 1.41756 w_k1_DIS_s012
                                 + 1.33877 w_k1_FDX_s012
                                 + 1.12556 w_k1_IBM_s012
                                 + 1.32362 w_k1_INTC_s012
                                 + 1.22395 w_k1_KO_s012 + 1.27906 w_k1_MAT_s012
                                 + 1.37724 w_k1_MSFT_s012
                                 + 1.28725 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.680532 w_k1_ADM_s012
                                 + 1.06786 w_k1_AXP_s012
                                 + 0.831088 w_k1_DIS_s012
                                 + 0.975089 w_k1_FDX_s012
                                 + 0.891531 w_k1_IBM_s012
                                 + 0.948599 w_k1_INTC_s012
                                 + 0.787621 w_k1_KO_s012
                                 + 0.984968 w_k1_MAT_s012
                                 + 0.885701 w_k1_MSFT_s012
                                 + 0.905835 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.29528 w_k1_ADM_s012
                                 + 1.04612 w_k1_AXP_s012
                                 + 1.22468 w_k1_DIS_s012
                                 + 1.05826 w_k1_FDX_s012
                                 + 0.919543 w_k1_IBM_s012
                                 + 0.958789 w_k1_INTC_s012
                                 + 1.00091 w_k1_KO_s012
                                 + 0.913157 w_k1_MAT_s012
                                 + 1.0173 w_k1_MSFT_s012
                                 + 0.991861 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.739507 w_k1_ADM_s012
                                 + 0.944077 w_k1_AXP_s012
                                 + 0.636268 w_k1_DIS_s012
                                 + 0.809492 w_k1_FDX_s012
                                 + 0.711408 w_k1_IBM_s012
                                 + 0.737737 w_k1_INTC_s012
                                 + 0.770754 w_k1_KO_s012
                                 + 0.996881 w_k1_MAT_s012
                                 + 0.812505 w_k1_MSFT_s012
                                 + 0.919376 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.12544 w_k1_ADM_s012
                                 + 1.1266 w_k1_AXP_s012 + 1.68385 w_k1_DIS_s012
                                 + 1.35743 w_k1_FDX_s012
                                 + 1.36653 w_k1_IBM_s012
                                 + 1.0255 w_k1_INTC_s012 + 1.45203 w_k1_KO_s012
                                 + 1.14425 w_k1_MAT_s012
                                 + 1.28226 w_k1_MSFT_s012
                                 + 1.17755 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.708305 w_k1_ADM_s012
                                 + 0.846163 w_k1_AXP_s012
                                 + 0.973255 w_k1_DIS_s012
                                 + 0.838335 w_k1_FDX_s012
                                 + 1.09354 w_k1_IBM_s012
                                 + 0.866083 w_k1_INTC_s012
                                 + 1.09685 w_k1_KO_s012 + 1.16943 w_k1_MAT_s012
                                 + 0.935753 w_k1_MSFT_s012
                                 + 1.03547 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.01703 w_k1_ADM_s013
                                 + 1.23237 w_k1_AXP_s013 + 1.3686 w_k1_DIS_s013
                                 + 1.21301 w_k1_FDX_s013
                                 + 1.01217 w_k1_IBM_s013
                                 + 1.42113 w_k1_INTC_s013
                                 + 1.21186 w_k1_KO_s013 + 1.00354 w_k1_MAT_s013
                                 + 1.14491 w_k1_MSFT_s013 + 1.1267 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.706275 w_k1_ADM_s013
                                 + 0.782825 w_k1_AXP_s013
                                 + 0.83952 w_k1_DIS_s013
                                 + 0.86236 w_k1_FDX_s013
                                 + 0.928163 w_k1_IBM_s013
                                 + 1.03044 w_k1_INTC_s013
                                 + 0.92365 w_k1_KO_s013
                                 + 0.786521 w_k1_MAT_s013
                                 + 0.832737 w_k1_MSFT_s013
                                 + 0.895062 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.800772 w_k1_ADM_s013
                                 + 0.855166 w_k1_AXP_s013
                                 + 0.940054 w_k1_DIS_s013
                                 + 1.00581 w_k1_FDX_s013
                                 + 0.751834 w_k1_IBM_s013
                                 + 0.900885 w_k1_INTC_s013
                                 + 0.974933 w_k1_KO_s013
                                 + 0.852572 w_k1_MAT_s013
                                 + 0.97251 w_k1_MSFT_s013
                                 + 0.957177 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.80298 w_k1_ADM_s013
                                 + 1.11611 w_k1_AXP_s013
                                 + 1.21185 w_k1_DIS_s013
                                 + 1.01152 w_k1_FDX_s013
                                 + 1.25011 w_k1_IBM_s013
                                 + 1.07269 w_k1_INTC_s013
                                 + 1.34063 w_k1_KO_s013 + 1.49189 w_k1_MAT_s013
                                 + 1.0325 w_k1_MSFT_s013 + 1.00155 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.799671 w_k1_ADM_s013
                                 + 0.619097 w_k1_AXP_s013
                                 + 0.646494 w_k1_DIS_s013
                                 + 0.756354 w_k1_FDX_s013
                                 + 0.647476 w_k1_IBM_s013
                                 + 0.751133 w_k1_INTC_s013
                                 + 0.709398 w_k1_KO_s013
                                 + 0.75615 w_k1_MAT_s013
                                 + 0.73338 w_k1_MSFT_s013
                                 + 0.934312 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.04084 w_k1_ADM_s013
                                 + 1.39243 w_k1_AXP_s013
                                 + 0.886544 w_k1_DIS_s013
                                 + 1.03749 w_k1_FDX_s013
                                 + 1.05745 w_k1_IBM_s013
                                 + 1.0651 w_k1_INTC_s013 + 1.05151 w_k1_KO_s013
                                 + 1.28574 w_k1_MAT_s013
                                 + 0.867085 w_k1_MSFT_s013
                                 + 0.993522 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.927881 w_k1_ADM_s013
                                 + 1.51838 w_k1_AXP_s013 + 1.5182 w_k1_DIS_s013
                                 + 1.189 w_k1_FDX_s013 + 1.45928 w_k1_IBM_s013
                                 + 1.02665 w_k1_INTC_s013
                                 + 1.42305 w_k1_KO_s013 + 1.92889 w_k1_MAT_s013
                                 + 0.967299 w_k1_MSFT_s013
                                 + 1.08709 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.876889 w_k1_ADM_s013
                                 + 0.976388 w_k1_AXP_s013
                                 + 0.780537 w_k1_DIS_s013
                                 + 0.852295 w_k1_FDX_s013
                                 + 1.16286 w_k1_IBM_s013
                                 + 1.00922 w_k1_INTC_s013
                                 + 0.980335 w_k1_KO_s013
                                 + 1.08426 w_k1_MAT_s013
                                 + 0.828444 w_k1_MSFT_s013
                                 + 0.936903 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.968225 w_k1_ADM_s014
                                 + 0.853682 w_k1_AXP_s014
                                 + 1.00227 w_k1_DIS_s014
                                 + 0.844522 w_k1_FDX_s014
                                 + 1.24202 w_k1_IBM_s014
                                 + 0.799023 w_k1_INTC_s014
                                 + 1.09995 w_k1_KO_s014 + 1.04706 w_k1_MAT_s014
                                 + 0.758226 w_k1_MSFT_s014
                                 + 1.07324 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.1023 w_k1_ADM_s014
                                 + 1.28819 w_k1_AXP_s014
                                 + 1.40844 w_k1_DIS_s014
                                 + 1.20081 w_k1_FDX_s014
                                 + 1.65934 w_k1_IBM_s014
                                 + 0.88304 w_k1_INTC_s014
                                 + 1.28638 w_k1_KO_s014 + 1.52702 w_k1_MAT_s014
                                 + 0.868769 w_k1_MSFT_s014 + 1.152 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.17913 w_k1_ADM_s014
                                 + 1.07345 w_k1_AXP_s014
                                 + 1.63963 w_k1_DIS_s014
                                 + 1.30461 w_k1_FDX_s014
                                 + 1.18973 w_k1_IBM_s014
                                 + 0.95274 w_k1_INTC_s014 + 1.3129 w_k1_KO_s014
                                 + 1.08226 w_k1_MAT_s014
                                 + 1.16894 w_k1_MSFT_s014 + 1.1648 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.958097 w_k1_ADM_s014
                                 + 1.06613 w_k1_AXP_s014
                                 + 1.28723 w_k1_DIS_s014
                                 + 1.17824 w_k1_FDX_s014
                                 + 0.870008 w_k1_IBM_s014
                                 + 1.1541 w_k1_INTC_s014
                                 + 0.954845 w_k1_KO_s014
                                 + 0.903631 w_k1_MAT_s014
                                 + 1.0187 w_k1_MSFT_s014
                                 + 0.937368 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.07374 w_k1_ADM_s014
                                 + 0.92443 w_k1_AXP_s014
                                 + 0.872007 w_k1_DIS_s014
                                 + 1.0272 w_k1_FDX_s014
                                 + 0.649893 w_k1_IBM_s014
                                 + 0.733575 w_k1_INTC_s014
                                 + 0.71389 w_k1_KO_s014
                                 + 0.617521 w_k1_MAT_s014
                                 + 0.779184 w_k1_MSFT_s014
                                 + 0.998721 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.26952 w_k1_ADM_s014
                                 + 1.11968 w_k1_AXP_s014
                                 + 0.984748 w_k1_DIS_s014
                                 + 0.983475 w_k1_FDX_s014
                                 + 1.08705 w_k1_IBM_s014
                                 + 1.14357 w_k1_INTC_s014
                                 + 1.06157 w_k1_KO_s014
                                 + 0.892172 w_k1_MAT_s014
                                 + 0.922379 w_k1_MSFT_s014
                                 + 1.03135 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.735218 w_k1_ADM_s014
                                 + 0.939782 w_k1_AXP_s014
                                 + 0.872472 w_k1_DIS_s014
                                 + 0.915309 w_k1_FDX_s014
                                 + 0.921937 w_k1_IBM_s014
                                 + 0.90904 w_k1_INTC_s014
                                 + 0.935018 w_k1_KO_s014
                                 + 0.82739 w_k1_MAT_s014
                                 + 0.748592 w_k1_MSFT_s014
                                 + 0.858438 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.962508 w_k1_ADM_s014
                                 + 0.615395 w_k1_AXP_s014
                                 + 0.606838 w_k1_DIS_s014
                                 + 0.725416 w_k1_FDX_s014
                                 + 0.780197 w_k1_IBM_s014
                                 + 0.757221 w_k1_INTC_s014
                                 + 0.729247 w_k1_KO_s014
                                 + 0.44695 w_k1_MAT_s014
                                 + 0.682565 w_k1_MSFT_s014
                                 + 0.775908 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.718674 w_k1_ADM_s015
                                 + 0.691099 w_k1_AXP_s015
                                 + 0.738134 w_k1_DIS_s015
                                 + 0.7284 w_k1_FDX_s015 + 1.07558 w_k1_IBM_s015
                                 + 1.00534 w_k1_INTC_s015
                                 + 0.987681 w_k1_KO_s015
                                 + 0.90449 w_k1_MAT_s015
                                 + 0.941902 w_k1_MSFT_s015
                                 + 0.938636 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.678247 w_k1_ADM_s015
                                 + 0.808074 w_k1_AXP_s015
                                 + 0.668786 w_k1_DIS_s015
                                 + 0.78898 w_k1_FDX_s015 + 0.7286 w_k1_IBM_s015
                                 + 0.756549 w_k1_INTC_s015
                                 + 0.726943 w_k1_KO_s015
                                 + 0.676549 w_k1_MAT_s015
                                 + 0.946777 w_k1_MSFT_s015
                                 + 0.883222 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.833866 w_k1_ADM_s015
                                 + 1.00654 w_k1_AXP_s015
                                 + 1.44872 w_k1_DIS_s015
                                 + 1.13236 w_k1_FDX_s015
                                 + 0.939546 w_k1_IBM_s015
                                 + 1.27394 w_k1_INTC_s015
                                 + 1.15578 w_k1_KO_s015
                                 + 0.799386 w_k1_MAT_s015
                                 + 1.32709 w_k1_MSFT_s015
                                 + 1.04487 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.2045 w_k1_ADM_s015
                                 + 1.81367 w_k1_AXP_s015
                                 + 1.79985 w_k1_DIS_s015
                                 + 1.69486 w_k1_FDX_s015
                                 + 1.19007 w_k1_IBM_s015
                                 + 1.13431 w_k1_INTC_s015
                                 + 1.25342 w_k1_KO_s015 + 1.07109 w_k1_MAT_s015
                                 + 1.21829 w_k1_MSFT_s015
                                 + 1.07619 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.71192 w_k1_ADM_s015
                                 + 1.02589 w_k1_AXP_s015
                                 + 0.81897 w_k1_DIS_s015
                                 + 0.966315 w_k1_FDX_s015
                                 + 0.862048 w_k1_IBM_s015
                                 + 1.17031 w_k1_INTC_s015
                                 + 0.878845 w_k1_KO_s015
                                 + 0.837648 w_k1_MAT_s015
                                 + 1.06554 w_k1_MSFT_s015
                                 + 0.960609 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.927154 w_k1_ADM_s015
                                 + 1.91274 w_k1_AXP_s015
                                 + 1.77879 w_k1_DIS_s015
                                 + 1.35058 w_k1_FDX_s015
                                 + 1.87728 w_k1_IBM_s015
                                 + 1.61239 w_k1_INTC_s015
                                 + 1.39884 w_k1_KO_s015 + 1.32494 w_k1_MAT_s015
                                 + 1.36335 w_k1_MSFT_s015
                                 + 1.33317 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.09401 w_k1_ADM_s015
                                 + 0.877233 w_k1_AXP_s015
                                 + 0.941381 w_k1_DIS_s015
                                 + 0.787805 w_k1_FDX_s015
                                 + 1.51325 w_k1_IBM_s015
                                 + 0.912246 w_k1_INTC_s015
                                 + 1.12496 w_k1_KO_s015 + 1.34715 w_k1_MAT_s015
                                 + 0.935117 w_k1_MSFT_s015
                                 + 1.03204 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.18718 w_k1_ADM_s015
                                 + 1.21961 w_k1_AXP_s015
                                 + 0.944362 w_k1_DIS_s015
                                 + 1.07409 w_k1_FDX_s015
                                 + 1.12957 w_k1_IBM_s015
                                 + 1.08626 w_k1_INTC_s015
                                 + 1.10163 w_k1_KO_s015 + 1.17464 w_k1_MAT_s015
                                 + 1.17807 w_k1_MSFT_s015
                                 + 1.09973 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.684419 w_k1_ADM_s016
                                 + 1.08937 w_k1_AXP_s016
                                 + 0.59042 w_k1_DIS_s016
                                 + 0.767297 w_k1_FDX_s016
                                 + 0.923542 w_k1_IBM_s016
                                 + 0.872604 w_k1_INTC_s016
                                 + 0.928308 w_k1_KO_s016
                                 + 1.29611 w_k1_MAT_s016
                                 + 0.79352 w_k1_MSFT_s016
                                 + 0.974802 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.929046 w_k1_ADM_s016
                                 + 0.991743 w_k1_AXP_s016
                                 + 1.04866 w_k1_DIS_s016
                                 + 1.02718 w_k1_FDX_s016
                                 + 0.97134 w_k1_IBM_s016
                                 + 0.977704 w_k1_INTC_s016
                                 + 1.0335 w_k1_KO_s016 + 0.992291 w_k1_MAT_s016
                                 + 1.0184 w_k1_MSFT_s016 + 1.16203 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.0115 w_k1_ADM_s016
                                 + 1.38515 w_k1_AXP_s016
                                 + 1.15468 w_k1_DIS_s016 + 1.0635 w_k1_FDX_s016
                                 + 1.35432 w_k1_IBM_s016
                                 + 1.16866 w_k1_INTC_s016
                                 + 1.18668 w_k1_KO_s016 + 1.85293 w_k1_MAT_s016
                                 + 1.19399 w_k1_MSFT_s016
                                 + 1.13484 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.847593 w_k1_ADM_s016
                                 + 1.02771 w_k1_AXP_s016
                                 + 0.785358 w_k1_DIS_s016
                                 + 0.894287 w_k1_FDX_s016
                                 + 0.655365 w_k1_IBM_s016
                                 + 0.808001 w_k1_INTC_s016
                                 + 0.837597 w_k1_KO_s016
                                 + 0.680812 w_k1_MAT_s016
                                 + 0.912927 w_k1_MSFT_s016
                                 + 0.898578 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.728111 w_k1_ADM_s016
                                 + 0.708419 w_k1_AXP_s016
                                 + 0.713335 w_k1_DIS_s016
                                 + 0.704371 w_k1_FDX_s016
                                 + 1.02907 w_k1_IBM_s016
                                 + 0.743572 w_k1_INTC_s016
                                 + 0.851544 w_k1_KO_s016
                                 + 0.875337 w_k1_MAT_s016
                                 + 0.761305 w_k1_MSFT_s016
                                 + 0.819337 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.23689 w_k1_ADM_s016
                                 + 0.984254 w_k1_AXP_s016
                                 + 1.00947 w_k1_DIS_s016
                                 + 1.04153 w_k1_FDX_s016
                                 + 0.794904 w_k1_IBM_s016
                                 + 0.843589 w_k1_INTC_s016
                                 + 0.992641 w_k1_KO_s016
                                 + 0.81576 w_k1_MAT_s016
                                 + 1.08416 w_k1_MSFT_s016
                                 + 0.959371 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.25186 w_k1_ADM_s016
                                 + 1.67991 w_k1_AXP_s016
                                 + 1.39801 w_k1_DIS_s016
                                 + 1.43049 w_k1_FDX_s016
                                 + 0.916058 w_k1_IBM_s016
                                 + 1.15446 w_k1_INTC_s016
                                 + 1.21856 w_k1_KO_s016 + 1.43291 w_k1_MAT_s016
                                 + 1.45298 w_k1_MSFT_s016
                                 + 1.12339 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.666232 w_k1_ADM_s016
                                 + 0.640734 w_k1_AXP_s016
                                 + 0.556005 w_k1_DIS_s016
                                 + 0.61405 w_k1_FDX_s016
                                 + 0.694059 w_k1_IBM_s016
                                 + 0.665364 w_k1_INTC_s016
                                 + 0.784622 w_k1_KO_s016
                                 + 0.663356 w_k1_MAT_s016
                                 + 0.70735 w_k1_MSFT_s016
                                 + 0.744558 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.979202 w_k1_ADM_s017
                                 + 0.861952 w_k1_AXP_s017
                                 + 0.859514 w_k1_DIS_s017
                                 + 0.957025 w_k1_FDX_s017
                                 + 0.802679 w_k1_IBM_s017
                                 + 0.871028 w_k1_INTC_s017
                                 + 0.762094 w_k1_KO_s017
                                 + 0.635802 w_k1_MAT_s017
                                 + 0.862255 w_k1_MSFT_s017
                                 + 0.876416 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.14639 w_k1_ADM_s017
                                 + 1.14427 w_k1_AXP_s017
                                 + 1.13335 w_k1_DIS_s017
                                 + 1.23826 w_k1_FDX_s017
                                 + 0.68659 w_k1_IBM_s017
                                 + 1.20606 w_k1_INTC_s017
                                 + 0.811698 w_k1_KO_s017
                                 + 0.568522 w_k1_MAT_s017
                                 + 0.938866 w_k1_MSFT_s017
                                 + 0.908578 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.38509 w_k1_ADM_s017
                                 + 1.60058 w_k1_AXP_s017
                                 + 1.82766 w_k1_DIS_s017
                                 + 1.35839 w_k1_FDX_s017
                                 + 1.38616 w_k1_IBM_s017
                                 + 1.15174 w_k1_INTC_s017
                                 + 1.40309 w_k1_KO_s017
                                 + 0.936457 w_k1_MAT_s017
                                 + 1.15989 w_k1_MSFT_s017
                                 + 1.14691 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.08857 w_k1_ADM_s017
                                 + 0.875536 w_k1_AXP_s017
                                 + 0.794962 w_k1_DIS_s017
                                 + 0.848608 w_k1_FDX_s017
                                 + 1.11651 w_k1_IBM_s017
                                 + 0.903458 w_k1_INTC_s017
                                 + 0.965389 w_k1_KO_s017 + 0.8279 w_k1_MAT_s017
                                 + 0.824612 w_k1_MSFT_s017
                                 + 0.854613 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.16474 w_k1_ADM_s017
                                 + 0.881287 w_k1_AXP_s017
                                 + 1.32857 w_k1_DIS_s017
                                 + 1.04089 w_k1_FDX_s017
                                 + 1.52373 w_k1_IBM_s017
                                 + 1.12799 w_k1_INTC_s017
                                 + 1.29511 w_k1_KO_s017 + 1.31833 w_k1_MAT_s017
                                 + 1.13397 w_k1_MSFT_s017
                                 + 0.988098 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.19119 w_k1_ADM_s017
                                 + 0.997067 w_k1_AXP_s017
                                 + 1.06798 w_k1_DIS_s017
                                 + 1.14299 w_k1_FDX_s017
                                 + 0.976593 w_k1_IBM_s017
                                 + 1.25923 w_k1_INTC_s017
                                 + 1.01406 w_k1_KO_s017
                                 + 0.699349 w_k1_MAT_s017
                                 + 1.07922 w_k1_MSFT_s017
                                 + 0.901081 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.910135 w_k1_ADM_s017
                                 + 1.54157 w_k1_AXP_s017 + 1.0899 w_k1_DIS_s017
                                 + 1.18331 w_k1_FDX_s017
                                 + 1.50504 w_k1_IBM_s017
                                 + 1.33412 w_k1_INTC_s017
                                 + 1.21908 w_k1_KO_s017 + 1.39668 w_k1_MAT_s017
                                 + 0.975629 w_k1_MSFT_s017
                                 + 1.06602 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.31087 w_k1_ADM_s017
                                 + 1.48491 w_k1_AXP_s017
                                 + 1.16258 w_k1_DIS_s017
                                 + 1.23307 w_k1_FDX_s017
                                 + 1.17371 w_k1_IBM_s017
                                 + 1.19419 w_k1_INTC_s017
                                 + 1.24932 w_k1_KO_s017 + 1.08033 w_k1_MAT_s017
                                 + 1.07661 w_k1_MSFT_s017
                                 + 1.10948 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.753803 w_k1_ADM_s100
                                 + 0.803789 w_k1_AXP_s100
                                 + 0.666696 w_k1_DIS_s100
                                 + 0.739661 w_k1_FDX_s100
                                 + 0.877849 w_k1_IBM_s100
                                 + 0.831369 w_k1_INTC_s100
                                 + 0.794426 w_k1_KO_s100
                                 + 0.876047 w_k1_MAT_s100
                                 + 0.872811 w_k1_MSFT_s100
                                 + 0.836284 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.27844 w_k1_ADM_s100
                                 + 1.22184 w_k1_AXP_s100
                                 + 0.99389 w_k1_DIS_s100
                                 + 1.18717 w_k1_FDX_s100
                                 + 1.00994 w_k1_IBM_s100
                                 + 1.32526 w_k1_INTC_s100
                                 + 0.961267 w_k1_KO_s100
                                 + 1.01313 w_k1_MAT_s100
                                 + 1.14786 w_k1_MSFT_s100
                                 + 1.10902 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.987502 w_k1_ADM_s100
                                 + 1.0908 w_k1_AXP_s100 + 1.02417 w_k1_DIS_s100
                                 + 0.93021 w_k1_FDX_s100
                                 + 0.968634 w_k1_IBM_s100
                                 + 1.13276 w_k1_INTC_s100
                                 + 1.00809 w_k1_KO_s100 + 1.18738 w_k1_MAT_s100
                                 + 1.08138 w_k1_MSFT_s100
                                 + 1.00509 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.36267 w_k1_ADM_s100
                                 + 1.01552 w_k1_AXP_s100
                                 + 0.888579 w_k1_DIS_s100
                                 + 1.01542 w_k1_FDX_s100
                                 + 0.725709 w_k1_IBM_s100
                                 + 0.801251 w_k1_INTC_s100
                                 + 0.796535 w_k1_KO_s100
                                 + 0.779994 w_k1_MAT_s100
                                 + 0.821905 w_k1_MSFT_s100
                                 + 0.939829 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.09277 w_k1_ADM_s100
                                 + 1.53214 w_k1_AXP_s100
                                 + 1.61763 w_k1_DIS_s100 + 1.3873 w_k1_FDX_s100
                                 + 1.2465 w_k1_IBM_s100
                                 + 1.08916 w_k1_INTC_s100
                                 + 1.21723 w_k1_KO_s100 + 1.69182 w_k1_MAT_s100
                                 + 1.33557 w_k1_MSFT_s100
                                 + 1.15063 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.02779 w_k1_ADM_s100
                                 + 1.08377 w_k1_AXP_s100
                                 + 0.923612 w_k1_DIS_s100
                                 + 1.03158 w_k1_FDX_s100
                                 + 1.07657 w_k1_IBM_s100
                                 + 1.08102 w_k1_INTC_s100
                                 + 1.07677 w_k1_KO_s100 + 1.25979 w_k1_MAT_s100
                                 + 1.09224 w_k1_MSFT_s100 + 1.0159 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.898769 w_k1_ADM_s100
                                 + 1.06687 w_k1_AXP_s100
                                 + 1.45389 w_k1_DIS_s100
                                 + 1.07801 w_k1_FDX_s100
                                 + 1.19244 w_k1_IBM_s100
                                 + 1.16533 w_k1_INTC_s100
                                 + 1.45597 w_k1_KO_s100 + 1.38358 w_k1_MAT_s100
                                 + 1.22836 w_k1_MSFT_s100
                                 + 1.00816 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.893347 w_k1_ADM_s100
                                 + 0.910429 w_k1_AXP_s100
                                 + 0.891736 w_k1_DIS_s100
                                 + 0.869547 w_k1_FDX_s100
                                 + 1.12374 w_k1_IBM_s100
                                 + 0.945238 w_k1_INTC_s100
                                 + 0.909455 w_k1_KO_s100
                                 + 0.648402 w_k1_MAT_s100
                                 + 0.93441 w_k1_MSFT_s100
                                 + 0.911202 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.27187 w_k1_ADM_s101
                                 + 1.45914 w_k1_AXP_s101 + 1.264 w_k1_DIS_s101
                                 + 1.28249 w_k1_FDX_s101
                                 + 1.38044 w_k1_IBM_s101
                                 + 1.24092 w_k1_INTC_s101
                                 + 1.22206 w_k1_KO_s101 + 1.23822 w_k1_MAT_s101
                                 + 1.20906 w_k1_MSFT_s101
                                 + 1.20328 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.958847 w_k1_ADM_s101
                                 + 0.814476 w_k1_AXP_s101
                                 + 1.03288 w_k1_DIS_s101
                                 + 1.10037 w_k1_FDX_s101
                                 + 0.914707 w_k1_IBM_s101
                                 + 1.19324 w_k1_INTC_s101
                                 + 0.982871 w_k1_KO_s101
                                 + 0.831848 w_k1_MAT_s101
                                 + 1.07075 w_k1_MSFT_s101
                                 + 0.914311 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.746711 w_k1_ADM_s101
                                 + 0.740249 w_k1_AXP_s101
                                 + 0.651614 w_k1_DIS_s101
                                 + 0.800904 w_k1_FDX_s101
                                 + 0.685766 w_k1_IBM_s101
                                 + 0.839641 w_k1_INTC_s101
                                 + 0.726626 w_k1_KO_s101
                                 + 0.520178 w_k1_MAT_s101
                                 + 0.759813 w_k1_MSFT_s101
                                 + 0.787141 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.15947 w_k1_ADM_s101
                                 + 1.24961 w_k1_AXP_s101
                                 + 1.29952 w_k1_DIS_s101
                                 + 1.33975 w_k1_FDX_s101
                                 + 0.866572 w_k1_IBM_s101
                                 + 1.09038 w_k1_INTC_s101
                                 + 0.991986 w_k1_KO_s101
                                 + 0.869814 w_k1_MAT_s101
                                 + 1.18184 w_k1_MSFT_s101
                                 + 1.01487 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.07355 w_k1_ADM_s101
                                 + 1.24482 w_k1_AXP_s101
                                 + 0.854182 w_k1_DIS_s101
                                 + 1.01414 w_k1_FDX_s101
                                 + 0.934783 w_k1_IBM_s101
                                 + 1.07872 w_k1_INTC_s101
                                 + 0.935605 w_k1_KO_s101
                                 + 1.14111 w_k1_MAT_s101
                                 + 1.00811 w_k1_MSFT_s101
                                 + 1.01457 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.826766 w_k1_ADM_s101
                                 + 0.678817 w_k1_AXP_s101
                                 + 0.776637 w_k1_DIS_s101
                                 + 0.791786 w_k1_FDX_s101
                                 + 1.20835 w_k1_IBM_s101
                                 + 0.895475 w_k1_INTC_s101
                                 + 1.04845 w_k1_KO_s101
                                 + 0.866073 w_k1_MAT_s101
                                 + 0.841316 w_k1_MSFT_s101
                                 + 0.917745 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.11939 w_k1_ADM_s101
                                 + 1.12489 w_k1_AXP_s101
                                 + 1.29547 w_k1_DIS_s101
                                 + 1.08771 w_k1_FDX_s101
                                 + 1.79231 w_k1_IBM_s101
                                 + 1.00876 w_k1_INTC_s101
                                 + 1.48499 w_k1_KO_s101 + 1.73556 w_k1_MAT_s101
                                 + 1.05413 w_k1_MSFT_s101
                                 + 1.24917 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.798355 w_k1_ADM_s101
                                 + 0.886495 w_k1_AXP_s101
                                 + 0.868305 w_k1_DIS_s101
                                 + 0.855986 w_k1_FDX_s101
                                 + 1.0948 w_k1_IBM_s101
                                 + 0.841185 w_k1_INTC_s101
                                 + 0.905401 w_k1_KO_s101
                                 + 1.10793 w_k1_MAT_s101
                                 + 0.828941 w_k1_MSFT_s101
                                 + 0.98258 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.05885 w_k1_ADM_s102
                                 + 1.24378 w_k1_AXP_s102
                                 + 1.02748 w_k1_DIS_s102
                                 + 0.951938 w_k1_FDX_s102
                                 + 1.5048 w_k1_IBM_s102
                                 + 1.34203 w_k1_INTC_s102
                                 + 1.16126 w_k1_KO_s102 + 1.16172 w_k1_MAT_s102
                                 + 1.01821 w_k1_MSFT_s102
                                 + 1.14632 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.34946 w_k1_ADM_s102
                                 + 1.43745 w_k1_AXP_s102
                                 + 1.61144 w_k1_DIS_s102
                                 + 1.40104 w_k1_FDX_s102
                                 + 1.20891 w_k1_IBM_s102
                                 + 1.34702 w_k1_INTC_s102 + 1.2702 w_k1_KO_s102
                                 + 1.18134 w_k1_MAT_s102
                                 + 1.30959 w_k1_MSFT_s102 + 1.2207 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.7101 w_k1_ADM_s102
                                 + 0.681518 w_k1_AXP_s102
                                 + 0.642241 w_k1_DIS_s102
                                 + 0.768637 w_k1_FDX_s102
                                 + 0.829051 w_k1_IBM_s102
                                 + 0.91137 w_k1_INTC_s102
                                 + 0.70799 w_k1_KO_s102
                                 + 0.700523 w_k1_MAT_s102
                                 + 0.743583 w_k1_MSFT_s102
                                 + 0.857222 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.00135 w_k1_ADM_s102
                                 + 0.836952 w_k1_AXP_s102
                                 + 0.813543 w_k1_DIS_s102
                                 + 0.839133 w_k1_FDX_s102
                                 + 0.819519 w_k1_IBM_s102
                                 + 0.947567 w_k1_INTC_s102
                                 + 0.998208 w_k1_KO_s102
                                 + 0.804815 w_k1_MAT_s102
                                 + 0.992086 w_k1_MSFT_s102
                                 + 1.08733 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.857347 w_k1_ADM_s102
                                 + 0.884611 w_k1_AXP_s102
                                 + 0.836555 w_k1_DIS_s102
                                 + 1.03104 w_k1_FDX_s102
                                 + 0.617576 w_k1_IBM_s102
                                 + 1.15542 w_k1_INTC_s102
                                 + 0.776414 w_k1_KO_s102
                                 + 0.506907 w_k1_MAT_s102
                                 + 1.04672 w_k1_MSFT_s102
                                 + 0.885393 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.13632 w_k1_ADM_s102
                                 + 0.933321 w_k1_AXP_s102
                                 + 1.07257 w_k1_DIS_s102
                                 + 1.02204 w_k1_FDX_s102
                                 + 0.974175 w_k1_IBM_s102
                                 + 0.914487 w_k1_INTC_s102
                                 + 1.0284 w_k1_KO_s102 + 0.669464 w_k1_MAT_s102
                                 + 1.04049 w_k1_MSFT_s102
                                 + 1.06715 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.02844 w_k1_ADM_s102
                                 + 1.23538 w_k1_AXP_s102
                                 + 0.92414 w_k1_DIS_s102 + 0.98 w_k1_FDX_s102
                                 + 1.0962 w_k1_IBM_s102 + 1.0551 w_k1_INTC_s102
                                 + 1.19925 w_k1_KO_s102 + 1.06712 w_k1_MAT_s102
                                 + 1.10146 w_k1_MSFT_s102
                                 + 1.09559 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.25702 w_k1_ADM_s102
                                 + 1.07864 w_k1_AXP_s102
                                 + 1.07234 w_k1_DIS_s102
                                 + 1.19364 w_k1_FDX_s102
                                 + 0.868813 w_k1_IBM_s102
                                 + 1.18827 w_k1_INTC_s102
                                 + 1.06079 w_k1_KO_s102
                                 + 0.880698 w_k1_MAT_s102
                                 + 1.15916 w_k1_MSFT_s102
                                 + 1.05125 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.18098 w_k1_ADM_s103
                                 + 1.02245 w_k1_AXP_s103
                                 + 1.18397 w_k1_DIS_s103
                                 + 1.15186 w_k1_FDX_s103
                                 + 0.749555 w_k1_IBM_s103
                                 + 1.01032 w_k1_INTC_s103
                                 + 0.877498 w_k1_KO_s103
                                 + 0.526773 w_k1_MAT_s103
                                 + 1.29446 w_k1_MSFT_s103
                                 + 1.01212 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.956944 w_k1_ADM_s103
                                 + 1.08579 w_k1_AXP_s103 + 1.0607 w_k1_DIS_s103
                                 + 0.939779 w_k1_FDX_s103
                                 + 1.04226 w_k1_IBM_s103
                                 + 0.810513 w_k1_INTC_s103
                                 + 0.919443 w_k1_KO_s103
                                 + 0.911266 w_k1_MAT_s103
                                 + 1.0999 w_k1_MSFT_s103 + 1.05467 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.02094 w_k1_ADM_s103
                                 + 1.37387 w_k1_AXP_s103
                                 + 1.42049 w_k1_DIS_s103
                                 + 1.09341 w_k1_FDX_s103
                                 + 1.49881 w_k1_IBM_s103
                                 + 0.918676 w_k1_INTC_s103
                                 + 1.38972 w_k1_KO_s103 + 1.91452 w_k1_MAT_s103
                                 + 1.13312 w_k1_MSFT_s103
                                 + 1.19428 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.18398 w_k1_ADM_s103
                                 + 1.56594 w_k1_AXP_s103
                                 + 1.41788 w_k1_DIS_s103
                                 + 1.31724 w_k1_FDX_s103
                                 + 0.908456 w_k1_IBM_s103
                                 + 1.25197 w_k1_INTC_s103
                                 + 0.966225 w_k1_KO_s103
                                 + 0.821261 w_k1_MAT_s103
                                 + 1.18147 w_k1_MSFT_s103
                                 + 1.05596 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.910026 w_k1_ADM_s103
                                 + 0.588547 w_k1_AXP_s103
                                 + 0.582942 w_k1_DIS_s103
                                 + 0.766604 w_k1_FDX_s103
                                 + 0.662728 w_k1_IBM_s103
                                 + 0.680435 w_k1_INTC_s103
                                 + 0.657223 w_k1_KO_s103
                                 + 0.414213 w_k1_MAT_s103
                                 + 0.796645 w_k1_MSFT_s103
                                 + 0.782833 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.883369 w_k1_ADM_s103
                                 + 1.11821 w_k1_AXP_s103
                                 + 0.936632 w_k1_DIS_s103
                                 + 0.885188 w_k1_FDX_s103
                                 + 1.4601 w_k1_IBM_s103
                                 + 0.815181 w_k1_INTC_s103
                                 + 1.06204 w_k1_KO_s103 + 1.42324 w_k1_MAT_s103
                                 + 0.929822 w_k1_MSFT_s103
                                 + 1.06251 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.01913 w_k1_ADM_s103
                                 + 1.08344 w_k1_AXP_s103
                                 + 0.810649 w_k1_DIS_s103
                                 + 0.946948 w_k1_FDX_s103
                                 + 0.796105 w_k1_IBM_s103
                                 + 0.875911 w_k1_INTC_s103
                                 + 0.792118 w_k1_KO_s103
                                 + 0.71466 w_k1_MAT_s103
                                 + 0.969643 w_k1_MSFT_s103
                                 + 0.91058 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.776498 w_k1_ADM_s103
                                 + 0.771567 w_k1_AXP_s103
                                 + 0.704524 w_k1_DIS_s103
                                 + 0.713422 w_k1_FDX_s103
                                 + 0.891081 w_k1_IBM_s103
                                 + 0.739754 w_k1_INTC_s103
                                 + 0.738375 w_k1_KO_s103
                                 + 0.813418 w_k1_MAT_s103
                                 + 0.818086 w_k1_MSFT_s103
                                 + 0.827539 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.21881 w_k1_ADM_s104
                                 + 1.63509 w_k1_AXP_s104
                                 + 1.64536 w_k1_DIS_s104
                                 + 1.34819 w_k1_FDX_s104
                                 + 1.69231 w_k1_IBM_s104
                                 + 0.987206 w_k1_INTC_s104
                                 + 1.44377 w_k1_KO_s104 + 1.41248 w_k1_MAT_s104
                                 + 1.17496 w_k1_MSFT_s104
                                 + 1.27929 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.791278 w_k1_ADM_s104
                                 + 0.691628 w_k1_AXP_s104
                                 + 0.581164 w_k1_DIS_s104
                                 + 0.77359 w_k1_FDX_s104
                                 + 0.678418 w_k1_IBM_s104
                                 + 0.724316 w_k1_INTC_s104
                                 + 0.714351 w_k1_KO_s104
                                 + 0.389749 w_k1_MAT_s104
                                 + 0.717188 w_k1_MSFT_s104
                                 + 0.769817 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.02559 w_k1_ADM_s104
                                 + 1.06572 w_k1_AXP_s104
                                 + 0.977306 w_k1_DIS_s104
                                 + 0.907699 w_k1_FDX_s104
                                 + 1.34656 w_k1_IBM_s104
                                 + 0.751355 w_k1_INTC_s104
                                 + 1.03538 w_k1_KO_s104 + 1.15579 w_k1_MAT_s104
                                 + 0.77801 w_k1_MSFT_s104
                                 + 0.915582 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.971928 w_k1_ADM_s104
                                 + 1.06104 w_k1_AXP_s104 + 1.4582 w_k1_DIS_s104
                                 + 1.30958 w_k1_FDX_s104
                                 + 0.806862 w_k1_IBM_s104
                                 + 1.27945 w_k1_INTC_s104
                                 + 1.18392 w_k1_KO_s104
                                 + 0.846897 w_k1_MAT_s104
                                 + 1.19328 w_k1_MSFT_s104
                                 + 1.06341 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.09514 w_k1_ADM_s104
                                 + 0.992373 w_k1_AXP_s104
                                 + 1.12147 w_k1_DIS_s104
                                 + 1.05912 w_k1_FDX_s104
                                 + 1.08102 w_k1_IBM_s104
                                 + 0.81343 w_k1_INTC_s104 + 1.206 w_k1_KO_s104
                                 + 1.03353 w_k1_MAT_s104
                                 + 0.945936 w_k1_MSFT_s104
                                 + 1.02634 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.867698 w_k1_ADM_s104
                                 + 0.867407 w_k1_AXP_s104
                                 + 0.902244 w_k1_DIS_s104
                                 + 0.881091 w_k1_FDX_s104
                                 + 0.851051 w_k1_IBM_s104
                                 + 0.700692 w_k1_INTC_s104
                                 + 0.851273 w_k1_KO_s104
                                 + 0.707032 w_k1_MAT_s104
                                 + 0.763133 w_k1_MSFT_s104
                                 + 0.844753 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.982873 w_k1_ADM_s104
                                 + 1.29729 w_k1_AXP_s104
                                 + 1.24189 w_k1_DIS_s104
                                 + 1.22733 w_k1_FDX_s104
                                 + 1.04678 w_k1_IBM_s104
                                 + 1.11512 w_k1_INTC_s104
                                 + 1.35902 w_k1_KO_s104 + 1.07196 w_k1_MAT_s104
                                 + 1.1654 w_k1_MSFT_s104 + 1.13348 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.777138 w_k1_ADM_s104
                                 + 0.900557 w_k1_AXP_s104
                                 + 0.93146 w_k1_DIS_s104
                                 + 1.04165 w_k1_FDX_s104
                                 + 0.620143 w_k1_IBM_s104
                                 + 0.986748 w_k1_INTC_s104
                                 + 0.794676 w_k1_KO_s104
                                 + 0.585539 w_k1_MAT_s104
                                 + 0.935827 w_k1_MSFT_s104
                                 + 0.853514 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.934791 w_k1_ADM_s105
                                 + 1.20154 w_k1_AXP_s105
                                 + 0.809081 w_k1_DIS_s105
                                 + 0.936161 w_k1_FDX_s105
                                 + 1.35295 w_k1_IBM_s105
                                 + 0.94539 w_k1_INTC_s105
                                 + 1.07868 w_k1_KO_s105 + 2.09055 w_k1_MAT_s105
                                 + 0.909636 w_k1_MSFT_s105
                                 + 0.961157 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.791539 w_k1_ADM_s105
                                 + 0.662755 w_k1_AXP_s105
                                 + 0.613401 w_k1_DIS_s105
                                 + 0.738337 w_k1_FDX_s105
                                 + 0.815055 w_k1_IBM_s105
                                 + 0.694389 w_k1_INTC_s105
                                 + 0.77975 w_k1_KO_s105
                                 + 0.865223 w_k1_MAT_s105
                                 + 0.750948 w_k1_MSFT_s105
                                 + 0.81757 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.18564 w_k1_ADM_s105
                                 + 0.958034 w_k1_AXP_s105
                                 + 1.23531 w_k1_DIS_s105
                                 + 1.13132 w_k1_FDX_s105 + 1.0989 w_k1_IBM_s105
                                 + 1.20466 w_k1_INTC_s105
                                 + 1.13752 w_k1_KO_s105
                                 + 0.907328 w_k1_MAT_s105
                                 + 1.08155 w_k1_MSFT_s105 + 1.1284 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.24303 w_k1_ADM_s105
                                 + 1.40114 w_k1_AXP_s105
                                 + 1.18396 w_k1_DIS_s105
                                 + 1.23841 w_k1_FDX_s105
                                 + 0.841012 w_k1_IBM_s105
                                 + 0.989801 w_k1_INTC_s105
                                 + 0.979079 w_k1_KO_s105 + 1.4339 w_k1_MAT_s105
                                 + 1.02491 w_k1_MSFT_s105 + 1.0897 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.0264 w_k1_ADM_s105
                                 + 1.25554 w_k1_AXP_s105
                                 + 1.27862 w_k1_DIS_s105
                                 + 1.08814 w_k1_FDX_s105
                                 + 1.88161 w_k1_IBM_s105
                                 + 1.05804 w_k1_INTC_s105
                                 + 1.32555 w_k1_KO_s105 + 2.46441 w_k1_MAT_s105
                                 + 1.07015 w_k1_MSFT_s105 + 1.34 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.31535 w_k1_ADM_s105
                                 + 1.09382 w_k1_AXP_s105
                                 + 1.56705 w_k1_DIS_s105
                                 + 1.35069 w_k1_FDX_s105
                                 + 0.868849 w_k1_IBM_s105
                                 + 0.997314 w_k1_INTC_s105
                                 + 1.22126 w_k1_KO_s105
                                 + 0.932352 w_k1_MAT_s105
                                 + 1.21261 w_k1_MSFT_s105
                                 + 1.08843 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.07605 w_k1_ADM_s105
                                 + 0.996358 w_k1_AXP_s105
                                 + 0.88068 w_k1_DIS_s105
                                 + 0.92843 w_k1_FDX_s105
                                 + 0.791867 w_k1_IBM_s105
                                 + 0.816239 w_k1_INTC_s105
                                 + 0.814585 w_k1_KO_s105
                                 + 1.19402 w_k1_MAT_s105
                                 + 0.881526 w_k1_MSFT_s105
                                 + 0.946677 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.858007 w_k1_ADM_s105
                                 + 0.61063 w_k1_AXP_s105
                                 + 0.872183 w_k1_DIS_s105
                                 + 0.867223 w_k1_FDX_s105
                                 + 0.749717 w_k1_IBM_s105
                                 + 0.799311 w_k1_INTC_s105
                                 + 0.892026 w_k1_KO_s105
                                 + 0.866269 w_k1_MAT_s105
                                 + 0.916112 w_k1_MSFT_s105
                                 + 1.01204 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.600115 w_k1_ADM_s106
                                 + 0.675324 w_k1_AXP_s106
                                 + 0.477553 w_k1_DIS_s106
                                 + 0.72975 w_k1_FDX_s106
                                 + 0.579244 w_k1_IBM_s106
                                 + 0.774458 w_k1_INTC_s106
                                 + 0.600806 w_k1_KO_s106
                                 + 0.622893 w_k1_MAT_s106
                                 + 0.684802 w_k1_MSFT_s106
                                 + 0.759061 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.752032 w_k1_ADM_s106
                                 + 1.02986 w_k1_AXP_s106
                                 + 0.863305 w_k1_DIS_s106
                                 + 0.866259 w_k1_FDX_s106
                                 + 1.1454 w_k1_IBM_s106
                                 + 0.920505 w_k1_INTC_s106
                                 + 0.8475 w_k1_KO_s106 + 1.28 w_k1_MAT_s106
                                 + 0.844706 w_k1_MSFT_s106
                                 + 0.866503 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.791727 w_k1_ADM_s106
                                 + 0.850589 w_k1_AXP_s106
                                 + 0.616157 w_k1_DIS_s106
                                 + 0.721351 w_k1_FDX_s106
                                 + 0.916769 w_k1_IBM_s106
                                 + 0.673307 w_k1_INTC_s106
                                 + 0.62305 w_k1_KO_s106
                                 + 0.939754 w_k1_MAT_s106
                                 + 0.677825 w_k1_MSFT_s106
                                 + 0.749993 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.99225 w_k1_ADM_s106
                                 + 1.86189 w_k1_AXP_s106
                                 + 1.49467 w_k1_DIS_s106
                                 + 1.36611 w_k1_FDX_s106
                                 + 1.75171 w_k1_IBM_s106
                                 + 1.38736 w_k1_INTC_s106
                                 + 1.33213 w_k1_KO_s106 + 2.46328 w_k1_MAT_s106
                                 + 1.20175 w_k1_MSFT_s106
                                 + 1.29847 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.983268 w_k1_ADM_s106
                                 + 1.29574 w_k1_AXP_s106
                                 + 1.01791 w_k1_DIS_s106
                                 + 1.13773 w_k1_FDX_s106
                                 + 0.909779 w_k1_IBM_s106
                                 + 0.950386 w_k1_INTC_s106
                                 + 0.978058 w_k1_KO_s106 + 1.241 w_k1_MAT_s106
                                 + 1.06666 w_k1_MSFT_s106
                                 + 1.05574 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.914364 w_k1_ADM_s106
                                 + 0.88684 w_k1_AXP_s106
                                 + 0.79646 w_k1_DIS_s106
                                 + 0.954631 w_k1_FDX_s106
                                 + 0.745305 w_k1_IBM_s106
                                 + 0.811623 w_k1_INTC_s106
                                 + 0.728493 w_k1_KO_s106
                                 + 0.712691 w_k1_MAT_s106
                                 + 0.830574 w_k1_MSFT_s106
                                 + 0.754574 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.1248 w_k1_ADM_s106
                                 + 1.28963 w_k1_AXP_s106
                                 + 1.27664 w_k1_DIS_s106
                                 + 1.15698 w_k1_FDX_s106
                                 + 1.39601 w_k1_IBM_s106
                                 + 0.85505 w_k1_INTC_s106
                                 + 1.10122 w_k1_KO_s106 + 1.49507 w_k1_MAT_s106
                                 + 0.988626 w_k1_MSFT_s106
                                 + 1.01277 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.08348 w_k1_ADM_s106
                                 + 1.01682 w_k1_AXP_s106 + 1.0764 w_k1_DIS_s106
                                 + 1.18111 w_k1_FDX_s106
                                 + 0.774954 w_k1_IBM_s106
                                 + 1.03876 w_k1_INTC_s106
                                 + 0.90463 w_k1_KO_s106
                                 + 0.620452 w_k1_MAT_s106
                                 + 1.05687 w_k1_MSFT_s106
                                 + 0.958165 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.03991 w_k1_ADM_s107
                                 + 0.802094 w_k1_AXP_s107
                                 + 1.26508 w_k1_DIS_s107
                                 + 0.957259 w_k1_FDX_s107
                                 + 1.16436 w_k1_IBM_s107
                                 + 1.02805 w_k1_INTC_s107
                                 + 1.23052 w_k1_KO_s107 + 1.00248 w_k1_MAT_s107
                                 + 1.15954 w_k1_MSFT_s107
                                 + 1.17679 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.947807 w_k1_ADM_s107
                                 + 1.11952 w_k1_AXP_s107
                                 + 0.773398 w_k1_DIS_s107
                                 + 0.945008 w_k1_FDX_s107
                                 + 0.656088 w_k1_IBM_s107
                                 + 0.777186 w_k1_INTC_s107
                                 + 0.701871 w_k1_KO_s107
                                 + 0.624287 w_k1_MAT_s107
                                 + 0.765447 w_k1_MSFT_s107
                                 + 0.944254 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.36271 w_k1_ADM_s107
                                 + 1.37446 w_k1_AXP_s107
                                 + 1.67249 w_k1_DIS_s107
                                 + 1.39047 w_k1_FDX_s107
                                 + 1.02048 w_k1_IBM_s107
                                 + 1.1511 w_k1_INTC_s107 + 1.13464 w_k1_KO_s107
                                 + 1.05025 w_k1_MAT_s107
                                 + 1.31191 w_k1_MSFT_s107
                                 + 1.08347 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.15757 w_k1_ADM_s107
                                 + 1.50814 w_k1_AXP_s107
                                 + 1.35992 w_k1_DIS_s107
                                 + 1.17687 w_k1_FDX_s107
                                 + 1.07472 w_k1_IBM_s107
                                 + 1.07959 w_k1_INTC_s107
                                 + 1.16592 w_k1_KO_s107 + 1.60878 w_k1_MAT_s107
                                 + 1.0827 w_k1_MSFT_s107 + 1.18663 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.25165 w_k1_ADM_s107
                                 + 1.01966 w_k1_AXP_s107
                                 + 0.972743 w_k1_DIS_s107
                                 + 0.93153 w_k1_FDX_s107
                                 + 0.977041 w_k1_IBM_s107
                                 + 0.994199 w_k1_INTC_s107
                                 + 0.996374 w_k1_KO_s107
                                 + 0.905845 w_k1_MAT_s107
                                 + 0.957372 w_k1_MSFT_s107
                                 + 1.11877 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.605979 w_k1_ADM_s107
                                 + 0.742003 w_k1_AXP_s107
                                 + 0.651599 w_k1_DIS_s107
                                 + 0.753148 w_k1_FDX_s107
                                 + 0.780839 w_k1_IBM_s107
                                 + 1.00754 w_k1_INTC_s107
                                 + 0.774823 w_k1_KO_s107
                                 + 0.977481 w_k1_MAT_s107
                                 + 0.833955 w_k1_MSFT_s107
                                 + 0.881119 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.861754 w_k1_ADM_s107
                                 + 0.809022 w_k1_AXP_s107
                                 + 0.935879 w_k1_DIS_s107
                                 + 0.817396 w_k1_FDX_s107
                                 + 0.964533 w_k1_IBM_s107
                                 + 0.776807 w_k1_INTC_s107
                                 + 1.02743 w_k1_KO_s107 + 1.12207 w_k1_MAT_s107
                                 + 0.800894 w_k1_MSFT_s107
                                 + 0.953002 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.06773 w_k1_ADM_s107
                                 + 0.922854 w_k1_AXP_s107
                                 + 1.17212 w_k1_DIS_s107
                                 + 1.14818 w_k1_FDX_s107
                                 + 0.667664 w_k1_IBM_s107
                                 + 1.04827 w_k1_INTC_s107
                                 + 0.843128 w_k1_KO_s107
                                 + 1.01351 w_k1_MAT_s107
                                 + 1.13714 w_k1_MSFT_s107
                                 + 1.01774 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.839505 w_k1_ADM_s110
                                 + 0.923425 w_k1_AXP_s110
                                 + 0.98642 w_k1_DIS_s110
                                 + 0.979062 w_k1_FDX_s110
                                 + 1.07823 w_k1_IBM_s110
                                 + 0.92634 w_k1_INTC_s110
                                 + 1.08066 w_k1_KO_s110 + 1.07905 w_k1_MAT_s110
                                 + 0.906775 w_k1_MSFT_s110
                                 + 0.943707 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.27782 w_k1_ADM_s110
                                 + 1.16063 w_k1_AXP_s110
                                 + 1.01679 w_k1_DIS_s110
                                 + 1.15489 w_k1_FDX_s110
                                 + 0.787865 w_k1_IBM_s110
                                 + 1.01998 w_k1_INTC_s110
                                 + 0.928694 w_k1_KO_s110
                                 + 0.804729 w_k1_MAT_s110
                                 + 1.14227 w_k1_MSFT_s110
                                 + 0.977023 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.33353 w_k1_ADM_s110
                                 + 1.47108 w_k1_AXP_s110
                                 + 1.51779 w_k1_DIS_s110 + 1.3791 w_k1_FDX_s110
                                 + 1.21888 w_k1_IBM_s110
                                 + 1.27276 w_k1_INTC_s110
                                 + 1.35903 w_k1_KO_s110 + 1.49821 w_k1_MAT_s110
                                 + 1.39154 w_k1_MSFT_s110
                                 + 1.30722 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.824655 w_k1_ADM_s110
                                 + 0.699833 w_k1_AXP_s110
                                 + 0.605975 w_k1_DIS_s110
                                 + 0.778986 w_k1_FDX_s110
                                 + 0.691319 w_k1_IBM_s110
                                 + 0.832634 w_k1_INTC_s110
                                 + 0.709077 w_k1_KO_s110
                                 + 0.464401 w_k1_MAT_s110
                                 + 0.764581 w_k1_MSFT_s110
                                 + 0.838491 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.984446 w_k1_ADM_s110
                                 + 1.04974 w_k1_AXP_s110
                                 + 1.11932 w_k1_DIS_s110
                                 + 0.963643 w_k1_FDX_s110
                                 + 1.46023 w_k1_IBM_s110
                                 + 0.977326 w_k1_INTC_s110
                                 + 1.29004 w_k1_KO_s110 + 1.27058 w_k1_MAT_s110
                                 + 1.03324 w_k1_MSFT_s110
                                 + 1.09695 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.09371 w_k1_ADM_s110
                                 + 1.26009 w_k1_AXP_s110
                                 + 1.14165 w_k1_DIS_s110 + 1.0996 w_k1_FDX_s110
                                 + 1.05368 w_k1_IBM_s110
                                 + 0.993607 w_k1_INTC_s110
                                 + 1.02564 w_k1_KO_s110
                                 + 0.948208 w_k1_MAT_s110
                                 + 1.05251 w_k1_MSFT_s110
                                 + 1.09568 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.7517 w_k1_ADM_s110
                                 + 0.682766 w_k1_AXP_s110
                                 + 0.779182 w_k1_DIS_s110
                                 + 0.743569 w_k1_FDX_s110
                                 + 1.12028 w_k1_IBM_s110
                                 + 0.9039 w_k1_INTC_s110 + 1.08292 w_k1_KO_s110
                                 + 0.922017 w_k1_MAT_s110
                                 + 0.864128 w_k1_MSFT_s110
                                 + 0.93213 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.923446 w_k1_ADM_s110
                                 + 0.939392 w_k1_AXP_s110
                                 + 0.534594 w_k1_DIS_s110
                                 + 0.747849 w_k1_FDX_s110
                                 + 1.06784 w_k1_IBM_s110
                                 + 0.59678 w_k1_INTC_s110
                                 + 0.748934 w_k1_KO_s110
                                 + 0.967454 w_k1_MAT_s110
                                 + 0.556173 w_k1_MSFT_s110
                                 + 0.849529 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.15564 w_k1_ADM_s111
                                 + 1.00228 w_k1_AXP_s111
                                 + 1.47723 w_k1_DIS_s111
                                 + 1.27663 w_k1_FDX_s111
                                 + 0.778599 w_k1_IBM_s111
                                 + 1.25651 w_k1_INTC_s111 + 1.0525 w_k1_KO_s111
                                 + 0.65511 w_k1_MAT_s111
                                 + 1.11675 w_k1_MSFT_s111
                                 + 1.10547 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.781623 w_k1_ADM_s111
                                 + 0.792653 w_k1_AXP_s111
                                 + 0.694352 w_k1_DIS_s111
                                 + 0.859625 w_k1_FDX_s111
                                 + 1.01635 w_k1_IBM_s111
                                 + 0.998986 w_k1_INTC_s111
                                 + 0.796716 w_k1_KO_s111
                                 + 0.870498 w_k1_MAT_s111
                                 + 0.86116 w_k1_MSFT_s111
                                 + 0.845124 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.920236 w_k1_ADM_s111
                                 + 0.877952 w_k1_AXP_s111
                                 + 0.987786 w_k1_DIS_s111
                                 + 1.12034 w_k1_FDX_s111
                                 + 0.692445 w_k1_IBM_s111
                                 + 1.04297 w_k1_INTC_s111
                                 + 0.856389 w_k1_KO_s111
                                 + 0.677703 w_k1_MAT_s111
                                 + 0.999315 w_k1_MSFT_s111
                                 + 0.968066 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.09345 w_k1_ADM_s111
                                 + 1.4306 w_k1_AXP_s111 + 1.26802 w_k1_DIS_s111
                                 + 1.29331 w_k1_FDX_s111
                                 + 1.09824 w_k1_IBM_s111
                                 + 1.38696 w_k1_INTC_s111
                                 + 1.17583 w_k1_KO_s111 + 1.61526 w_k1_MAT_s111
                                 + 1.33305 w_k1_MSFT_s111
                                 + 1.17998 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.790648 w_k1_ADM_s111
                                 + 0.721202 w_k1_AXP_s111
                                 + 0.621249 w_k1_DIS_s111
                                 + 0.8762 w_k1_FDX_s111
                                 + 0.651421 w_k1_IBM_s111
                                 + 0.946273 w_k1_INTC_s111
                                 + 0.709651 w_k1_KO_s111
                                 + 0.810444 w_k1_MAT_s111
                                 + 0.879916 w_k1_MSFT_s111
                                 + 0.793259 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.06344 w_k1_ADM_s111
                                 + 1.25954 w_k1_AXP_s111
                                 + 1.57277 w_k1_DIS_s111
                                 + 1.19471 w_k1_FDX_s111
                                 + 1.72767 w_k1_IBM_s111
                                 + 1.60203 w_k1_INTC_s111
                                 + 1.54972 w_k1_KO_s111 + 1.80351 w_k1_MAT_s111
                                 + 1.51593 w_k1_MSFT_s111
                                 + 1.20606 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.620066 w_k1_ADM_s111
                                 + 0.945962 w_k1_AXP_s111
                                 + 0.914958 w_k1_DIS_s111
                                 + 0.90461 w_k1_FDX_s111
                                 + 1.57211 w_k1_IBM_s111
                                 + 1.09852 w_k1_INTC_s111
                                 + 1.20577 w_k1_KO_s111 + 1.81459 w_k1_MAT_s111
                                 + 0.950775 w_k1_MSFT_s111
                                 + 0.974197 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.811922 w_k1_ADM_s111
                                 + 1.11973 w_k1_AXP_s111
                                 + 1.08278 w_k1_DIS_s111
                                 + 1.08389 w_k1_FDX_s111
                                 + 1.06435 w_k1_IBM_s111
                                 + 1.27437 w_k1_INTC_s111
                                 + 1.01742 w_k1_KO_s111
                                 + 0.751014 w_k1_MAT_s111
                                 + 1.07015 w_k1_MSFT_s111
                                 + 1.00307 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.09769 w_k1_ADM_s112
                                 + 1.10712 w_k1_AXP_s112
                                 + 1.21402 w_k1_DIS_s112
                                 + 1.15854 w_k1_FDX_s112
                                 + 0.903406 w_k1_IBM_s112
                                 + 1.08359 w_k1_INTC_s112
                                 + 0.879609 w_k1_KO_s112 + 1.5105 w_k1_MAT_s112
                                 + 1.11171 w_k1_MSFT_s112
                                 + 1.03049 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.3031 w_k1_ADM_s112
                                 + 1.43637 w_k1_AXP_s112 + 1.5557 w_k1_DIS_s112
                                 + 1.33466 w_k1_FDX_s112
                                 + 1.48931 w_k1_IBM_s112
                                 + 1.37586 w_k1_INTC_s112
                                 + 1.39734 w_k1_KO_s112 + 1.68206 w_k1_MAT_s112
                                 + 1.2689 w_k1_MSFT_s112 + 1.1887 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.04622 w_k1_ADM_s112
                                 + 0.876505 w_k1_AXP_s112
                                 + 0.864141 w_k1_DIS_s112
                                 + 0.866681 w_k1_FDX_s112
                                 + 0.990729 w_k1_IBM_s112
                                 + 0.806569 w_k1_INTC_s112
                                 + 0.843565 w_k1_KO_s112
                                 + 1.26722 w_k1_MAT_s112
                                 + 0.897884 w_k1_MSFT_s112
                                 + 0.916929 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.53639 w_k1_ADM_s112
                                 + 1.20015 w_k1_AXP_s112
                                 + 1.09954 w_k1_DIS_s112
                                 + 1.20764 w_k1_FDX_s112
                                 + 0.690834 w_k1_IBM_s112
                                 + 0.932573 w_k1_INTC_s112
                                 + 0.873803 w_k1_KO_s112
                                 + 0.926111 w_k1_MAT_s112
                                 + 1.06722 w_k1_MSFT_s112
                                 + 0.975476 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.28561 w_k1_ADM_s112
                                 + 1.33515 w_k1_AXP_s112
                                 + 1.69032 w_k1_DIS_s112
                                 + 1.52475 w_k1_FDX_s112
                                 + 0.915799 w_k1_IBM_s112
                                 + 1.25753 w_k1_INTC_s112
                                 + 1.14559 w_k1_KO_s112 + 1.32072 w_k1_MAT_s112
                                 + 1.37159 w_k1_MSFT_s112
                                 + 1.18232 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.21441 w_k1_ADM_s112
                                 + 1.33702 w_k1_AXP_s112
                                 + 1.25114 w_k1_DIS_s112
                                 + 1.18026 w_k1_FDX_s112 + 1.3978 w_k1_IBM_s112
                                 + 1.10215 w_k1_INTC_s112
                                 + 1.27904 w_k1_KO_s112 + 2.01832 w_k1_MAT_s112
                                 + 1.13164 w_k1_MSFT_s112
                                 + 1.15815 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.915968 w_k1_ADM_s112
                                 + 1.0659 w_k1_AXP_s112
                                 + 0.921956 w_k1_DIS_s112
                                 + 0.916267 w_k1_FDX_s112
                                 + 1.28898 w_k1_IBM_s112
                                 + 0.743291 w_k1_INTC_s112
                                 + 1.05694 w_k1_KO_s112 + 1.95444 w_k1_MAT_s112
                                 + 0.850834 w_k1_MSFT_s112
                                 + 1.01423 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.980189 w_k1_ADM_s112
                                 + 0.784422 w_k1_AXP_s112
                                 + 0.686411 w_k1_DIS_s112
                                 + 0.859802 w_k1_FDX_s112
                                 + 0.672219 w_k1_IBM_s112
                                 + 0.835484 w_k1_INTC_s112
                                 + 0.730654 w_k1_KO_s112
                                 + 0.806065 w_k1_MAT_s112
                                 + 0.809255 w_k1_MSFT_s112
                                 + 0.826783 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.24239 w_k1_ADM_s113
                                 + 1.14128 w_k1_AXP_s113
                                 + 1.53907 w_k1_DIS_s113
                                 + 1.37918 w_k1_FDX_s113
                                 + 1.26172 w_k1_IBM_s113
                                 + 1.23911 w_k1_INTC_s113
                                 + 1.34352 w_k1_KO_s113 + 1.26879 w_k1_MAT_s113
                                 + 1.343 w_k1_MSFT_s113 + 1.22657 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.961102 w_k1_ADM_s113
                                 + 0.920599 w_k1_AXP_s113
                                 + 1.06863 w_k1_DIS_s113
                                 + 1.00948 w_k1_FDX_s113
                                 + 1.13682 w_k1_IBM_s113
                                 + 1.04779 w_k1_INTC_s113
                                 + 1.20279 w_k1_KO_s113 + 1.02714 w_k1_MAT_s113
                                 + 1.0954 w_k1_MSFT_s113 + 1.05769 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.13269 w_k1_ADM_s113
                                 + 1.08418 w_k1_AXP_s113
                                 + 0.985835 w_k1_DIS_s113
                                 + 1.0746 w_k1_FDX_s113 + 1.2138 w_k1_IBM_s113
                                 + 1.02313 w_k1_INTC_s113
                                 + 1.02241 w_k1_KO_s113 + 1.22181 w_k1_MAT_s113
                                 + 1.032 w_k1_MSFT_s113 + 0.988569 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.91719 w_k1_ADM_s113
                                 + 0.906117 w_k1_AXP_s113
                                 + 0.791812 w_k1_DIS_s113
                                 + 0.854842 w_k1_FDX_s113
                                 + 1.22659 w_k1_IBM_s113
                                 + 1.06152 w_k1_INTC_s113
                                 + 0.971923 w_k1_KO_s113
                                 + 0.937081 w_k1_MAT_s113
                                 + 0.896433 w_k1_MSFT_s113
                                 + 0.884995 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.878136 w_k1_ADM_s113
                                 + 0.737295 w_k1_AXP_s113
                                 + 0.631666 w_k1_DIS_s113
                                 + 0.810445 w_k1_FDX_s113
                                 + 0.777274 w_k1_IBM_s113
                                 + 0.75334 w_k1_INTC_s113
                                 + 0.863209 w_k1_KO_s113
                                 + 0.612513 w_k1_MAT_s113
                                 + 0.835461 w_k1_MSFT_s113
                                 + 0.879075 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.15107 w_k1_ADM_s113
                                 + 1.26438 w_k1_AXP_s113
                                 + 1.15795 w_k1_DIS_s113
                                 + 1.07241 w_k1_FDX_s113
                                 + 1.57937 w_k1_IBM_s113
                                 + 1.16943 w_k1_INTC_s113
                                 + 1.31294 w_k1_KO_s113 + 1.66906 w_k1_MAT_s113
                                 + 1.09756 w_k1_MSFT_s113
                                 + 1.15327 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.00179 w_k1_ADM_s113
                                 + 1.09141 w_k1_AXP_s113
                                 + 1.16503 w_k1_DIS_s113
                                 + 1.02639 w_k1_FDX_s113
                                 + 0.972361 w_k1_IBM_s113
                                 + 1.0158 w_k1_INTC_s113
                                 + 0.968102 w_k1_KO_s113
                                 + 0.843597 w_k1_MAT_s113
                                 + 1.14744 w_k1_MSFT_s113
                                 + 1.03355 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.0358 w_k1_ADM_s113
                                 + 0.955082 w_k1_AXP_s113
                                 + 1.13694 w_k1_DIS_s113
                                 + 1.18174 w_k1_FDX_s113
                                 + 0.758462 w_k1_IBM_s113
                                 + 0.964256 w_k1_INTC_s113
                                 + 0.937549 w_k1_KO_s113
                                 + 0.721795 w_k1_MAT_s113
                                 + 1.0939 w_k1_MSFT_s113 + 1.0767 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.00158 w_k1_ADM_s114
                                 + 1.03198 w_k1_AXP_s114
                                 + 0.925607 w_k1_DIS_s114
                                 + 1.00559 w_k1_FDX_s114
                                 + 0.971304 w_k1_IBM_s114
                                 + 0.957214 w_k1_INTC_s114
                                 + 1.0457 w_k1_KO_s114 + 1.15904 w_k1_MAT_s114
                                 + 0.963636 w_k1_MSFT_s114
                                 + 1.08823 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.30442 w_k1_ADM_s114
                                 + 1.40919 w_k1_AXP_s114
                                 + 1.37879 w_k1_DIS_s114
                                 + 1.30131 w_k1_FDX_s114
                                 + 1.18948 w_k1_IBM_s114
                                 + 1.27959 w_k1_INTC_s114
                                 + 1.26198 w_k1_KO_s114 + 1.33795 w_k1_MAT_s114
                                 + 1.22838 w_k1_MSFT_s114
                                 + 1.25638 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.915319 w_k1_ADM_s114
                                 + 1.02736 w_k1_AXP_s114
                                 + 1.23552 w_k1_DIS_s114
                                 + 1.01258 w_k1_FDX_s114
                                 + 1.56294 w_k1_IBM_s114
                                 + 1.18238 w_k1_INTC_s114
                                 + 1.43253 w_k1_KO_s114 + 1.51752 w_k1_MAT_s114
                                 + 1.20593 w_k1_MSFT_s114 + 1.2505 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.0346 w_k1_ADM_s114
                                 + 0.757794 w_k1_AXP_s114
                                 + 0.773586 w_k1_DIS_s114
                                 + 0.901998 w_k1_FDX_s114
                                 + 0.700157 w_k1_IBM_s114
                                 + 0.934321 w_k1_INTC_s114
                                 + 0.821856 w_k1_KO_s114
                                 + 0.677637 w_k1_MAT_s114
                                 + 0.91196 w_k1_MSFT_s114
                                 + 0.91672 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.08976 w_k1_ADM_s114
                                 + 1.17603 w_k1_AXP_s114
                                 + 1.18553 w_k1_DIS_s114 + 1.1287 w_k1_FDX_s114
                                 + 0.791896 w_k1_IBM_s114
                                 + 0.982244 w_k1_INTC_s114
                                 + 1.04837 w_k1_KO_s114
                                 + 0.771459 w_k1_MAT_s114
                                 + 1.02733 w_k1_MSFT_s114
                                 + 1.02573 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.97837 w_k1_ADM_s114
                                 + 0.944348 w_k1_AXP_s114
                                 + 0.911812 w_k1_DIS_s114
                                 + 0.829147 w_k1_FDX_s114
                                 + 1.10683 w_k1_IBM_s114
                                 + 0.779637 w_k1_INTC_s114
                                 + 1.08079 w_k1_KO_s114 + 1.3155 w_k1_MAT_s114
                                 + 0.855203 w_k1_MSFT_s114
                                 + 1.01675 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.861017 w_k1_ADM_s114
                                 + 0.673711 w_k1_AXP_s114
                                 + 0.904142 w_k1_DIS_s114
                                 + 0.839424 w_k1_FDX_s114
                                 + 1.15404 w_k1_IBM_s114
                                 + 0.816412 w_k1_INTC_s114
                                 + 1.08053 w_k1_KO_s114
                                 + 0.892976 w_k1_MAT_s114
                                 + 0.813644 w_k1_MSFT_s114
                                 + 0.915289 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.28092 w_k1_ADM_s114
                                 + 1.20747 w_k1_AXP_s114
                                 + 1.47068 w_k1_DIS_s114
                                 + 1.24724 w_k1_FDX_s114
                                 + 0.956327 w_k1_IBM_s114
                                 + 1.02572 w_k1_INTC_s114
                                 + 1.10613 w_k1_KO_s114 + 1.00556 w_k1_MAT_s114
                                 + 1.10976 w_k1_MSFT_s114
                                 + 1.06046 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.16646 w_k1_ADM_s115
                                 + 1.12457 w_k1_AXP_s115
                                 + 1.19831 w_k1_DIS_s115
                                 + 1.15678 w_k1_FDX_s115
                                 + 1.02556 w_k1_IBM_s115
                                 + 0.973228 w_k1_INTC_s115
                                 + 1.02676 w_k1_KO_s115 + 1.20698 w_k1_MAT_s115
                                 + 1.13449 w_k1_MSFT_s115
                                 + 1.17278 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.667673 w_k1_ADM_s115
                                 + 0.881853 w_k1_AXP_s115
                                 + 0.80985 w_k1_DIS_s115
                                 + 0.796043 w_k1_FDX_s115
                                 + 1.05249 w_k1_IBM_s115
                                 + 0.630811 w_k1_INTC_s115
                                 + 0.813731 w_k1_KO_s115
                                 + 1.04517 w_k1_MAT_s115
                                 + 0.639966 w_k1_MSFT_s115
                                 + 0.808232 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.26701 w_k1_ADM_s115
                                 + 1.66122 w_k1_AXP_s115
                                 + 1.13306 w_k1_DIS_s115
                                 + 1.25564 w_k1_FDX_s115
                                 + 1.19152 w_k1_IBM_s115
                                 + 0.759144 w_k1_INTC_s115
                                 + 0.994487 w_k1_KO_s115
                                 + 1.84849 w_k1_MAT_s115
                                 + 0.857591 w_k1_MSFT_s115
                                 + 1.15211 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.068 w_k1_ADM_s115
                                 + 0.963769 w_k1_AXP_s115
                                 + 1.02845 w_k1_DIS_s115
                                 + 0.853067 w_k1_FDX_s115
                                 + 1.54451 w_k1_IBM_s115
                                 + 0.697031 w_k1_INTC_s115
                                 + 1.10982 w_k1_KO_s115 + 1.78836 w_k1_MAT_s115
                                 + 0.751739 w_k1_MSFT_s115
                                 + 1.03025 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.48211 w_k1_ADM_s115
                                 + 1.68019 w_k1_AXP_s115
                                 + 2.01878 w_k1_DIS_s115
                                 + 1.56292 w_k1_FDX_s115
                                 + 1.30898 w_k1_IBM_s115
                                 + 1.22817 w_k1_INTC_s115
                                 + 1.53289 w_k1_KO_s115 + 1.23593 w_k1_MAT_s115
                                 + 1.16496 w_k1_MSFT_s115
                                 + 1.40437 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.793397 w_k1_ADM_s115
                                 + 0.768098 w_k1_AXP_s115
                                 + 0.58238 w_k1_DIS_s115
                                 + 0.692437 w_k1_FDX_s115
                                 + 0.827267 w_k1_IBM_s115
                                 + 0.766177 w_k1_INTC_s115
                                 + 0.785724 w_k1_KO_s115
                                 + 0.863974 w_k1_MAT_s115
                                 + 0.727276 w_k1_MSFT_s115
                                 + 0.827349 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.839698 w_k1_ADM_s115
                                 + 0.850367 w_k1_AXP_s115
                                 + 0.7818 w_k1_DIS_s115
                                 + 0.883702 w_k1_FDX_s115
                                 + 0.655958 w_k1_IBM_s115
                                 + 0.725457 w_k1_INTC_s115
                                 + 0.808297 w_k1_KO_s115
                                 + 0.883985 w_k1_MAT_s115
                                 + 0.760201 w_k1_MSFT_s115
                                 + 0.904045 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.25999 w_k1_ADM_s115
                                 + 1.29497 w_k1_AXP_s115
                                 + 1.08664 w_k1_DIS_s115 + 1.2222 w_k1_FDX_s115
                                 + 0.770815 w_k1_IBM_s115
                                 + 0.803476 w_k1_INTC_s115
                                 + 0.791031 w_k1_KO_s115
                                 + 0.973114 w_k1_MAT_s115
                                 + 0.843542 w_k1_MSFT_s115
                                 + 0.975748 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.920641 w_k1_ADM_s116
                                 + 1.07689 w_k1_AXP_s116
                                 + 0.779548 w_k1_DIS_s116
                                 + 0.938264 w_k1_FDX_s116
                                 + 0.973657 w_k1_IBM_s116
                                 + 1.01919 w_k1_INTC_s116
                                 + 0.74148 w_k1_KO_s116 + 1.11669 w_k1_MAT_s116
                                 + 0.765079 w_k1_MSFT_s116
                                 + 1.04208 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.15558 w_k1_ADM_s116
                                 + 1.0619 w_k1_AXP_s116 + 1.58471 w_k1_DIS_s116
                                 + 1.21656 w_k1_FDX_s116
                                 + 1.20074 w_k1_IBM_s116
                                 + 1.19224 w_k1_INTC_s116
                                 + 1.31013 w_k1_KO_s116
                                 + 0.861537 w_k1_MAT_s116
                                 + 1.18982 w_k1_MSFT_s116
                                 + 1.22531 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.679997 w_k1_ADM_s116
                                 + 0.707102 w_k1_AXP_s116
                                 + 0.576303 w_k1_DIS_s116
                                 + 0.669881 w_k1_FDX_s116
                                 + 1.0144 w_k1_IBM_s116
                                 + 0.892333 w_k1_INTC_s116
                                 + 0.797917 w_k1_KO_s116
                                 + 0.951922 w_k1_MAT_s116
                                 + 0.689689 w_k1_MSFT_s116
                                 + 0.96438 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.44342 w_k1_ADM_s116
                                 + 1.63734 w_k1_AXP_s116
                                 + 1.36127 w_k1_DIS_s116
                                 + 1.29544 w_k1_FDX_s116
                                 + 1.49389 w_k1_IBM_s116
                                 + 1.54667 w_k1_INTC_s116 + 1.3874 w_k1_KO_s116
                                 + 1.25015 w_k1_MAT_s116
                                 + 1.15532 w_k1_MSFT_s116
                                 + 1.40455 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.797389 w_k1_ADM_s116
                                 + 0.568393 w_k1_AXP_s116
                                 + 0.886148 w_k1_DIS_s116
                                 + 0.899676 w_k1_FDX_s116
                                 + 0.808792 w_k1_IBM_s116
                                 + 0.964315 w_k1_INTC_s116
                                 + 0.763004 w_k1_KO_s116
                                 + 0.460663 w_k1_MAT_s116
                                 + 0.906684 w_k1_MSFT_s116
                                 + 0.898596 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.08487 w_k1_ADM_s116
                                 + 0.863944 w_k1_AXP_s116
                                 + 0.981801 w_k1_DIS_s116
                                 + 0.902394 w_k1_FDX_s116
                                 + 1.14995 w_k1_IBM_s116
                                 + 1.0252 w_k1_INTC_s116 + 1.08019 w_k1_KO_s116
                                 + 0.943847 w_k1_MAT_s116
                                 + 0.971304 w_k1_MSFT_s116
                                 + 1.1152 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.3861 w_k1_ADM_s116
                                 + 1.41511 w_k1_AXP_s116 + 1.2779 w_k1_DIS_s116
                                 + 1.27379 w_k1_FDX_s116
                                 + 0.889848 w_k1_IBM_s116
                                 + 1.19517 w_k1_INTC_s116 + 1.0675 w_k1_KO_s116
                                 + 1.06475 w_k1_MAT_s116
                                 + 1.12276 w_k1_MSFT_s116
                                 + 1.14791 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.18499 w_k1_ADM_s116
                                 + 1.0502 w_k1_AXP_s116 + 1.00541 w_k1_DIS_s116
                                 + 1.12507 w_k1_FDX_s116
                                 + 0.751316 w_k1_IBM_s116
                                 + 1.24131 w_k1_INTC_s116
                                 + 0.937465 w_k1_KO_s116
                                 + 0.665923 w_k1_MAT_s116
                                 + 1.11203 w_k1_MSFT_s116
                                 + 1.06932 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.689812 w_k1_ADM_s117
                                 + 0.823447 w_k1_AXP_s117
                                 + 0.644873 w_k1_DIS_s117
                                 + 0.737208 w_k1_FDX_s117
                                 + 0.926923 w_k1_IBM_s117
                                 + 0.731451 w_k1_INTC_s117
                                 + 0.85298 w_k1_KO_s117
                                 + 0.968249 w_k1_MAT_s117
                                 + 0.948196 w_k1_MSFT_s117
                                 + 0.930989 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.08375 w_k1_ADM_s117
                                 + 1.49357 w_k1_AXP_s117
                                 + 1.56268 w_k1_DIS_s117
                                 + 1.27689 w_k1_FDX_s117
                                 + 1.53029 w_k1_IBM_s117
                                 + 1.37552 w_k1_INTC_s117
                                 + 1.52049 w_k1_KO_s117 + 1.74225 w_k1_MAT_s117
                                 + 1.46109 w_k1_MSFT_s117
                                 + 1.23685 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.984111 w_k1_ADM_s117
                                 + 1.18432 w_k1_AXP_s117
                                 + 1.31215 w_k1_DIS_s117
                                 + 1.27875 w_k1_FDX_s117
                                 + 1.00574 w_k1_IBM_s117
                                 + 1.08182 w_k1_INTC_s117
                                 + 1.25349 w_k1_KO_s117 + 1.2275 w_k1_MAT_s117
                                 + 1.34493 w_k1_MSFT_s117
                                 + 1.11167 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.762524 w_k1_ADM_s117
                                 + 0.99838 w_k1_AXP_s117
                                 + 0.831314 w_k1_DIS_s117
                                 + 0.872722 w_k1_FDX_s117
                                 + 1.02229 w_k1_IBM_s117
                                 + 0.756346 w_k1_INTC_s117
                                 + 0.884494 w_k1_KO_s117
                                 + 0.984231 w_k1_MAT_s117
                                 + 0.8903 w_k1_MSFT_s117 + 1.035 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.794624 w_k1_ADM_s117
                                 + 1.3283 w_k1_AXP_s117 + 1.13384 w_k1_DIS_s117
                                 + 1.02853 w_k1_FDX_s117
                                 + 1.42576 w_k1_IBM_s117
                                 + 0.825686 w_k1_INTC_s117
                                 + 1.17697 w_k1_KO_s117 + 1.37273 w_k1_MAT_s117
                                 + 1.01975 w_k1_MSFT_s117
                                 + 1.06511 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.707597 w_k1_ADM_s117
                                 + 0.852058 w_k1_AXP_s117
                                 + 0.703504 w_k1_DIS_s117
                                 + 0.877004 w_k1_FDX_s117
                                 + 0.657471 w_k1_IBM_s117
                                 + 0.889626 w_k1_INTC_s117
                                 + 0.704551 w_k1_KO_s117
                                 + 0.623994 w_k1_MAT_s117
                                 + 0.914975 w_k1_MSFT_s117
                                 + 0.854006 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.08223 w_k1_ADM_s117
                                 + 1.3859 w_k1_AXP_s117
                                 + 0.918324 w_k1_DIS_s117
                                 + 1.03232 w_k1_FDX_s117
                                 + 0.962276 w_k1_IBM_s117
                                 + 1.0005 w_k1_INTC_s117 + 1.00552 w_k1_KO_s117
                                 + 1.19933 w_k1_MAT_s117
                                 + 0.969138 w_k1_MSFT_s117
                                 + 1.00479 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.08823 w_k1_ADM_s117
                                 + 1.18327 w_k1_AXP_s117
                                 + 0.923165 w_k1_DIS_s117
                                 + 1.11054 w_k1_FDX_s117
                                 + 0.742406 w_k1_IBM_s117
                                 + 0.982775 w_k1_INTC_s117
                                 + 0.899207 w_k1_KO_s117
                                 + 0.937491 w_k1_MAT_s117
                                 + 1.01846 w_k1_MSFT_s117
                                 + 0.99624 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0146894283567101 w_k1_ADM_s000
                                 + 0.0384483422452726 w_k1_ADM_s001
                                 + 0.00927280362017457 w_k1_ADM_s002
                                 + 0.0148472312833603 w_k1_ADM_s003
                                 + 0.0286620830943541 w_k1_ADM_s004
                                 + 0.0269810740432565 w_k1_ADM_s005
                                 + 0.0142543485008581 w_k1_ADM_s006
                                 + 0.0201681588218097 w_k1_ADM_s007
                                 + 0.0327449125713175 w_k1_ADM_s010
                                 + 0.021082263568051 w_k1_ADM_s011
                                 + 0.00712542317242466 w_k1_ADM_s012
                                 + 0.0166287012554752 w_k1_ADM_s013
                                 + 0.021203367818671 w_k1_ADM_s014
                                 + 0.00839314096855921 w_k1_ADM_s015
                                 + 0.0122275618779179 w_k1_ADM_s016
                                 + 0.0214417247019699 w_k1_ADM_s017
                                 + 0.0131669869049198 w_k1_ADM_s100
                                 + 0.0265090801558848 w_k1_ADM_s101
                                 + 0.0213539758155097 w_k1_ADM_s102
                                 + 0.0182930768115821 w_k1_ADM_s103
                                 + 0.0531932498517898 w_k1_ADM_s104
                                 + 0.0325397414497842 w_k1_ADM_s105
                                 + 0.0282915063928357 w_k1_ADM_s106
                                 + 0.0152746888620361 w_k1_ADM_s107
                                 + 0.076324706944578 w_k1_ADM_s110
                                 + 0.0378222384275003 w_k1_ADM_s111
                                 + 0.123743601335726 w_k1_ADM_s112
                                 + 0.0568787284282801 w_k1_ADM_s113
                                 + 0.0547889202785328 w_k1_ADM_s114
                                 + 0.0290210955545648 w_k1_ADM_s115
                                 + 0.0797073902238401 w_k1_ADM_s116
                                 + 0.0401483055401843 w_k1_ADM_s117
                                 + 0.0126755714908721 w_k1_AXP_s000
                                 + 0.0403033666355185 w_k1_AXP_s001
                                 + 0.00925716945763801 w_k1_AXP_s002
                                 + 0.0148474880917192 w_k1_AXP_s003
                                 + 0.0273576999813181 w_k1_AXP_s004
                                 + 0.0235927498346582 w_k1_AXP_s005
                                 + 0.01505304824886 w_k1_AXP_s006
                                 + 0.020360880507972 w_k1_AXP_s007
                                 + 0.0362538663114716 w_k1_AXP_s010
                                 + 0.020753933612555 w_k1_AXP_s011
                                 + 0.00847987093357676 w_k1_AXP_s012
                                 + 0.0201911057621698 w_k1_AXP_s013
                                 + 0.0207409378309245 w_k1_AXP_s014
                                 + 0.0100683339455188 w_k1_AXP_s015
                                 + 0.0144806508672155 w_k1_AXP_s016
                                 + 0.0209383653827287 w_k1_AXP_s017
                                 + 0.0145172936907088 w_k1_AXP_s100
                                 + 0.0273089166907299 w_k1_AXP_s101
                                 + 0.0211470394752746 w_k1_AXP_s102
                                 + 0.0209157381473709 w_k1_AXP_s103
                                 + 0.0572174962401847 w_k1_AXP_s104
                                 + 0.0316656529863907 w_k1_AXP_s105
                                 + 0.0339364454203984 w_k1_AXP_s106
                                 + 0.0154707380477542 w_k1_AXP_s107
                                 + 0.0779488462093993 w_k1_AXP_s110
                                 + 0.0414010011456743 w_k1_AXP_s111
                                 + 0.115646668126141 w_k1_AXP_s112
                                 + 0.0541786114463189 w_k1_AXP_s113
                                 + 0.0505226377037896 w_k1_AXP_s114
                                 + 0.0304996128990545 w_k1_AXP_s115
                                 + 0.0785796084534526 w_k1_AXP_s116
                                 + 0.0522393663514298 w_k1_AXP_s117
                                 + 0.0137128561369109 w_k1_DIS_s000
                                 + 0.0404262551541529 w_k1_DIS_s001
                                 + 0.0100917073811016 w_k1_DIS_s002
                                 + 0.0146884071583698 w_k1_DIS_s003
                                 + 0.0279290447252895 w_k1_DIS_s004
                                 + 0.024815589259451 w_k1_DIS_s005
                                 + 0.0128015065066641 w_k1_DIS_s006
                                 + 0.0209922192939014 w_k1_DIS_s007
                                 + 0.0376846937796137 w_k1_DIS_s010
                                 + 0.0212983877900263 w_k1_DIS_s011
                                 + 0.00843580096959066 w_k1_DIS_s012
                                 + 0.0203646623899022 w_k1_DIS_s013
                                 + 0.0233878953990314 w_k1_DIS_s014
                                 + 0.00951220330859064 w_k1_DIS_s015
                                 + 0.0128775826057797 w_k1_DIS_s016
                                 + 0.0202432995085172 w_k1_DIS_s017
                                 + 0.0146023417066431 w_k1_DIS_s100
                                 + 0.0265223524011951 w_k1_DIS_s101
                                 + 0.020257483721735 w_k1_DIS_s102
                                 + 0.0194893529408647 w_k1_DIS_s103
                                 + 0.0580264104789909 w_k1_DIS_s104
                                 + 0.0311795433374199 w_k1_DIS_s105
                                 + 0.0292713244716703 w_k1_DIS_s106
                                 + 0.0154667895302323 w_k1_DIS_s107
                                 + 0.0757953323623193 w_k1_DIS_s110
                                 + 0.0424605705836056 w_k1_DIS_s111
                                 + 0.112179004805978 w_k1_DIS_s112
                                 + 0.0553114931484054 w_k1_DIS_s113
                                 + 0.0537573386095399 w_k1_DIS_s114
                                 + 0.0289001258726766 w_k1_DIS_s115
                                 + 0.075091534118213 w_k1_DIS_s116
                                 + 0.0470107879419754 w_k1_DIS_s117
                                 + 0.0136788862571425 w_k1_FDX_s000
                                 + 0.0396755347651414 w_k1_FDX_s001
                                 + 0.00966837658389661 w_k1_FDX_s002
                                 + 0.0151891695812599 w_k1_FDX_s003
                                 + 0.028975872873689 w_k1_FDX_s004
                                 + 0.0232508649392606 w_k1_FDX_s005
                                 + 0.0131713800409653 w_k1_FDX_s006
                                 + 0.0224516036351465 w_k1_FDX_s007
                                 + 0.0360523498708735 w_k1_FDX_s010
                                 + 0.0203900529141296 w_k1_FDX_s011
                                 + 0.00808703443993226 w_k1_FDX_s012
                                 + 0.0194060533154117 w_k1_FDX_s013
                                 + 0.0215501713421305 w_k1_FDX_s014
                                 + 0.00944350739996124 w_k1_FDX_s015
                                 + 0.0128758275866249 w_k1_FDX_s016
                                 + 0.0202945003950164 w_k1_FDX_s017
                                 + 0.013564550785121 w_k1_FDX_s100
                                 + 0.027716611708879 w_k1_FDX_s101
                                 + 0.0206318542304695 w_k1_FDX_s102
                                 + 0.0182479914387705 w_k1_FDX_s103
                                 + 0.0565202751396386 w_k1_FDX_s104
                                 + 0.031232429112181 w_k1_FDX_s105
                                 + 0.0312420846147122 w_k1_FDX_s106
                                 + 0.0143986325925314 w_k1_FDX_s107
                                 + 0.0763084173764848 w_k1_FDX_s110
                                 + 0.0445775689974062 w_k1_FDX_s111
                                 + 0.113506229903227 w_k1_FDX_s112
                                 + 0.0568832432978744 w_k1_FDX_s113
                                 + 0.0523894921400931 w_k1_FDX_s114
                                 + 0.0285249921058403 w_k1_FDX_s115
                                 + 0.0750944684125658 w_k1_FDX_s116
                                 + 0.0472929071101005 w_k1_FDX_s117
                                 + 0.0136972914844882 w_k1_IBM_s000
                                 + 0.0388200010706277 w_k1_IBM_s001
                                 + 0.00989457778599416 w_k1_IBM_s002
                                 + 0.0160793039848781 w_k1_IBM_s003
                                 + 0.028294006400995 w_k1_IBM_s004
                                 + 0.0239411340069133 w_k1_IBM_s005
                                 + 0.0135952177609713 w_k1_IBM_s006
                                 + 0.0218214767517222 w_k1_IBM_s007
                                 + 0.0377655219942699 w_k1_IBM_s010
                                 + 0.0202585850758136 w_k1_IBM_s011
                                 + 0.00881463144763449 w_k1_IBM_s012
                                 + 0.0203956916104008 w_k1_IBM_s013
                                 + 0.0220479083308124 w_k1_IBM_s014
                                 + 0.00995361391486347 w_k1_IBM_s015
                                 + 0.013111951498798 w_k1_IBM_s016
                                 + 0.0211544532207931 w_k1_IBM_s017
                                 + 0.0141594593920318 w_k1_IBM_s100
                                 + 0.0291921103927655 w_k1_IBM_s101
                                 + 0.0200251743058194 w_k1_IBM_s102
                                 + 0.0196988846903206 w_k1_IBM_s103
                                 + 0.0565636308572807 w_k1_IBM_s104
                                 + 0.0300014259591296 w_k1_IBM_s105
                                 + 0.030652240206086 w_k1_IBM_s106
                                 + 0.0133157821289578 w_k1_IBM_s107
                                 + 0.0762516953719221 w_k1_IBM_s110
                                 + 0.0428563289094784 w_k1_IBM_s111
                                 + 0.109425300033947 w_k1_IBM_s112
                                 + 0.0588427643624341 w_k1_IBM_s113
                                 + 0.0509922181478794 w_k1_IBM_s114
                                 + 0.0293378609381996 w_k1_IBM_s115
                                 + 0.0780295801865482 w_k1_IBM_s116
                                 + 0.0480058397537829 w_k1_IBM_s117
                                 + 0.0127654372988309 w_k1_INTC_s000
                                 + 0.0441535062766169 w_k1_INTC_s001
                                 + 0.00867501144843276 w_k1_INTC_s002
                                 + 0.0138839682609622 w_k1_INTC_s003
                                 + 0.0317222187941048 w_k1_INTC_s004
                                 + 0.0259398640862599 w_k1_INTC_s005
                                 + 0.0134443471320911 w_k1_INTC_s006
                                 + 0.0211334934420096 w_k1_INTC_s007
                                 + 0.0362636197222425 w_k1_INTC_s010
                                 + 0.0194885912319786 w_k1_INTC_s011
                                 + 0.00744344160251251 w_k1_INTC_s012
                                 + 0.0206566638929074 w_k1_INTC_s013
                                 + 0.0194386794881913 w_k1_INTC_s014
                                 + 0.00999207657025343 w_k1_INTC_s015
                                 + 0.0126053983503681 w_k1_INTC_s016
                                 + 0.0204012354380574 w_k1_INTC_s017
                                 + 0.0141176871968514 w_k1_INTC_s100
                                 + 0.0276676085425024 w_k1_INTC_s101
                                 + 0.0218611720938587 w_k1_INTC_s102
                                 + 0.0165873310262223 w_k1_INTC_s103
                                 + 0.0478785003219042 w_k1_INTC_s104
                                 + 0.0277435789474627 w_k1_INTC_s105
                                 + 0.0273589912119858 w_k1_INTC_s106
                                 + 0.0142834920030179 w_k1_INTC_s107
                                 + 0.072552652389301 w_k1_INTC_s110
                                 + 0.049287727557943 w_k1_INTC_s111
                                 + 0.101972681379597 w_k1_INTC_s112
                                 + 0.0554993711448641 w_k1_INTC_s113
                                 + 0.0509674879173802 w_k1_INTC_s114
                                 + 0.0232494225747509 w_k1_INTC_s115
                                 + 0.0833058891378557 w_k1_INTC_s116
                                 + 0.0426181202986194 w_k1_INTC_s117
                                 + 0.0134978886103721 w_k1_KO_s000
                                 + 0.0388820896634063 w_k1_KO_s001
                                 + 0.00932414870887864 w_k1_KO_s002
                                 + 0.0139932670055978 w_k1_KO_s003
                                 + 0.0297106546939609 w_k1_KO_s004
                                 + 0.0240337187840094 w_k1_KO_s005
                                 + 0.0137568779378925 w_k1_KO_s006
                                 + 0.0195606855256292 w_k1_KO_s007
                                 + 0.0376313062762829 w_k1_KO_s010
                                 + 0.0185805207973715 w_k1_KO_s011
                                 + 0.00884253577418137 w_k1_KO_s012
                                 + 0.0213315322863211 w_k1_KO_s013
                                 + 0.0215283446349088 w_k1_KO_s014
                                 + 0.00956707235104885 w_k1_KO_s015
                                 + 0.0133882758262032 w_k1_KO_s016
                                 + 0.0198951988166868 w_k1_KO_s017
                                 + 0.0141601286432116 w_k1_KO_s100
                                 + 0.0274059004946476 w_k1_KO_s101
                                 + 0.0209601980624429 w_k1_KO_s102
                                 + 0.0179562070901614 w_k1_KO_s103
                                 + 0.0584882624150717 w_k1_KO_s104
                                 + 0.0298997103631305 w_k1_KO_s105
                                 + 0.0273394676384013 w_k1_KO_s106
                                 + 0.0141980617735467 w_k1_KO_s107
                                 + 0.0776751883493419 w_k1_KO_s110
                                 + 0.0414394026432306 w_k1_KO_s111
                                 + 0.104764456837752 w_k1_KO_s112
                                 + 0.0580011538690373 w_k1_KO_s113
                                 + 0.0544694845454293 w_k1_KO_s114
                                 + 0.0273404521874139 w_k1_KO_s115
                                 + 0.074510409603039 w_k1_KO_s116
                                 + 0.0480551708461203 w_k1_KO_s117
                                 + 0.0102302460610424 w_k1_MAT_s000
                                 + 0.0353409945686865 w_k1_MAT_s001
                                 + 0.0109421208466054 w_k1_MAT_s002
                                 + 0.0166520338445551 w_k1_MAT_s003
                                 + 0.0307797053504442 w_k1_MAT_s004
                                 + 0.0201677548587976 w_k1_MAT_s005
                                 + 0.0147865212800803 w_k1_MAT_s006
                                 + 0.024847884330715 w_k1_MAT_s007
                                 + 0.0415089126128123 w_k1_MAT_s010
                                 + 0.0179824503538833 w_k1_MAT_s011
                                 + 0.0103681320564054 w_k1_MAT_s012
                                 + 0.0219144666785677 w_k1_MAT_s013
                                 + 0.0195654557117235 w_k1_MAT_s014
                                 + 0.00899816210447055 w_k1_MAT_s015
                                 + 0.0155221311196462 w_k1_MAT_s016
                                 + 0.0170236207331887 w_k1_MAT_s017
                                 + 0.0151200520099566 w_k1_MAT_s100
                                 + 0.0267703363467829 w_k1_MAT_s101
                                 + 0.0178336285265425 w_k1_MAT_s102
                                 + 0.0194794122755443 w_k1_MAT_s103
                                 + 0.0498319421503955 w_k1_MAT_s104
                                 + 0.0401076137354715 w_k1_MAT_s105
                                 + 0.0348980138040761 w_k1_MAT_s106
                                 + 0.0155165266474183 w_k1_MAT_s107
                                 + 0.0753465405344386 w_k1_MAT_s110
                                 + 0.0425152556491771 w_k1_MAT_s111
                                 + 0.149930713627754 w_k1_MAT_s112
                                 + 0.0531063422895946 w_k1_MAT_s113
                                 + 0.0518095704049329 w_k1_MAT_s114
                                 + 0.0338707381602639 w_k1_MAT_s115
                                 + 0.0712392565326357 w_k1_MAT_s116
                                 + 0.0519365784288813 w_k1_MAT_s117
                                 + 0.0133156266924802 w_k1_MSFT_s000
                                 + 0.0353246455639008 w_k1_MSFT_s001
                                 + 0.0106388464624664 w_k1_MSFT_s002
                                 + 0.0151674448326649 w_k1_MSFT_s003
                                 + 0.0282435011939484 w_k1_MSFT_s004
                                 + 0.0250781310434809 w_k1_MSFT_s005
                                 + 0.0123060263514131 w_k1_MSFT_s006
                                 + 0.0208482087397646 w_k1_MSFT_s007
                                 + 0.035125721492035 w_k1_MSFT_s010
                                 + 0.0208126036330594 w_k1_MSFT_s011
                                 + 0.00818767708581948 w_k1_MSFT_s012
                                 + 0.0183635734678529 w_k1_MSFT_s013
                                 + 0.0184536371860864 w_k1_MSFT_s014
                                 + 0.0104751863980053 w_k1_MSFT_s015
                                 + 0.0135216843522792 w_k1_MSFT_s016
                                 + 0.0184824179719945 w_k1_MSFT_s017
                                 + 0.0144759495976088 w_k1_MSFT_s100
                                 + 0.0266360499541982 w_k1_MSFT_s101
                                 + 0.0212025233834181 w_k1_MSFT_s102
                                 + 0.0191229941874052 w_k1_MSFT_s103
                                 + 0.0506907184593715 w_k1_MSFT_s104
                                 + 0.0293087618897649 w_k1_MSFT_s105
                                 + 0.0282827003234648 w_k1_MSFT_s106
                                 + 0.0142599488793932 w_k1_MSFT_s107
                                 + 0.0752796746686224 w_k1_MSFT_s110
                                 + 0.0444079078497631 w_k1_MSFT_s111
                                 + 0.108001765957106 w_k1_MSFT_s112
                                 + 0.0577640197858113 w_k1_MSFT_s113
                                 + 0.0514182265378303 w_k1_MSFT_s114
                                 + 0.0244729785953902 w_k1_MSFT_s115
                                 + 0.0713635366875809 w_k1_MSFT_s116
                                 + 0.0491537527399785 w_k1_MSFT_s117
                                 + 0.0132870075342158 w_k1_PG_s000
                                 + 0.036738380043167 w_k1_PG_s001
                                 + 0.00891578469107111 w_k1_PG_s002
                                 + 0.0160695747096224 w_k1_PG_s003
                                 + 0.0297454884158732 w_k1_PG_s004
                                 + 0.0232348108288584 w_k1_PG_s005
                                 + 0.0126287364401882 w_k1_PG_s006
                                 + 0.0207619674188156 w_k1_PG_s007
                                 + 0.0353940455704756 w_k1_PG_s010
                                 + 0.0193461304197631 w_k1_PG_s011
                                 + 0.00867857716773212 w_k1_PG_s012
                                 + 0.0193463781158934 w_k1_PG_s013
                                 + 0.0207810129939375 w_k1_PG_s014
                                 + 0.00969779420085523 w_k1_PG_s015
                                 + 0.0135698268628661 w_k1_PG_s016
                                 + 0.0182067892815481 w_k1_PG_s017
                                 + 0.0132903144667471 w_k1_PG_s100
                                 + 0.0270996524057575 w_k1_PG_s101
                                 + 0.0212468656536493 w_k1_PG_s102
                                 + 0.0187565799488498 w_k1_PG_s103
                                 + 0.0533316589926478 w_k1_PG_s104
                                 + 0.0313350543096018 w_k1_PG_s105
                                 + 0.0284537245770475 w_k1_PG_s106
                                 + 0.0153029542513366 w_k1_PG_s107
                                 + 0.0752521574895569 w_k1_PG_s110
                                 + 0.0415647262246985 w_k1_PG_s111
                                 + 0.108109928243905 w_k1_PG_s112
                                 + 0.0567869401720853 w_k1_PG_s113
                                 + 0.0536033692977077 w_k1_PG_s114
                                 + 0.0288946277834762 w_k1_PG_s115
                                 + 0.0822903262034719 w_k1_PG_s116
                                 + 0.0474117661468401 w_k1_PG_s117 >= 1.15
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
