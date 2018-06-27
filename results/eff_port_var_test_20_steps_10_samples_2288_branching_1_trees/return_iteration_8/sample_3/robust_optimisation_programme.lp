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
 w_k1_ADM_s0_return_constraint:  1.0223 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.05165 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.22609 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.776804 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.988 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.04099 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.894549 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.10917 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.773454 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.885522 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.16175 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.09096 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.796708 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.27911 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.02796 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.3904 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.79051 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.05889 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.29373 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.07504 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.914363 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.03742 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.953411 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.51782 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.862672 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.17268 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.24999 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.0651 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.985521 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.881801 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.981738 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.764634 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.06403 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.839165 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.835431 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.06362 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.876642 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.62188 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.982065 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.11478 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.2416 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.840484 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.25958 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.956711 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.864409 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.26962 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.779031 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.21363 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.18562 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.64852 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.882922 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.16661 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.14939 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.68515 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.565322 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.995263 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.23774 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.863885 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.635855 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.818501 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.684889 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.3404 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.782619 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.23348 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.69183 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.83031 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.03502 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.892783 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.43526 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.833773 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.24818 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.904583 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.695185 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.29467 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.73152 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 2.10868 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.01417 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.12304 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.31954 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.736875 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.25818 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.91788 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.798869 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.2657 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.759086 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.20444 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.27776 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.85189 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.869992 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.38133 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.13237 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.77756 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.503375 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.0294 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.23282 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.10902 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.628048 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.796638 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.569593 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.64034 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.841846 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.47633 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.56177 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.72944 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.05646 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.822455 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.60219 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.643357 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.45605 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.944841 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.742125 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.12021 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.770877 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 2.46511 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.01354 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.0231 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.19892 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.852357 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.1164 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.966919 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.877309 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.24136 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.851839 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.0812 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.2564 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.43138 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.806036 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.40614 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.91545 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.50615 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.597511 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.928658 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.27257 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.0815 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.770287 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.847659 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.749109 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.36246 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.821345 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.15711 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.35448 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.807104 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.960439 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.889541 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.25482 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.752045 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.35091 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.932508 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.8037 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.11713 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.882817 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.75093 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.865961 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.27939 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.29649 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.88216 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.40512 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.889202 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.07898 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.828153 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.729229 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.14257 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.02312 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.63847 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.54021 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.612853 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.63751 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.35616 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.885479 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.21172 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.94268 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.732676 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.611846 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.934297 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.638231 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.853153 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.19229 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.79588 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.23079 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.990185 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.34075 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.836879 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.65125 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.07643 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.934412 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.3143 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.845107 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.11373 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.642952 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.41384 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.975062 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.07996 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.32314 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.911063 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.18456 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.865003 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.809425 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.00675 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.814632 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.96026 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.00143 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.13926 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.751077 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.07938 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.965084 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.3065 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.601688 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.947121 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.22753 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.02593 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.817109 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.893479 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.752017 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.14088 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.731374 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.834392 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.24735 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.8715 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.902563 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.90724 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.26754 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.922909 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.26923 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.10911 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.8815 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.12887 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.973754 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.61492 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.915009 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.14707 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.23339 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.846627 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.3155 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.932016 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.943182 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.988909 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.857377 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.27498 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.10757 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.6417 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.21408 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.01688 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.29808 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.31993 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.7613 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.1928 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.06664 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.865774 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.73098 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.965462 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.59337 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.05464 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.901461 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.43762 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.14628 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.875171 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.07725 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.846907 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.55503 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.854909 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.12436 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.02461 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.767875 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.03755 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.767762 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.63385 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.896539 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.2039 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.22005 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.990058 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.44857 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.903492 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.18025 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.945066 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.724682 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.4639 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.969454 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 2.39727 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.57709 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.856988 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.23624 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.51043 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.618246 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.991956 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.757008 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.553445 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.51756 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.807154 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.557532 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.07395 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.25332 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 2.29574 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.38192 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.991227 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.47683 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.887641 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.88621 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.24592 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.922608 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.09077 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.664422 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.70312 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.569987 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.64726 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.00607 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.03835 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.22053 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.828865 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.18333 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.986499 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.810644 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.12646 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.822095 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.05139 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.07665 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.39799 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.786649 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.31035 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.947051 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.47357 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.686812 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.02732 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.22386 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.00017 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.882618 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.992676 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.676899 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.30201 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.758964 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.04916 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.16688 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.754705 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.947329 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.913361 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.23151 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.823383 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.22563 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.893478 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.814031 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.981424 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.826044 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.55303 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.98848 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.03706 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.15203 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.879684 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.15155 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.959772 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.875867 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.993759 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.857999 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.1263 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.14074 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.31216 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.996293 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.04953 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.05965 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.25626 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.726136 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.955795 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.02448 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.990739 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.85014 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.886781 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.781622 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.15576 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.901691 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.24381 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.23872 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.915138 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.993438 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.962667 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.29222 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.900946 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.16351 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.03505 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.898358 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.15136 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.93071 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.56624 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.309000539937467 z_k1_s0000
                                 - 0.813001914158438 z_k1_s0001
                                 - 0.841999647614949 z_k1_s0002
                                 - 0.348000382313749 z_k1_s0003
                                 - 0.441000796842826 z_k1_s0004
                                 - 0.694999201733774 z_k1_s0005
                                 - 0.686000278109821 z_k1_s0006
                                 - 1.27400206163452 z_k1_s0007
                                 - 0.275000449089977 z_k1_s0010
                                 - 0.327999011149329 z_k1_s0011
                                 - 0.138999698566176 z_k1_s0012
                                 - 0.216999610831981 z_k1_s0013
                                 - 0.200999734346548 z_k1_s0014
                                 - 0.515999103051124 z_k1_s0015
                                 - 0.428998745862943 z_k1_s0016
                                 - 0.466999995713809 z_k1_s0017
                                 - 0.963999806741482 z_k1_s0020
                                 - 0.606000161061948 z_k1_s0021
                                 - 0.436000173875589 z_k1_s0022
                                 - 0.65599989886972 z_k1_s0023
                                 - 0.809000399877406 z_k1_s0024
                                 - 1.24400038862485 z_k1_s0025
                                 - 0.59000012781147 z_k1_s0026
                                 - 2.01699851708884 z_k1_s0027
                                 - 0.240000452285776 z_k1_s0030
                                 - 0.247000467007864 z_k1_s0031
                                 - 0.20600017089244 z_k1_s0032
                                 - 1.01200128884902 z_k1_s0033
                                 - 0.458001802789377 z_k1_s0034
                                 - 0.35300048005628 z_k1_s0035
                                 - 0.880999858962374 z_k1_s0036
                                 - 0.276000318113316 z_k1_s0037
                                 - 0.236999873293334 z_k1_s0040
                                 - 0.363998163797203 z_k1_s0041
                                 - 0.503999304948246 z_k1_s0042
                                 - 0.192999900616931 z_k1_s0043
                                 - 0.241999699620357 z_k1_s0044
                                 - 0.537000260104637 z_k1_s0045
                                 - 0.294999659885122 z_k1_s0046
                                 - 0.629999131185308 z_k1_s0047
                                 - 0.255000132998961 z_k1_s0050
                                 - 0.169999861866133 z_k1_s0051
                                 - 0.17799977265048 z_k1_s0052
                                 - 0.706998607712081 z_k1_s0053
                                 - 0.322999261265987 z_k1_s0054
                                 - 0.203999879599328 z_k1_s0055
                                 - 0.250999383807345 z_k1_s0056
                                 - 0.179999920446447 z_k1_s0057
                                 - 0.175999830468214 z_k1_s0060
                                 - 0.193000657956 z_k1_s0061
                                 - 0.320000204487285 z_k1_s0062
                                 - 0.13799998243522 z_k1_s0063
                                 - 0.145999940547436 z_k1_s0064
                                 - 0.358000616519865 z_k1_s0065
                                 - 0.189000020900376 z_k1_s0066
                                 - 0.35999905504906 z_k1_s0067
                                 - 0.241999249567595 z_k1_s0070
                                 - 0.135999775078999 z_k1_s0071
                                 - 0.200000239938802 z_k1_s0072
                                 - 0.229998776531992 z_k1_s0073
                                 - 0.141999835197079 z_k1_s0074
                                 - 0.180000039545201 z_k1_s0075
                                 - 0.137999795118359 z_k1_s0076
                                 - 0.495998972170794 z_k1_s0077
                                 - 1.16900242764499 z_k1_s0100
                                 - 0.632001928924185 z_k1_s0101
                                 - 0.296000903420188 z_k1_s0102
                                 - 0.387001409211721 z_k1_s0103
                                 - 0.416001725773349 z_k1_s0104
                                 - 0.286000872899235 z_k1_s0105
                                 - 0.336001481605731 z_k1_s0106
                                 - 1.03900431138128 z_k1_s0107
                                 - 0.651999103369719 z_k1_s0110
                                 - 0.232999914542903 z_k1_s0111
                                 - 0.341999457730823 z_k1_s0112
                                 - 0.433997817996347 z_k1_s0113
                                 - 0.510999381590283 z_k1_s0114
                                 - 0.634000308566777 z_k1_s0115
                                 - 0.254999785356103 z_k1_s0116
                                 - 0.603998652232652 z_k1_s0117
                                 - 0.115000057990316 z_k1_s0120
                                 - 0.176000164910448 z_k1_s0121
                                 - 0.189999538209717 z_k1_s0122
                                 - 0.22000020613806 z_k1_s0123
                                 - 0.20600083283879 z_k1_s0124
                                 - 0.408000951020052 z_k1_s0125
                                 - 0.145000100318243 z_k1_s0126
                                 - 0.104000125883465 z_k1_s0127
                                 - 0.593998955724886 z_k1_s0130
                                 - 0.644998849577555 z_k1_s0131
                                 - 0.495999479742761 z_k1_s0132
                                 - 1.20999799368013 z_k1_s0133
                                 - 0.301999704291727 z_k1_s0134
                                 - 1.17799941728798 z_k1_s0135
                                 - 1.45999952166043 z_k1_s0136
                                 - 0.643999261320947 z_k1_s0137
                                 - 1.95700502654907 z_k1_s0140
                                 - 0.55500188887985 z_k1_s0141
                                 - 0.740003864910522 z_k1_s0142
                                 - 0.666002132015415 z_k1_s0143
                                 - 0.552001427442384 z_k1_s0144
                                 - 0.549001639206945 z_k1_s0145
                                 - 1.28100247841215 z_k1_s0146
                                 - 0.432001146398475 z_k1_s0147
                                 - 0.50700018600495 z_k1_s0150
                                 - 0.549999707351189 z_k1_s0151
                                 - 0.339999622635539 z_k1_s0152
                                 - 0.483998532310641 z_k1_s0153
                                 - 0.672000962609168 z_k1_s0154
                                 - 1.0860004751466 z_k1_s0155
                                 - 0.286999438665613 z_k1_s0156
                                 - 0.395999270653539 z_k1_s0157
                                 - 0.598004342993028 z_k1_s0160
                                 - 0.436001735479829 z_k1_s0161
                                 - 0.494004310128071 z_k1_s0162
                                 - 0.24300100345613 z_k1_s0163
                                 - 0.322001316006824 z_k1_s0164
                                 - 0.520004318344311 z_k1_s0165
                                 - 0.391001657351422 z_k1_s0166
                                 - 1.15000707374982 z_k1_s0167
                                 - 0.26199939856941 z_k1_s0170
                                 - 0.891997966014909 z_k1_s0171
                                 - 0.610998246362987 z_k1_s0172
                                 - 0.270999480161274 z_k1_s0173
                                 - 0.277999245955572 z_k1_s0174
                                 - 1.04599995747444 z_k1_s0175
                                 - 0.426999108424005 z_k1_s0176
                                 - 0.677999895600232 z_k1_s0177
                                 - 0.104000012903113 z_k1_s1000
                                 - 0.342000799815984 z_k1_s1001
                                 - 0.117000014516002 z_k1_s1002
                                 - 0.270999654930239 z_k1_s1003
                                 - 0.249000409585327 z_k1_s1004
                                 - 0.132000066869334 z_k1_s1005
                                 - 0.124000028007557 z_k1_s1006
                                 - 0.301999785007124 z_k1_s1007
                                 - 0.176000395945913 z_k1_s1010
                                 - 0.271999477373304 z_k1_s1011
                                 - 0.212998839415208 z_k1_s1012
                                 - 0.306999564976166 z_k1_s1013
                                 - 0.109999740467245 z_k1_s1014
                                 - 0.220000104933199 z_k1_s1015
                                 - 0.190999764919952 z_k1_s1016
                                 - 0.0709998211379723 z_k1_s1017
                                 - 0.12000020243159 z_k1_s1020
                                 - 0.104000139673993 z_k1_s1021
                                 - 0.0990001670060617 z_k1_s1022
                                 - 0.107999752987807 z_k1_s1023
                                 - 0.0820000667948158 z_k1_s1024
                                 - 0.116000052616996 z_k1_s1025
                                 - 0.193999754996905 z_k1_s1026
                                 - 0.250000779399666 z_k1_s1027
                                 - 0.115000410596579 z_k1_s1030
                                 - 0.120000241665881 z_k1_s1031
                                 - 0.115000410596579 z_k1_s1032
                                 - 0.0910000758632271 z_k1_s1033
                                 - 0.0729998248132131 z_k1_s1034
                                 - 0.0740000774272493 z_k1_s1035
                                 - 0.0540000371496024 z_k1_s1036
                                 - 0.0740000774272493 z_k1_s1037
                                 - 0.156999550282942 z_k1_s1040
                                 - 0.162998966212566 z_k1_s1041
                                 - 0.203999919550925 z_k1_s1042
                                 - 0.227999816263065 z_k1_s1043
                                 - 0.204999981705408 z_k1_s1044
                                 - 0.138999547999064 z_k1_s1045
                                 - 0.287998760545685 z_k1_s1046
                                 - 0.210998759636847 z_k1_s1047
                                 - 0.31900182373496 z_k1_s1050
                                 - 0.113000574151523 z_k1_s1051
                                 - 0.244001537033093 z_k1_s1052
                                 - 0.119000570427539 z_k1_s1053
                                 - 0.264001968955367 z_k1_s1054
                                 - 0.122000635215881 z_k1_s1055
                                 - 0.0850003693623379 z_k1_s1056
                                 - 0.0670003805342894 z_k1_s1057
                                 - 0.123999643081181 z_k1_s1060
                                 - 0.186000008122289 z_k1_s1061
                                 - 0.135000093556583 z_k1_s1062
                                 - 0.191000212883835 z_k1_s1063
                                 - 0.0710000814112707 z_k1_s1064
                                 - 0.104000128436236 z_k1_s1065
                                 - 0.107000142593061 z_k1_s1066
                                 - 0.169000398934065 z_k1_s1067
                                 - 0.490001697092093 z_k1_s1070
                                 - 0.21300053683348 z_k1_s1071
                                 - 0.520001989322624 z_k1_s1072
                                 - 0.20400044916432 z_k1_s1073
                                 - 0.407001811389598 z_k1_s1074
                                 - 0.39800141611978 z_k1_s1075
                                 - 0.591002373334223 z_k1_s1076
                                 - 0.253000618873529 z_k1_s1077
                                 - 0.128999296533663 z_k1_s1100
                                 - 0.307000082368993 z_k1_s1101
                                 - 0.121999415775938 z_k1_s1102
                                 - 0.0639999887837933 z_k1_s1103
                                 - 0.119000296878912 z_k1_s1104
                                 - 0.203000027970404 z_k1_s1105
                                 - 0.143999539013986 z_k1_s1106
                                 - 0.0739999180375824 z_k1_s1107
                                 - 0.399999850310429 z_k1_s1110
                                 - 0.199000462730419 z_k1_s1111
                                 - 0.216000456368612 z_k1_s1112
                                 - 0.492000890283789 z_k1_s1113
                                 - 0.279000701392995 z_k1_s1114
                                 - 0.253000442522327 z_k1_s1115
                                 - 0.276999627739481 z_k1_s1116
                                 - 0.569999786692361 z_k1_s1117
                                 - 0.33499996326305 z_k1_s1120
                                 - 0.533001152771686 z_k1_s1121
                                 - 1.15399996385896 z_k1_s1122
                                 - 0.235000113832258 z_k1_s1123
                                 - 0.394999783841545 z_k1_s1124
                                 - 1.02600094049137 z_k1_s1125
                                 - 0.485000405509089 z_k1_s1126
                                 - 0.290999797905553 z_k1_s1127
                                 - 0.936998872901093 z_k1_s1130
                                 - 0.339999126127096 z_k1_s1131
                                 - 0.372000415297032 z_k1_s1132
                                 - 0.529998457336777 z_k1_s1133
                                 - 0.195999640597636 z_k1_s1134
                                 - 0.373999796628036 z_k1_s1135
                                 - 0.387999074936635 z_k1_s1136
                                 - 0.471998353779803 z_k1_s1137
                                 - 0.375999853263075 z_k1_s1140
                                 - 0.300999850727106 z_k1_s1141
                                 - 0.92899754284416 z_k1_s1142
                                 - 0.886001408066872 z_k1_s1143
                                 - 0.429999786753009 z_k1_s1144
                                 - 0.574000748325575 z_k1_s1145
                                 - 0.324999868622607 z_k1_s1146
                                 - 0.450000869415374 z_k1_s1147
                                 - 0.531002608203084 z_k1_s1150
                                 - 0.180000788978632 z_k1_s1151
                                 - 0.326001729985833 z_k1_s1152
                                 - 0.156000537320898 z_k1_s1153
                                 - 0.231000805036752 z_k1_s1154
                                 - 0.370000930169991 z_k1_s1155
                                 - 0.171000786144847 z_k1_s1156
                                 - 0.476001777215597 z_k1_s1157
                                 - 0.538999788846373 z_k1_s1160
                                 - 0.728000056260444 z_k1_s1161
                                 - 0.353000318611249 z_k1_s1162
                                 - 0.538999788846373 z_k1_s1163
                                 - 0.5499986219643 z_k1_s1164
                                 - 0.625001023417066 z_k1_s1165
                                 - 0.456000227855672 z_k1_s1166
                                 - 0.592000142058058 z_k1_s1167
                                 - 0.0589999404388326 z_k1_s1170
                                 - 0.189999493040635 z_k1_s1171
                                 - 0.154999423322792 z_k1_s1172
                                 - 0.164999885956003 z_k1_s1173
                                 - 0.0399999911347691 z_k1_s1174
                                 - 0.0589999404388326 z_k1_s1175
                                 - 0.186999664150351 z_k1_s1176
                                 - 0.178000177479497 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.18354 w_k1_ADM_s000
                                 + 1.48376 w_k1_AXP_s000
                                 + 1.39959 w_k1_DIS_s000
                                 + 1.17568 w_k1_FDX_s000
                                 + 1.77141 w_k1_IBM_s000
                                 + 1.07006 w_k1_INTC_s000
                                 + 1.40393 w_k1_KO_s000 + 1.55158 w_k1_MAT_s000
                                 + 1.15021 w_k1_MSFT_s000 + 1.145 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.972516 w_k1_ADM_s000
                                 + 1.37548 w_k1_AXP_s000
                                 + 1.28505 w_k1_DIS_s000
                                 + 1.20991 w_k1_FDX_s000
                                 + 0.866985 w_k1_IBM_s000
                                 + 1.26796 w_k1_INTC_s000
                                 + 1.09503 w_k1_KO_s000 + 0.91441 w_k1_MAT_s000
                                 + 1.31198 w_k1_MSFT_s000
                                 + 1.15325 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.810128 w_k1_ADM_s000
                                 + 0.69145 w_k1_AXP_s000
                                 + 0.698829 w_k1_DIS_s000
                                 + 0.786456 w_k1_FDX_s000
                                 + 0.641776 w_k1_IBM_s000
                                 + 0.868522 w_k1_INTC_s000
                                 + 0.801376 w_k1_KO_s000
                                 + 0.623721 w_k1_MAT_s000
                                 + 0.950325 w_k1_MSFT_s000
                                 + 0.951554 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.00048 w_k1_ADM_s000
                                 + 1.16244 w_k1_AXP_s000
                                 + 1.01909 w_k1_DIS_s000
                                 + 0.971453 w_k1_FDX_s000
                                 + 1.29313 w_k1_IBM_s000
                                 + 0.892106 w_k1_INTC_s000
                                 + 1.06104 w_k1_KO_s000 + 1.47618 w_k1_MAT_s000
                                 + 1.00785 w_k1_MSFT_s000
                                 + 1.04326 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.85124 w_k1_ADM_s000
                                 + 0.808947 w_k1_AXP_s000
                                 + 0.759865 w_k1_DIS_s000
                                 + 0.720483 w_k1_FDX_s000
                                 + 1.15419 w_k1_IBM_s000
                                 + 0.658819 w_k1_INTC_s000
                                 + 0.94021 w_k1_KO_s000
                                 + 0.919229 w_k1_MAT_s000
                                 + 0.711487 w_k1_MSFT_s000
                                 + 0.901925 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.13516 w_k1_ADM_s000
                                 + 1.33044 w_k1_AXP_s000
                                 + 1.44819 w_k1_DIS_s000
                                 + 1.20687 w_k1_FDX_s000
                                 + 1.29643 w_k1_IBM_s000
                                 + 1.05426 w_k1_INTC_s000
                                 + 1.30394 w_k1_KO_s000 + 1.16393 w_k1_MAT_s000
                                 + 1.19028 w_k1_MSFT_s000
                                 + 1.23024 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.889772 w_k1_ADM_s000
                                 + 1.11077 w_k1_AXP_s000
                                 + 1.14496 w_k1_DIS_s000
                                 + 1.03353 w_k1_FDX_s000
                                 + 1.03681 w_k1_IBM_s000
                                 + 0.990742 w_k1_INTC_s000
                                 + 1.07611 w_k1_KO_s000 + 1.20535 w_k1_MAT_s000
                                 + 1.08983 w_k1_MSFT_s000
                                 + 1.07919 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.935081 w_k1_ADM_s000
                                 + 0.843556 w_k1_AXP_s000
                                 + 0.914933 w_k1_DIS_s000
                                 + 0.839523 w_k1_FDX_s000
                                 + 0.877796 w_k1_IBM_s000
                                 + 0.874075 w_k1_INTC_s000
                                 + 0.901541 w_k1_KO_s000
                                 + 0.712201 w_k1_MAT_s000
                                 + 0.931134 w_k1_MSFT_s000
                                 + 0.970339 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.772525 w_k1_ADM_s001
                                 + 0.71216 w_k1_AXP_s001
                                 + 0.762728 w_k1_DIS_s001
                                 + 0.887903 w_k1_FDX_s001
                                 + 0.679815 w_k1_IBM_s001
                                 + 0.888478 w_k1_INTC_s001
                                 + 0.820598 w_k1_KO_s001
                                 + 0.38945 w_k1_MAT_s001
                                 + 0.821691 w_k1_MSFT_s001
                                 + 0.901676 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.23967 w_k1_ADM_s001
                                 + 1.73227 w_k1_AXP_s001
                                 + 1.73995 w_k1_DIS_s001
                                 + 1.41277 w_k1_FDX_s001 + 1.207 w_k1_IBM_s001
                                 + 1.30182 w_k1_INTC_s001
                                 + 1.26302 w_k1_KO_s001 + 1.3614 w_k1_MAT_s001
                                 + 1.23205 w_k1_MSFT_s001
                                 + 1.19499 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.716828 w_k1_ADM_s001
                                 + 0.61583 w_k1_AXP_s001
                                 + 0.529283 w_k1_DIS_s001
                                 + 0.653644 w_k1_FDX_s001
                                 + 0.793523 w_k1_IBM_s001
                                 + 0.808236 w_k1_INTC_s001
                                 + 0.762468 w_k1_KO_s001
                                 + 0.481482 w_k1_MAT_s001
                                 + 0.689312 w_k1_MSFT_s001
                                 + 0.817118 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.05555 w_k1_ADM_s001
                                 + 1.22409 w_k1_AXP_s001
                                 + 1.37176 w_k1_DIS_s001
                                 + 1.04596 w_k1_FDX_s001
                                 + 1.47751 w_k1_IBM_s001
                                 + 1.00285 w_k1_INTC_s001 + 1.4454 w_k1_KO_s001
                                 + 2.05243 w_k1_MAT_s001
                                 + 1.05336 w_k1_MSFT_s001
                                 + 1.36638 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.05229 w_k1_ADM_s001
                                 + 1.4857 w_k1_AXP_s001
                                 + 0.963647 w_k1_DIS_s001
                                 + 1.0309 w_k1_FDX_s001 + 1.15272 w_k1_IBM_s001
                                 + 0.849121 w_k1_INTC_s001
                                 + 1.07479 w_k1_KO_s001 + 1.80895 w_k1_MAT_s001
                                 + 0.818346 w_k1_MSFT_s001
                                 + 1.14533 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.92833 w_k1_ADM_s001
                                 + 1.03444 w_k1_AXP_s001
                                 + 0.926355 w_k1_DIS_s001
                                 + 1.00982 w_k1_FDX_s001
                                 + 0.821638 w_k1_IBM_s001
                                 + 0.978131 w_k1_INTC_s001
                                 + 0.996911 w_k1_KO_s001
                                 + 1.07393 w_k1_MAT_s001
                                 + 0.921986 w_k1_MSFT_s001
                                 + 1.01745 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.864281 w_k1_ADM_s001
                                 + 0.816645 w_k1_AXP_s001
                                 + 0.693279 w_k1_DIS_s001
                                 + 0.812808 w_k1_FDX_s001
                                 + 0.9612 w_k1_IBM_s001
                                 + 0.842293 w_k1_INTC_s001
                                 + 0.839407 w_k1_KO_s001
                                 + 0.845095 w_k1_MAT_s001
                                 + 0.778841 w_k1_MSFT_s001
                                 + 0.985141 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.989278 w_k1_ADM_s001
                                 + 1.17239 w_k1_AXP_s001
                                 + 1.21285 w_k1_DIS_s001
                                 + 1.13804 w_k1_FDX_s001
                                 + 0.896332 w_k1_IBM_s001
                                 + 1.17244 w_k1_INTC_s001
                                 + 1.09557 w_k1_KO_s001 + 1.06281 w_k1_MAT_s001
                                 + 1.11052 w_k1_MSFT_s001
                                 + 1.07672 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.26459 w_k1_ADM_s002
                                 + 1.19224 w_k1_AXP_s002 + 1.3144 w_k1_DIS_s002
                                 + 1.27921 w_k1_FDX_s002
                                 + 0.848574 w_k1_IBM_s002
                                 + 1.22087 w_k1_INTC_s002
                                 + 1.01253 w_k1_KO_s002
                                 + 0.641557 w_k1_MAT_s002
                                 + 1.16376 w_k1_MSFT_s002
                                 + 1.06102 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.46913 w_k1_ADM_s002
                                 + 1.43483 w_k1_AXP_s002
                                 + 1.97275 w_k1_DIS_s002
                                 + 1.44951 w_k1_FDX_s002
                                 + 1.24241 w_k1_IBM_s002
                                 + 1.13744 w_k1_INTC_s002
                                 + 1.50266 w_k1_KO_s002 + 1.73184 w_k1_MAT_s002
                                 + 1.51245 w_k1_MSFT_s002
                                 + 1.30595 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.07127 w_k1_ADM_s002
                                 + 1.04765 w_k1_AXP_s002 + 1.2767 w_k1_DIS_s002
                                 + 1.03587 w_k1_FDX_s002
                                 + 1.33702 w_k1_IBM_s002
                                 + 1.04963 w_k1_INTC_s002
                                 + 1.12486 w_k1_KO_s002 + 1.56336 w_k1_MAT_s002
                                 + 1.11987 w_k1_MSFT_s002
                                 + 1.09903 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.877422 w_k1_ADM_s002
                                 + 0.651535 w_k1_AXP_s002
                                 + 0.999239 w_k1_DIS_s002
                                 + 0.848976 w_k1_FDX_s002
                                 + 0.946817 w_k1_IBM_s002
                                 + 0.96779 w_k1_INTC_s002
                                 + 1.07512 w_k1_KO_s002
                                 + 0.600229 w_k1_MAT_s002
                                 + 1.04772 w_k1_MSFT_s002
                                 + 0.919156 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.16626 w_k1_ADM_s002
                                 + 1.55497 w_k1_AXP_s002 + 1.2079 w_k1_DIS_s002
                                 + 1.11017 w_k1_FDX_s002
                                 + 1.18291 w_k1_IBM_s002
                                 + 1.24359 w_k1_INTC_s002
                                 + 1.21277 w_k1_KO_s002 + 1.41627 w_k1_MAT_s002
                                 + 1.08757 w_k1_MSFT_s002
                                 + 1.16116 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.803613 w_k1_ADM_s002
                                 + 0.759278 w_k1_AXP_s002
                                 + 0.55124 w_k1_DIS_s002
                                 + 0.736557 w_k1_FDX_s002
                                 + 0.71674 w_k1_IBM_s002
                                 + 0.754436 w_k1_INTC_s002
                                 + 0.663024 w_k1_KO_s002
                                 + 0.725066 w_k1_MAT_s002
                                 + 0.758326 w_k1_MSFT_s002
                                 + 0.81577 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.812474 w_k1_ADM_s002
                                 + 0.987351 w_k1_AXP_s002
                                 + 0.792671 w_k1_DIS_s002
                                 + 0.805763 w_k1_FDX_s002
                                 + 1.35801 w_k1_IBM_s002
                                 + 0.672419 w_k1_INTC_s002
                                 + 0.965321 w_k1_KO_s002 + 1.5155 w_k1_MAT_s002
                                 + 0.783579 w_k1_MSFT_s002
                                 + 0.99203 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.11498 w_k1_ADM_s002
                                 + 1.02231 w_k1_AXP_s002
                                 + 0.954816 w_k1_DIS_s002
                                 + 0.980949 w_k1_FDX_s002
                                 + 0.806456 w_k1_IBM_s002
                                 + 0.852865 w_k1_INTC_s002
                                 + 0.890671 w_k1_KO_s002
                                 + 0.773085 w_k1_MAT_s002
                                 + 0.98028 w_k1_MSFT_s002
                                 + 1.04464 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.2638 w_k1_ADM_s003
                                 + 1.46564 w_k1_AXP_s003
                                 + 1.00978 w_k1_DIS_s003
                                 + 1.20625 w_k1_FDX_s003
                                 + 1.03436 w_k1_IBM_s003
                                 + 1.12945 w_k1_INTC_s003
                                 + 0.857265 w_k1_KO_s003 + 1.3433 w_k1_MAT_s003
                                 + 1.09624 w_k1_MSFT_s003
                                 + 0.983878 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.897885 w_k1_ADM_s003
                                 + 0.901066 w_k1_AXP_s003
                                 + 1.41209 w_k1_DIS_s003
                                 + 1.19123 w_k1_FDX_s003
                                 + 0.903193 w_k1_IBM_s003
                                 + 1.26745 w_k1_INTC_s003
                                 + 0.970195 w_k1_KO_s003
                                 + 0.769307 w_k1_MAT_s003
                                 + 1.3454 w_k1_MSFT_s003 + 1.08521 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.04003 w_k1_ADM_s003
                                 + 0.820233 w_k1_AXP_s003
                                 + 1.10973 w_k1_DIS_s003
                                 + 0.919082 w_k1_FDX_s003
                                 + 1.25964 w_k1_IBM_s003
                                 + 1.0375 w_k1_INTC_s003 + 1.15981 w_k1_KO_s003
                                 + 0.91361 w_k1_MAT_s003
                                 + 1.18401 w_k1_MSFT_s003
                                 + 0.987063 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.909032 w_k1_ADM_s003
                                 + 0.871996 w_k1_AXP_s003
                                 + 0.734987 w_k1_DIS_s003
                                 + 0.842015 w_k1_FDX_s003
                                 + 0.876746 w_k1_IBM_s003
                                 + 0.836589 w_k1_INTC_s003
                                 + 0.759404 w_k1_KO_s003
                                 + 0.84418 w_k1_MAT_s003
                                 + 0.815138 w_k1_MSFT_s003
                                 + 0.746842 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.13331 w_k1_ADM_s003
                                 + 1.2562 w_k1_AXP_s003 + 1.63158 w_k1_DIS_s003
                                 + 1.27712 w_k1_FDX_s003
                                 + 1.38614 w_k1_IBM_s003
                                 + 1.04318 w_k1_INTC_s003
                                 + 1.24368 w_k1_KO_s003 + 1.39552 w_k1_MAT_s003
                                 + 1.22664 w_k1_MSFT_s003
                                 + 1.07555 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.864733 w_k1_ADM_s003
                                 + 1.17382 w_k1_AXP_s003
                                 + 0.839133 w_k1_DIS_s003
                                 + 0.848303 w_k1_FDX_s003
                                 + 1.43671 w_k1_IBM_s003
                                 + 1.18243 w_k1_INTC_s003
                                 + 0.986747 w_k1_KO_s003
                                 + 1.49792 w_k1_MAT_s003
                                 + 0.817991 w_k1_MSFT_s003
                                 + 1.02509 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.1145 w_k1_ADM_s003
                                 + 0.918369 w_k1_AXP_s003
                                 + 1.02143 w_k1_DIS_s003
                                 + 1.03001 w_k1_FDX_s003
                                 + 0.945635 w_k1_IBM_s003
                                 + 1.17323 w_k1_INTC_s003
                                 + 0.871462 w_k1_KO_s003
                                 + 0.715685 w_k1_MAT_s003
                                 + 1.00124 w_k1_MSFT_s003
                                 + 0.826457 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.385 w_k1_ADM_s003
                                 + 1.33947 w_k1_AXP_s003
                                 + 1.25612 w_k1_DIS_s003
                                 + 1.14834 w_k1_FDX_s003
                                 + 1.42792 w_k1_IBM_s003
                                 + 1.0949 w_k1_INTC_s003 + 1.11188 w_k1_KO_s003
                                 + 1.18505 w_k1_MAT_s003
                                 + 1.10276 w_k1_MSFT_s003
                                 + 1.08908 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.00434 w_k1_ADM_s004
                                 + 1.02067 w_k1_AXP_s004
                                 + 0.980367 w_k1_DIS_s004
                                 + 1.13345 w_k1_FDX_s004
                                 + 0.849008 w_k1_IBM_s004
                                 + 1.10376 w_k1_INTC_s004
                                 + 1.05649 w_k1_KO_s004 + 1.19216 w_k1_MAT_s004
                                 + 1.22255 w_k1_MSFT_s004
                                 + 0.953621 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.98848 w_k1_ADM_s004
                                 + 0.943384 w_k1_AXP_s004
                                 + 0.739784 w_k1_DIS_s004
                                 + 0.913712 w_k1_FDX_s004
                                 + 0.791735 w_k1_IBM_s004
                                 + 0.848342 w_k1_INTC_s004
                                 + 0.862477 w_k1_KO_s004
                                 + 0.890039 w_k1_MAT_s004
                                 + 0.890793 w_k1_MSFT_s004
                                 + 0.829457 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.30034 w_k1_ADM_s004
                                 + 1.33119 w_k1_AXP_s004
                                 + 1.61927 w_k1_DIS_s004
                                 + 1.30342 w_k1_FDX_s004 + 1.0859 w_k1_IBM_s004
                                 + 1.38634 w_k1_INTC_s004
                                 + 1.26375 w_k1_KO_s004
                                 + 0.919537 w_k1_MAT_s004
                                 + 1.24525 w_k1_MSFT_s004
                                 + 1.09409 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.838873 w_k1_ADM_s004
                                 + 1.32455 w_k1_AXP_s004
                                 + 0.891887 w_k1_DIS_s004
                                 + 0.952147 w_k1_FDX_s004
                                 + 1.0244 w_k1_IBM_s004
                                 + 1.14985 w_k1_INTC_s004
                                 + 1.02987 w_k1_KO_s004 + 1.5185 w_k1_MAT_s004
                                 + 1.00696 w_k1_MSFT_s004
                                 + 0.935112 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.816188 w_k1_ADM_s004
                                 + 0.719807 w_k1_AXP_s004
                                 + 0.556484 w_k1_DIS_s004
                                 + 0.695557 w_k1_FDX_s004
                                 + 0.753234 w_k1_IBM_s004
                                 + 0.779945 w_k1_INTC_s004
                                 + 0.703847 w_k1_KO_s004
                                 + 0.825061 w_k1_MAT_s004
                                 + 0.711409 w_k1_MSFT_s004
                                 + 0.776227 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.899182 w_k1_ADM_s004
                                 + 0.826046 w_k1_AXP_s004
                                 + 0.868583 w_k1_DIS_s004
                                 + 0.816896 w_k1_FDX_s004
                                 + 1.08172 w_k1_IBM_s004
                                 + 0.853824 w_k1_INTC_s004
                                 + 1.14374 w_k1_KO_s004 + 1.02203 w_k1_MAT_s004
                                 + 1.04124 w_k1_MSFT_s004
                                 + 0.925727 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.933508 w_k1_ADM_s004
                                 + 0.841871 w_k1_AXP_s004
                                 + 0.779809 w_k1_DIS_s004
                                 + 0.904881 w_k1_FDX_s004
                                 + 0.643769 w_k1_IBM_s004
                                 + 1.01983 w_k1_INTC_s004
                                 + 0.80808 w_k1_KO_s004
                                 + 0.711382 w_k1_MAT_s004
                                 + 0.969721 w_k1_MSFT_s004
                                 + 0.901759 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.990465 w_k1_ADM_s004
                                 + 1.48834 w_k1_AXP_s004 + 1.1592 w_k1_DIS_s004
                                 + 1.10713 w_k1_FDX_s004
                                 + 1.21101 w_k1_IBM_s004
                                 + 1.30892 w_k1_INTC_s004
                                 + 1.16736 w_k1_KO_s004 + 1.45057 w_k1_MAT_s004
                                 + 1.15656 w_k1_MSFT_s004
                                 + 1.09714 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.06493 w_k1_ADM_s005
                                 + 0.974067 w_k1_AXP_s005
                                 + 1.11429 w_k1_DIS_s005
                                 + 1.01163 w_k1_FDX_s005
                                 + 1.10499 w_k1_IBM_s005
                                 + 1.09798 w_k1_INTC_s005
                                 + 0.973148 w_k1_KO_s005
                                 + 0.681096 w_k1_MAT_s005
                                 + 0.961067 w_k1_MSFT_s005
                                 + 1.01326 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.07134 w_k1_ADM_s005
                                 + 1.16198 w_k1_AXP_s005
                                 + 1.21578 w_k1_DIS_s005
                                 + 1.00714 w_k1_FDX_s005 + 1.7797 w_k1_IBM_s005
                                 + 0.944477 w_k1_INTC_s005
                                 + 1.41908 w_k1_KO_s005 + 1.49461 w_k1_MAT_s005
                                 + 0.899779 w_k1_MSFT_s005
                                 + 1.12485 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.21963 w_k1_ADM_s005
                                 + 1.46004 w_k1_AXP_s005
                                 + 1.89285 w_k1_DIS_s005
                                 + 1.57025 w_k1_FDX_s005
                                 + 0.842264 w_k1_IBM_s005
                                 + 1.31375 w_k1_INTC_s005
                                 + 1.14918 w_k1_KO_s005
                                 + 0.708577 w_k1_MAT_s005
                                 + 1.47238 w_k1_MSFT_s005
                                 + 1.28928 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.00575 w_k1_ADM_s005
                                 + 0.801909 w_k1_AXP_s005
                                 + 0.799844 w_k1_DIS_s005
                                 + 0.875052 w_k1_FDX_s005
                                 + 0.764532 w_k1_IBM_s005
                                 + 0.89578 w_k1_INTC_s005
                                 + 0.858874 w_k1_KO_s005
                                 + 0.680044 w_k1_MAT_s005
                                 + 0.938012 w_k1_MSFT_s005
                                 + 0.826729 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.36127 w_k1_ADM_s005
                                 + 1.30175 w_k1_AXP_s005
                                 + 0.90766 w_k1_DIS_s005
                                 + 1.07694 w_k1_FDX_s005
                                 + 0.969808 w_k1_IBM_s005
                                 + 1.12014 w_k1_INTC_s005
                                 + 1.02368 w_k1_KO_s005
                                 + 0.927213 w_k1_MAT_s005
                                 + 0.910577 w_k1_MSFT_s005
                                 + 1.0146 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.705844 w_k1_ADM_s005
                                 + 0.758678 w_k1_AXP_s005
                                 + 0.78329 w_k1_DIS_s005
                                 + 0.778818 w_k1_FDX_s005
                                 + 1.14022 w_k1_IBM_s005
                                 + 1.08523 w_k1_INTC_s005
                                 + 0.976576 w_k1_KO_s005
                                 + 0.792535 w_k1_MAT_s005
                                 + 0.871114 w_k1_MSFT_s005
                                 + 0.934645 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.18932 w_k1_ADM_s005
                                 + 1.51596 w_k1_AXP_s005
                                 + 1.43796 w_k1_DIS_s005
                                 + 1.25998 w_k1_FDX_s005
                                 + 1.03882 w_k1_IBM_s005
                                 + 1.1758 w_k1_INTC_s005 + 1.12189 w_k1_KO_s005
                                 + 0.973728 w_k1_MAT_s005
                                 + 1.18626 w_k1_MSFT_s005
                                 + 1.09486 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.248 w_k1_ADM_s005
                                 + 0.857153 w_k1_AXP_s005
                                 + 0.661635 w_k1_DIS_s005
                                 + 0.731095 w_k1_FDX_s005
                                 + 1.43463 w_k1_IBM_s005
                                 + 0.759237 w_k1_INTC_s005
                                 + 0.985024 w_k1_KO_s005
                                 + 1.08659 w_k1_MAT_s005
                                 + 0.637717 w_k1_MSFT_s005
                                 + 0.89803 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.765765 w_k1_ADM_s006
                                 + 0.728743 w_k1_AXP_s006
                                 + 0.693433 w_k1_DIS_s006
                                 + 0.694685 w_k1_FDX_s006
                                 + 0.877898 w_k1_IBM_s006
                                 + 0.736685 w_k1_INTC_s006
                                 + 0.888372 w_k1_KO_s006
                                 + 0.830726 w_k1_MAT_s006
                                 + 0.826608 w_k1_MSFT_s006
                                 + 0.851777 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.09001 w_k1_ADM_s006
                                 + 1.0938 w_k1_AXP_s006 + 1.12918 w_k1_DIS_s006
                                 + 1.0341 w_k1_FDX_s006 + 1.05781 w_k1_IBM_s006
                                 + 1.27059 w_k1_INTC_s006
                                 + 1.07985 w_k1_KO_s006 + 1.04099 w_k1_MAT_s006
                                 + 1.13561 w_k1_MSFT_s006
                                 + 1.04548 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.22652 w_k1_ADM_s006
                                 + 1.4069 w_k1_AXP_s006 + 1.67256 w_k1_DIS_s006
                                 + 1.28311 w_k1_FDX_s006
                                 + 1.29085 w_k1_IBM_s006
                                 + 1.30029 w_k1_INTC_s006
                                 + 1.37656 w_k1_KO_s006 + 1.36439 w_k1_MAT_s006
                                 + 1.34509 w_k1_MSFT_s006
                                 + 1.13746 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.842836 w_k1_ADM_s006
                                 + 0.645623 w_k1_AXP_s006
                                 + 0.542608 w_k1_DIS_s006
                                 + 0.66244 w_k1_FDX_s006
                                 + 0.692697 w_k1_IBM_s006
                                 + 0.810235 w_k1_INTC_s006
                                 + 0.75489 w_k1_KO_s006
                                 + 0.488677 w_k1_MAT_s006
                                 + 0.804149 w_k1_MSFT_s006
                                 + 0.843251 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.920337 w_k1_ADM_s006
                                 + 0.908375 w_k1_AXP_s006
                                 + 0.826735 w_k1_DIS_s006
                                 + 0.925818 w_k1_FDX_s006
                                 + 0.678167 w_k1_IBM_s006
                                 + 1.10637 w_k1_INTC_s006
                                 + 0.767412 w_k1_KO_s006
                                 + 0.805123 w_k1_MAT_s006
                                 + 1.12817 w_k1_MSFT_s006
                                 + 0.922868 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.854444 w_k1_ADM_s006
                                 + 1.02563 w_k1_AXP_s006
                                 + 0.967677 w_k1_DIS_s006
                                 + 0.932135 w_k1_FDX_s006
                                 + 1.39691 w_k1_IBM_s006
                                 + 0.992065 w_k1_INTC_s006
                                 + 1.18178 w_k1_KO_s006 + 1.30817 w_k1_MAT_s006
                                 + 0.916324 w_k1_MSFT_s006
                                 + 1.05488 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.891032 w_k1_ADM_s006
                                 + 0.932511 w_k1_AXP_s006
                                 + 0.767968 w_k1_DIS_s006
                                 + 0.856475 w_k1_FDX_s006
                                 + 1.00169 w_k1_IBM_s006
                                 + 0.986023 w_k1_INTC_s006
                                 + 0.850556 w_k1_KO_s006
                                 + 0.981837 w_k1_MAT_s006
                                 + 0.922384 w_k1_MSFT_s006
                                 + 0.960157 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.21444 w_k1_ADM_s006
                                 + 1.11217 w_k1_AXP_s006 + 0.9404 w_k1_DIS_s006
                                 + 1.08889 w_k1_FDX_s006
                                 + 0.854782 w_k1_IBM_s006
                                 + 1.19401 w_k1_INTC_s006
                                 + 0.857991 w_k1_KO_s006
                                 + 0.73054 w_k1_MAT_s006
                                 + 1.1389 w_k1_MSFT_s006 + 1.03835 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.771618 w_k1_ADM_s007
                                 + 0.683439 w_k1_AXP_s007
                                 + 0.563839 w_k1_DIS_s007
                                 + 0.710987 w_k1_FDX_s007
                                 + 0.797269 w_k1_IBM_s007
                                 + 0.709314 w_k1_INTC_s007
                                 + 0.734232 w_k1_KO_s007
                                 + 0.551773 w_k1_MAT_s007
                                 + 0.693225 w_k1_MSFT_s007
                                 + 0.798675 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.02248 w_k1_ADM_s007
                                 + 0.816605 w_k1_AXP_s007
                                 + 0.763467 w_k1_DIS_s007
                                 + 0.963007 w_k1_FDX_s007
                                 + 0.876722 w_k1_IBM_s007
                                 + 0.933059 w_k1_INTC_s007
                                 + 0.869999 w_k1_KO_s007
                                 + 0.538891 w_k1_MAT_s007
                                 + 0.829033 w_k1_MSFT_s007
                                 + 1.00345 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.837498 w_k1_ADM_s007
                                 + 0.845174 w_k1_AXP_s007
                                 + 0.643252 w_k1_DIS_s007
                                 + 0.827517 w_k1_FDX_s007
                                 + 0.926917 w_k1_IBM_s007
                                 + 0.742946 w_k1_INTC_s007
                                 + 0.788017 w_k1_KO_s007
                                 + 0.782544 w_k1_MAT_s007
                                 + 0.71773 w_k1_MSFT_s007
                                 + 0.875898 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.924758 w_k1_ADM_s007
                                 + 0.949702 w_k1_AXP_s007
                                 + 1.34066 w_k1_DIS_s007 + 1.0322 w_k1_FDX_s007
                                 + 1.41118 w_k1_IBM_s007
                                 + 0.900637 w_k1_INTC_s007
                                 + 1.35147 w_k1_KO_s007 + 1.2015 w_k1_MAT_s007
                                 + 1.07776 w_k1_MSFT_s007
                                 + 1.14801 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.20905 w_k1_ADM_s007
                                 + 1.5603 w_k1_AXP_s007 + 1.10189 w_k1_DIS_s007
                                 + 1.11046 w_k1_FDX_s007
                                 + 1.39306 w_k1_IBM_s007
                                 + 0.994666 w_k1_INTC_s007
                                 + 1.09119 w_k1_KO_s007
                                 + 0.870488 w_k1_MAT_s007
                                 + 0.989105 w_k1_MSFT_s007
                                 + 1.10591 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.84476 w_k1_ADM_s007
                                 + 1.08456 w_k1_AXP_s007 + 0.8941 w_k1_DIS_s007
                                 + 0.925728 w_k1_FDX_s007
                                 + 1.35176 w_k1_IBM_s007
                                 + 0.983157 w_k1_INTC_s007
                                 + 1.02829 w_k1_KO_s007
                                 + 0.990192 w_k1_MAT_s007
                                 + 0.900954 w_k1_MSFT_s007
                                 + 1.05654 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.916015 w_k1_ADM_s007
                                 + 0.837741 w_k1_AXP_s007
                                 + 0.848621 w_k1_DIS_s007
                                 + 0.939379 w_k1_FDX_s007
                                 + 0.72832 w_k1_IBM_s007
                                 + 0.886636 w_k1_INTC_s007
                                 + 0.850279 w_k1_KO_s007
                                 + 0.612455 w_k1_MAT_s007
                                 + 0.87988 w_k1_MSFT_s007
                                 + 0.925549 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.12681 w_k1_ADM_s007
                                 + 1.26353 w_k1_AXP_s007
                                 + 1.28464 w_k1_DIS_s007
                                 + 1.28219 w_k1_FDX_s007
                                 + 1.10112 w_k1_IBM_s007
                                 + 1.20521 w_k1_INTC_s007
                                 + 1.21261 w_k1_KO_s007 + 1.01493 w_k1_MAT_s007
                                 + 1.1172 w_k1_MSFT_s007 + 1.11846 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.03672 w_k1_ADM_s010
                                 + 1.26303 w_k1_AXP_s010 + 1.1459 w_k1_DIS_s010
                                 + 1.1859 w_k1_FDX_s010 + 1.01124 w_k1_IBM_s010
                                 + 1.20449 w_k1_INTC_s010
                                 + 1.04677 w_k1_KO_s010 + 1.36583 w_k1_MAT_s010
                                 + 1.32795 w_k1_MSFT_s010
                                 + 1.04148 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.3537 w_k1_ADM_s010
                                 + 1.22291 w_k1_AXP_s010
                                 + 1.58999 w_k1_DIS_s010
                                 + 1.27662 w_k1_FDX_s010
                                 + 1.54616 w_k1_IBM_s010
                                 + 1.21779 w_k1_INTC_s010 + 1.5213 w_k1_KO_s010
                                 + 1.62236 w_k1_MAT_s010
                                 + 1.35388 w_k1_MSFT_s010
                                 + 1.18093 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.777799 w_k1_ADM_s010
                                 + 0.46571 w_k1_AXP_s010
                                 + 0.610628 w_k1_DIS_s010
                                 + 0.651845 w_k1_FDX_s010
                                 + 0.799696 w_k1_IBM_s010
                                 + 0.677797 w_k1_INTC_s010
                                 + 0.824927 w_k1_KO_s010
                                 + 0.71864 w_k1_MAT_s010
                                 + 0.722634 w_k1_MSFT_s010
                                 + 0.835143 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.29659 w_k1_ADM_s010
                                 + 1.43798 w_k1_AXP_s010
                                 + 1.05651 w_k1_DIS_s010
                                 + 1.04205 w_k1_FDX_s010
                                 + 1.46845 w_k1_IBM_s010
                                 + 0.916352 w_k1_INTC_s010
                                 + 1.10904 w_k1_KO_s010 + 1.8661 w_k1_MAT_s010
                                 + 0.867805 w_k1_MSFT_s010
                                 + 1.2265 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.068 w_k1_ADM_s010
                                 + 0.755168 w_k1_AXP_s010
                                 + 0.70116 w_k1_DIS_s010
                                 + 0.938546 w_k1_FDX_s010
                                 + 0.604794 w_k1_IBM_s010
                                 + 0.991974 w_k1_INTC_s010
                                 + 0.767741 w_k1_KO_s010
                                 + 0.679854 w_k1_MAT_s010
                                 + 1.02054 w_k1_MSFT_s010
                                 + 0.924872 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.1747 w_k1_ADM_s010
                                 + 0.823695 w_k1_AXP_s010
                                 + 0.859847 w_k1_DIS_s010
                                 + 0.840125 w_k1_FDX_s010
                                 + 1.15366 w_k1_IBM_s010
                                 + 0.847915 w_k1_INTC_s010
                                 + 1.05082 w_k1_KO_s010 + 1.3592 w_k1_MAT_s010
                                 + 0.903694 w_k1_MSFT_s010
                                 + 0.935341 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.619723 w_k1_ADM_s010
                                 + 0.684103 w_k1_AXP_s010
                                 + 0.573493 w_k1_DIS_s010
                                 + 0.709429 w_k1_FDX_s010
                                 + 1.03682 w_k1_IBM_s010
                                 + 0.722403 w_k1_INTC_s010
                                 + 0.766474 w_k1_KO_s010
                                 + 1.11564 w_k1_MAT_s010
                                 + 0.708406 w_k1_MSFT_s010
                                 + 0.839598 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.10086 w_k1_ADM_s010
                                 + 0.918512 w_k1_AXP_s010
                                 + 1.01883 w_k1_DIS_s010 + 1.0997 w_k1_FDX_s010
                                 + 0.817145 w_k1_IBM_s010
                                 + 0.951572 w_k1_INTC_s010
                                 + 0.780248 w_k1_KO_s010
                                 + 0.824263 w_k1_MAT_s010
                                 + 1.06961 w_k1_MSFT_s010
                                 + 1.01027 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.26981 w_k1_ADM_s011
                                 + 1.27364 w_k1_AXP_s011
                                 + 1.60715 w_k1_DIS_s011
                                 + 1.24746 w_k1_FDX_s011
                                 + 1.44031 w_k1_IBM_s011
                                 + 1.15425 w_k1_INTC_s011
                                 + 1.63167 w_k1_KO_s011 + 1.21968 w_k1_MAT_s011
                                 + 1.22762 w_k1_MSFT_s011
                                 + 1.12514 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.15268 w_k1_ADM_s011
                                 + 1.16396 w_k1_AXP_s011
                                 + 0.899661 w_k1_DIS_s011
                                 + 0.972907 w_k1_FDX_s011
                                 + 0.918627 w_k1_IBM_s011
                                 + 0.986204 w_k1_INTC_s011
                                 + 1.06586 w_k1_KO_s011 + 1.1688 w_k1_MAT_s011
                                 + 0.880847 w_k1_MSFT_s011
                                 + 1.04716 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.988243 w_k1_ADM_s011
                                 + 0.850824 w_k1_AXP_s011
                                 + 0.726842 w_k1_DIS_s011
                                 + 0.797217 w_k1_FDX_s011
                                 + 0.798573 w_k1_IBM_s011
                                 + 0.715447 w_k1_INTC_s011
                                 + 0.893578 w_k1_KO_s011
                                 + 0.93369 w_k1_MAT_s011
                                 + 0.790978 w_k1_MSFT_s011
                                 + 0.859892 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.992493 w_k1_ADM_s011
                                 + 0.695941 w_k1_AXP_s011
                                 + 0.975841 w_k1_DIS_s011
                                 + 0.804495 w_k1_FDX_s011
                                 + 1.1468 w_k1_IBM_s011
                                 + 0.730852 w_k1_INTC_s011
                                 + 1.10688 w_k1_KO_s011
                                 + 0.810301 w_k1_MAT_s011
                                 + 0.884094 w_k1_MSFT_s011
                                 + 0.941533 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.00577 w_k1_ADM_s011
                                 + 1.0195 w_k1_AXP_s011
                                 + 0.980604 w_k1_DIS_s011
                                 + 0.810799 w_k1_FDX_s011
                                 + 1.5032 w_k1_IBM_s011 + 1.2429 w_k1_INTC_s011
                                 + 1.31304 w_k1_KO_s011 + 1.45614 w_k1_MAT_s011
                                 + 0.925158 w_k1_MSFT_s011
                                 + 0.983757 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.962218 w_k1_ADM_s011
                                 + 0.738116 w_k1_AXP_s011
                                 + 1.05302 w_k1_DIS_s011
                                 + 0.936354 w_k1_FDX_s011
                                 + 0.89144 w_k1_IBM_s011
                                 + 1.10925 w_k1_INTC_s011
                                 + 1.25321 w_k1_KO_s011 + 0.51705 w_k1_MAT_s011
                                 + 0.992273 w_k1_MSFT_s011
                                 + 0.975926 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.872082 w_k1_ADM_s011
                                 + 0.575114 w_k1_AXP_s011
                                 + 0.647118 w_k1_DIS_s011
                                 + 0.682059 w_k1_FDX_s011
                                 + 0.720611 w_k1_IBM_s011
                                 + 0.881459 w_k1_INTC_s011
                                 + 0.832372 w_k1_KO_s011
                                 + 0.435923 w_k1_MAT_s011
                                 + 0.773144 w_k1_MSFT_s011
                                 + 0.822544 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.28786 w_k1_ADM_s011
                                 + 0.915757 w_k1_AXP_s011
                                 + 1.11595 w_k1_DIS_s011
                                 + 1.11991 w_k1_FDX_s011
                                 + 0.738932 w_k1_IBM_s011
                                 + 1.09256 w_k1_INTC_s011
                                 + 1.02778 w_k1_KO_s011
                                 + 0.792784 w_k1_MAT_s011
                                 + 1.10168 w_k1_MSFT_s011
                                 + 1.06446 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.24684 w_k1_ADM_s012
                                 + 1.4316 w_k1_AXP_s012 + 1.38362 w_k1_DIS_s012
                                 + 1.22955 w_k1_FDX_s012
                                 + 1.25257 w_k1_IBM_s012
                                 + 1.25479 w_k1_INTC_s012
                                 + 1.29355 w_k1_KO_s012 + 1.83749 w_k1_MAT_s012
                                 + 1.08191 w_k1_MSFT_s012
                                 + 1.23429 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.27434 w_k1_ADM_s012
                                 + 1.4592 w_k1_AXP_s012 + 1.64455 w_k1_DIS_s012
                                 + 1.38839 w_k1_FDX_s012
                                 + 1.21685 w_k1_IBM_s012
                                 + 1.29497 w_k1_INTC_s012
                                 + 1.19797 w_k1_KO_s012 + 1.33918 w_k1_MAT_s012
                                 + 1.25262 w_k1_MSFT_s012
                                 + 1.32667 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.898784 w_k1_ADM_s012
                                 + 0.988963 w_k1_AXP_s012
                                 + 1.34306 w_k1_DIS_s012
                                 + 1.10636 w_k1_FDX_s012
                                 + 0.821412 w_k1_IBM_s012
                                 + 1.22631 w_k1_INTC_s012
                                 + 0.966059 w_k1_KO_s012
                                 + 0.862226 w_k1_MAT_s012
                                 + 1.13094 w_k1_MSFT_s012
                                 + 0.969747 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.830431 w_k1_ADM_s012
                                 + 0.818825 w_k1_AXP_s012
                                 + 0.902634 w_k1_DIS_s012
                                 + 0.925416 w_k1_FDX_s012
                                 + 1.25129 w_k1_IBM_s012
                                 + 1.23449 w_k1_INTC_s012
                                 + 1.02996 w_k1_KO_s012 + 1.26552 w_k1_MAT_s012
                                 + 0.960891 w_k1_MSFT_s012
                                 + 1.09926 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.868547 w_k1_ADM_s012
                                 + 0.704334 w_k1_AXP_s012
                                 + 0.775067 w_k1_DIS_s012
                                 + 0.800258 w_k1_FDX_s012
                                 + 0.806491 w_k1_IBM_s012
                                 + 1.06056 w_k1_INTC_s012
                                 + 0.731293 w_k1_KO_s012
                                 + 0.574704 w_k1_MAT_s012
                                 + 0.822161 w_k1_MSFT_s012
                                 + 0.829795 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.839466 w_k1_ADM_s012
                                 + 0.760381 w_k1_AXP_s012
                                 + 1.00714 w_k1_DIS_s012
                                 + 0.979885 w_k1_FDX_s012
                                 + 1.00369 w_k1_IBM_s012
                                 + 1.08707 w_k1_INTC_s012
                                 + 0.960179 w_k1_KO_s012 + 1.0183 w_k1_MAT_s012
                                 + 1.04881 w_k1_MSFT_s012 + 1.044 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.651698 w_k1_ADM_s012
                                 + 0.6679 w_k1_AXP_s012
                                 + 0.755306 w_k1_DIS_s012
                                 + 0.827501 w_k1_FDX_s012
                                 + 0.559821 w_k1_IBM_s012
                                 + 1.00149 w_k1_INTC_s012 + 0.6945 w_k1_KO_s012
                                 + 0.657444 w_k1_MAT_s012
                                 + 0.932285 w_k1_MSFT_s012
                                 + 0.886078 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.04115 w_k1_ADM_s012
                                 + 1.18549 w_k1_AXP_s012
                                 + 1.31561 w_k1_DIS_s012
                                 + 1.08582 w_k1_FDX_s012
                                 + 1.88587 w_k1_IBM_s012
                                 + 1.13399 w_k1_INTC_s012
                                 + 1.36121 w_k1_KO_s012 + 2.21874 w_k1_MAT_s012
                                 + 1.02538 w_k1_MSFT_s012
                                 + 1.24209 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.799133 w_k1_ADM_s013
                                 + 0.715782 w_k1_AXP_s013
                                 + 0.728629 w_k1_DIS_s013
                                 + 0.856388 w_k1_FDX_s013
                                 + 0.89224 w_k1_IBM_s013
                                 + 0.860313 w_k1_INTC_s013
                                 + 0.780815 w_k1_KO_s013
                                 + 0.947483 w_k1_MAT_s013
                                 + 0.876698 w_k1_MSFT_s013
                                 + 0.985146 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.890526 w_k1_ADM_s013
                                 + 0.774412 w_k1_AXP_s013
                                 + 0.805485 w_k1_DIS_s013
                                 + 0.854486 w_k1_FDX_s013
                                 + 1.1235 w_k1_IBM_s013
                                 + 0.887514 w_k1_INTC_s013
                                 + 0.99543 w_k1_KO_s013
                                 + 0.780293 w_k1_MAT_s013
                                 + 0.883154 w_k1_MSFT_s013
                                 + 0.999212 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.739838 w_k1_ADM_s013
                                 + 0.670871 w_k1_AXP_s013
                                 + 0.532769 w_k1_DIS_s013
                                 + 0.696432 w_k1_FDX_s013
                                 + 0.986141 w_k1_IBM_s013
                                 + 0.844927 w_k1_INTC_s013
                                 + 0.740799 w_k1_KO_s013
                                 + 1.02559 w_k1_MAT_s013
                                 + 0.786976 w_k1_MSFT_s013
                                 + 0.853636 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.79824 w_k1_ADM_s013
                                 + 0.797507 w_k1_AXP_s013
                                 + 0.824811 w_k1_DIS_s013
                                 + 0.881211 w_k1_FDX_s013
                                 + 0.728625 w_k1_IBM_s013
                                 + 0.958998 w_k1_INTC_s013
                                 + 0.738966 w_k1_KO_s013
                                 + 0.836285 w_k1_MAT_s013
                                 + 0.965001 w_k1_MSFT_s013
                                 + 0.950619 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.841078 w_k1_ADM_s013
                                 + 1.06175 w_k1_AXP_s013
                                 + 0.867775 w_k1_DIS_s013
                                 + 0.906994 w_k1_FDX_s013
                                 + 1.17592 w_k1_IBM_s013
                                 + 1.01337 w_k1_INTC_s013
                                 + 0.915969 w_k1_KO_s013
                                 + 1.34686 w_k1_MAT_s013
                                 + 0.954787 w_k1_MSFT_s013
                                 + 1.16104 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.17044 w_k1_ADM_s013
                                 + 1.15445 w_k1_AXP_s013
                                 + 1.13195 w_k1_DIS_s013
                                 + 1.08983 w_k1_FDX_s013
                                 + 0.873683 w_k1_IBM_s013
                                 + 1.07345 w_k1_INTC_s013
                                 + 0.981406 w_k1_KO_s013
                                 + 0.867759 w_k1_MAT_s013
                                 + 1.05262 w_k1_MSFT_s013
                                 + 1.16316 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.09899 w_k1_ADM_s013
                                 + 1.25036 w_k1_AXP_s013
                                 + 1.24392 w_k1_DIS_s013 + 1.2276 w_k1_FDX_s013
                                 + 1.12244 w_k1_IBM_s013
                                 + 1.31038 w_k1_INTC_s013
                                 + 1.12063 w_k1_KO_s013 + 1.00468 w_k1_MAT_s013
                                 + 1.21657 w_k1_MSFT_s013
                                 + 1.36079 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.01442 w_k1_ADM_s013
                                 + 1.14204 w_k1_AXP_s013
                                 + 1.37487 w_k1_DIS_s013 + 1.1547 w_k1_FDX_s013
                                 + 1.55094 w_k1_IBM_s013
                                 + 1.2889 w_k1_INTC_s013 + 1.30597 w_k1_KO_s013
                                 + 1.57122 w_k1_MAT_s013
                                 + 1.18887 w_k1_MSFT_s013
                                 + 1.16567 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.877388 w_k1_ADM_s014
                                 + 0.88135 w_k1_AXP_s014
                                 + 0.821549 w_k1_DIS_s014
                                 + 0.885201 w_k1_FDX_s014
                                 + 0.734178 w_k1_IBM_s014
                                 + 0.87178 w_k1_INTC_s014
                                 + 0.900583 w_k1_KO_s014
                                 + 0.796235 w_k1_MAT_s014
                                 + 0.89011 w_k1_MSFT_s014
                                 + 1.06532 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.23534 w_k1_ADM_s014
                                 + 1.08678 w_k1_AXP_s014
                                 + 0.883945 w_k1_DIS_s014
                                 + 0.990779 w_k1_FDX_s014
                                 + 0.772054 w_k1_IBM_s014
                                 + 0.727218 w_k1_INTC_s014
                                 + 0.851933 w_k1_KO_s014
                                 + 0.935158 w_k1_MAT_s014
                                 + 0.856219 w_k1_MSFT_s014
                                 + 1.11392 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.30335 w_k1_ADM_s014
                                 + 1.17873 w_k1_AXP_s014 + 1.7264 w_k1_DIS_s014
                                 + 1.178 w_k1_FDX_s014 + 1.40378 w_k1_IBM_s014
                                 + 1.16038 w_k1_INTC_s014
                                 + 1.42863 w_k1_KO_s014 + 1.14132 w_k1_MAT_s014
                                 + 1.26526 w_k1_MSFT_s014
                                 + 1.39857 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.03426 w_k1_ADM_s014
                                 + 1.30297 w_k1_AXP_s014
                                 + 0.992102 w_k1_DIS_s014
                                 + 0.944473 w_k1_FDX_s014
                                 + 1.27633 w_k1_IBM_s014
                                 + 0.931195 w_k1_INTC_s014
                                 + 1.13766 w_k1_KO_s014 + 1.78544 w_k1_MAT_s014
                                 + 0.836475 w_k1_MSFT_s014
                                 + 1.13729 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.17565 w_k1_ADM_s014
                                 + 0.950804 w_k1_AXP_s014
                                 + 1.21676 w_k1_DIS_s014
                                 + 1.04266 w_k1_FDX_s014
                                 + 1.06192 w_k1_IBM_s014
                                 + 0.974559 w_k1_INTC_s014
                                 + 1.13335 w_k1_KO_s014
                                 + 0.804138 w_k1_MAT_s014
                                 + 1.07394 w_k1_MSFT_s014 + 1.2026 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.14078 w_k1_ADM_s014
                                 + 1.3854 w_k1_AXP_s014 + 1.43354 w_k1_DIS_s014
                                 + 1.40871 w_k1_FDX_s014
                                 + 0.877207 w_k1_IBM_s014
                                 + 1.06473 w_k1_INTC_s014
                                 + 1.22608 w_k1_KO_s014 + 1.3165 w_k1_MAT_s014
                                 + 1.1889 w_k1_MSFT_s014 + 1.30757 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.80945 w_k1_ADM_s014
                                 + 0.93803 w_k1_AXP_s014
                                 + 0.864417 w_k1_DIS_s014
                                 + 0.809264 w_k1_FDX_s014
                                 + 1.15945 w_k1_IBM_s014
                                 + 0.841418 w_k1_INTC_s014
                                 + 0.963076 w_k1_KO_s014 + 1.2384 w_k1_MAT_s014
                                 + 0.810927 w_k1_MSFT_s014
                                 + 1.02848 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.16116 w_k1_ADM_s014
                                 + 1.2879 w_k1_AXP_s014 + 1.28044 w_k1_DIS_s014
                                 + 1.1959 w_k1_FDX_s014
                                 + 0.835698 w_k1_IBM_s014
                                 + 1.15516 w_k1_INTC_s014
                                 + 1.09604 w_k1_KO_s014
                                 + 0.975962 w_k1_MAT_s014
                                 + 1.14976 w_k1_MSFT_s014
                                 + 1.11006 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.24703 w_k1_ADM_s015
                                 + 1.18652 w_k1_AXP_s015
                                 + 1.31436 w_k1_DIS_s015
                                 + 1.03103 w_k1_FDX_s015
                                 + 1.26971 w_k1_IBM_s015
                                 + 1.14441 w_k1_INTC_s015
                                 + 1.35723 w_k1_KO_s015 + 1.00485 w_k1_MAT_s015
                                 + 1.28 w_k1_MSFT_s015 + 1.09005 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.931381 w_k1_ADM_s015
                                 + 0.866468 w_k1_AXP_s015
                                 + 1.01631 w_k1_DIS_s015
                                 + 0.938822 w_k1_FDX_s015
                                 + 0.941274 w_k1_IBM_s015
                                 + 0.978539 w_k1_INTC_s015
                                 + 1.09808 w_k1_KO_s015 + 1.21738 w_k1_MAT_s015
                                 + 1.29671 w_k1_MSFT_s015 + 1.0106 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.15891 w_k1_ADM_s015
                                 + 0.861551 w_k1_AXP_s015
                                 + 1.0415 w_k1_DIS_s015 + 1.08827 w_k1_FDX_s015
                                 + 0.720256 w_k1_IBM_s015
                                 + 1.15657 w_k1_INTC_s015
                                 + 1.06384 w_k1_KO_s015
                                 + 0.678991 w_k1_MAT_s015
                                 + 1.36781 w_k1_MSFT_s015
                                 + 1.01866 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.37709 w_k1_ADM_s015
                                 + 1.23815 w_k1_AXP_s015
                                 + 1.78735 w_k1_DIS_s015
                                 + 1.31248 w_k1_FDX_s015
                                 + 1.29908 w_k1_IBM_s015
                                 + 1.46433 w_k1_INTC_s015
                                 + 1.55898 w_k1_KO_s015 + 1.47982 w_k1_MAT_s015
                                 + 1.64879 w_k1_MSFT_s015 + 1.197 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.15609 w_k1_ADM_s015
                                 + 1.11013 w_k1_AXP_s015
                                 + 0.85247 w_k1_DIS_s015 + 1.013 w_k1_FDX_s015
                                 + 0.886103 w_k1_IBM_s015
                                 + 1.04678 w_k1_INTC_s015
                                 + 0.944306 w_k1_KO_s015
                                 + 0.995073 w_k1_MAT_s015
                                 + 0.96812 w_k1_MSFT_s015
                                 + 0.944728 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.826245 w_k1_ADM_s015
                                 + 0.709973 w_k1_AXP_s015
                                 + 0.713866 w_k1_DIS_s015
                                 + 0.771607 w_k1_FDX_s015
                                 + 0.81766 w_k1_IBM_s015
                                 + 0.826662 w_k1_INTC_s015
                                 + 0.948258 w_k1_KO_s015
                                 + 0.873222 w_k1_MAT_s015
                                 + 0.927238 w_k1_MSFT_s015
                                 + 0.869058 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.4325 w_k1_ADM_s015
                                 + 1.28683 w_k1_AXP_s015
                                 + 1.17356 w_k1_DIS_s015
                                 + 1.23817 w_k1_FDX_s015
                                 + 0.937729 w_k1_IBM_s015
                                 + 0.924976 w_k1_INTC_s015
                                 + 1.04552 w_k1_KO_s015 + 1.12219 w_k1_MAT_s015
                                 + 1.22979 w_k1_MSFT_s015
                                 + 1.00807 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.962939 w_k1_ADM_s015
                                 + 0.98422 w_k1_AXP_s015
                                 + 1.09605 w_k1_DIS_s015
                                 + 0.910624 w_k1_FDX_s015
                                 + 1.42459 w_k1_IBM_s015
                                 + 0.997749 w_k1_INTC_s015
                                 + 1.28963 w_k1_KO_s015 + 1.69859 w_k1_MAT_s015
                                 + 1.20661 w_k1_MSFT_s015
                                 + 0.978756 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.810174 w_k1_ADM_s016
                                 + 0.937972 w_k1_AXP_s016
                                 + 0.998217 w_k1_DIS_s016
                                 + 0.843755 w_k1_FDX_s016
                                 + 1.3063 w_k1_IBM_s016
                                 + 0.876818 w_k1_INTC_s016
                                 + 0.958245 w_k1_KO_s016
                                 + 1.22523 w_k1_MAT_s016
                                 + 0.856446 w_k1_MSFT_s016
                                 + 0.979948 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.23673 w_k1_ADM_s016
                                 + 1.46391 w_k1_AXP_s016
                                 + 1.45174 w_k1_DIS_s016
                                 + 1.16556 w_k1_FDX_s016
                                 + 1.64892 w_k1_IBM_s016
                                 + 1.03333 w_k1_INTC_s016 + 1.2606 w_k1_KO_s016
                                 + 1.90669 w_k1_MAT_s016
                                 + 1.0121 w_k1_MSFT_s016 + 1.11262 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.16052 w_k1_ADM_s016
                                 + 1.15489 w_k1_AXP_s016
                                 + 1.31786 w_k1_DIS_s016
                                 + 1.17186 w_k1_FDX_s016
                                 + 1.22424 w_k1_IBM_s016
                                 + 1.19871 w_k1_INTC_s016
                                 + 1.28688 w_k1_KO_s016
                                 + 0.929316 w_k1_MAT_s016
                                 + 1.07931 w_k1_MSFT_s016
                                 + 1.04547 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.38848 w_k1_ADM_s016
                                 + 1.84465 w_k1_AXP_s016 + 1.5915 w_k1_DIS_s016
                                 + 1.52125 w_k1_FDX_s016
                                 + 1.14146 w_k1_IBM_s016
                                 + 1.39958 w_k1_INTC_s016
                                 + 1.26315 w_k1_KO_s016 + 1.46555 w_k1_MAT_s016
                                 + 1.24558 w_k1_MSFT_s016
                                 + 1.13637 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.06752 w_k1_ADM_s016
                                 + 1.05655 w_k1_AXP_s016
                                 + 1.41031 w_k1_DIS_s016 + 1.243 w_k1_FDX_s016
                                 + 0.884344 w_k1_IBM_s016
                                 + 1.21542 w_k1_INTC_s016
                                 + 0.936777 w_k1_KO_s016
                                 + 0.697863 w_k1_MAT_s016
                                 + 1.17778 w_k1_MSFT_s016
                                 + 0.994179 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.835739 w_k1_ADM_s016
                                 + 0.701812 w_k1_AXP_s016
                                 + 0.559146 w_k1_DIS_s016
                                 + 0.713033 w_k1_FDX_s016
                                 + 0.762797 w_k1_IBM_s016
                                 + 0.721906 w_k1_INTC_s016
                                 + 0.657323 w_k1_KO_s016
                                 + 0.671262 w_k1_MAT_s016
                                 + 0.653767 w_k1_MSFT_s016
                                 + 0.776004 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.15014 w_k1_ADM_s016
                                 + 1.46493 w_k1_AXP_s016
                                 + 1.06408 w_k1_DIS_s016
                                 + 1.17671 w_k1_FDX_s016
                                 + 0.96771 w_k1_IBM_s016
                                 + 0.99128 w_k1_INTC_s016
                                 + 0.948506 w_k1_KO_s016
                                 + 1.51859 w_k1_MAT_s016
                                 + 0.947749 w_k1_MSFT_s016
                                 + 1.04372 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.945346 w_k1_ADM_s016
                                 + 0.813417 w_k1_AXP_s016
                                 + 0.874333 w_k1_DIS_s016
                                 + 0.932895 w_k1_FDX_s016
                                 + 0.829201 w_k1_IBM_s016
                                 + 0.937299 w_k1_INTC_s016
                                 + 0.790563 w_k1_KO_s016
                                 + 0.664844 w_k1_MAT_s016
                                 + 0.84557 w_k1_MSFT_s016
                                 + 0.866517 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.837416 w_k1_ADM_s017
                                 + 0.942628 w_k1_AXP_s017
                                 + 1.35491 w_k1_DIS_s017
                                 + 1.04101 w_k1_FDX_s017
                                 + 1.15325 w_k1_IBM_s017
                                 + 0.964544 w_k1_INTC_s017
                                 + 1.00811 w_k1_KO_s017 + 1.13757 w_k1_MAT_s017
                                 + 0.983802 w_k1_MSFT_s017
                                 + 1.04245 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.26356 w_k1_ADM_s017
                                 + 1.4089 w_k1_AXP_s017 + 1.54425 w_k1_DIS_s017
                                 + 1.22302 w_k1_FDX_s017
                                 + 1.39111 w_k1_IBM_s017
                                 + 1.11004 w_k1_INTC_s017
                                 + 1.31362 w_k1_KO_s017 + 1.32075 w_k1_MAT_s017
                                 + 1.06302 w_k1_MSFT_s017
                                 + 1.21971 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.963641 w_k1_ADM_s017
                                 + 0.893598 w_k1_AXP_s017
                                 + 0.696609 w_k1_DIS_s017
                                 + 0.807829 w_k1_FDX_s017
                                 + 0.830367 w_k1_IBM_s017
                                 + 0.779563 w_k1_INTC_s017
                                 + 0.840931 w_k1_KO_s017
                                 + 0.801293 w_k1_MAT_s017
                                 + 0.799586 w_k1_MSFT_s017
                                 + 0.983448 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.822162 w_k1_ADM_s017
                                 + 0.794443 w_k1_AXP_s017
                                 + 0.722078 w_k1_DIS_s017
                                 + 0.812475 w_k1_FDX_s017
                                 + 1.36501 w_k1_IBM_s017
                                 + 0.904688 w_k1_INTC_s017
                                 + 0.842147 w_k1_KO_s017
                                 + 0.958586 w_k1_MAT_s017
                                 + 0.833575 w_k1_MSFT_s017
                                 + 0.934676 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.82478 w_k1_ADM_s017
                                 + 0.564886 w_k1_AXP_s017
                                 + 0.729125 w_k1_DIS_s017
                                 + 0.826168 w_k1_FDX_s017
                                 + 0.63239 w_k1_IBM_s017
                                 + 0.773878 w_k1_INTC_s017
                                 + 0.604228 w_k1_KO_s017
                                 + 0.469385 w_k1_MAT_s017
                                 + 0.781787 w_k1_MSFT_s017
                                 + 0.851297 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.821047 w_k1_ADM_s017
                                 + 0.953681 w_k1_AXP_s017
                                 + 1.01199 w_k1_DIS_s017
                                 + 1.05623 w_k1_FDX_s017
                                 + 0.854772 w_k1_IBM_s017
                                 + 0.997517 w_k1_INTC_s017
                                 + 0.913778 w_k1_KO_s017
                                 + 0.773742 w_k1_MAT_s017
                                 + 0.97848 w_k1_MSFT_s017
                                 + 0.984494 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.06402 w_k1_ADM_s017
                                 + 0.985652 w_k1_AXP_s017
                                 + 0.923898 w_k1_DIS_s017
                                 + 0.935144 w_k1_FDX_s017
                                 + 1.23404 w_k1_IBM_s017
                                 + 1.05164 w_k1_INTC_s017
                                 + 1.20642 w_k1_KO_s017 + 1.15664 w_k1_MAT_s017
                                 + 0.88826 w_k1_MSFT_s017
                                 + 1.06076 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.04221 w_k1_ADM_s017
                                 + 1.19963 w_k1_AXP_s017
                                 + 0.863662 w_k1_DIS_s017
                                 + 1.11799 w_k1_FDX_s017
                                 + 0.94359 w_k1_IBM_s017
                                 + 0.962167 w_k1_INTC_s017
                                 + 0.729196 w_k1_KO_s017
                                 + 0.952731 w_k1_MAT_s017
                                 + 0.800223 w_k1_MSFT_s017
                                 + 1.02627 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.08167 w_k1_ADM_s100
                                 + 1.35782 w_k1_AXP_s100
                                 + 1.40303 w_k1_DIS_s100
                                 + 1.11891 w_k1_FDX_s100
                                 + 1.10303 w_k1_IBM_s100
                                 + 0.953951 w_k1_INTC_s100
                                 + 1.22063 w_k1_KO_s100 + 1.34999 w_k1_MAT_s100
                                 + 1.27104 w_k1_MSFT_s100
                                 + 1.06389 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.886772 w_k1_ADM_s100
                                 + 0.862526 w_k1_AXP_s100
                                 + 0.959256 w_k1_DIS_s100
                                 + 0.916973 w_k1_FDX_s100
                                 + 1.17609 w_k1_IBM_s100
                                 + 0.94666 w_k1_INTC_s100
                                 + 1.08422 w_k1_KO_s100 + 1.30424 w_k1_MAT_s100
                                 + 0.967342 w_k1_MSFT_s100
                                 + 0.930574 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.1115 w_k1_ADM_s100
                                 + 1.5735 w_k1_AXP_s100 + 2.18007 w_k1_DIS_s100
                                 + 1.47926 w_k1_FDX_s100
                                 + 1.15087 w_k1_IBM_s100
                                 + 1.28752 w_k1_INTC_s100
                                 + 1.30794 w_k1_KO_s100 + 1.16519 w_k1_MAT_s100
                                 + 1.32088 w_k1_MSFT_s100
                                 + 1.11605 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.02544 w_k1_ADM_s100
                                 + 0.775312 w_k1_AXP_s100
                                 + 0.866608 w_k1_DIS_s100
                                 + 0.906136 w_k1_FDX_s100
                                 + 0.730693 w_k1_IBM_s100
                                 + 0.894035 w_k1_INTC_s100
                                 + 0.925313 w_k1_KO_s100
                                 + 0.635388 w_k1_MAT_s100
                                 + 1.09883 w_k1_MSFT_s100
                                 + 0.932018 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.00321 w_k1_ADM_s100
                                 + 1.2151 w_k1_AXP_s100 + 1.29915 w_k1_DIS_s100
                                 + 1.29994 w_k1_FDX_s100
                                 + 0.847794 w_k1_IBM_s100
                                 + 1.39546 w_k1_INTC_s100
                                 + 1.10162 w_k1_KO_s100 + 1.1795 w_k1_MAT_s100
                                 + 1.45132 w_k1_MSFT_s100 + 1.1747 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.87549 w_k1_ADM_s100
                                 + 1.20962 w_k1_AXP_s100
                                 + 1.37517 w_k1_DIS_s100
                                 + 1.10164 w_k1_FDX_s100
                                 + 1.53589 w_k1_IBM_s100
                                 + 1.11897 w_k1_INTC_s100
                                 + 1.28146 w_k1_KO_s100 + 1.59081 w_k1_MAT_s100
                                 + 1.24208 w_k1_MSFT_s100
                                 + 1.24417 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.03532 w_k1_ADM_s100
                                 + 0.888631 w_k1_AXP_s100
                                 + 1.20412 w_k1_DIS_s100
                                 + 1.10831 w_k1_FDX_s100
                                 + 1.04521 w_k1_IBM_s100
                                 + 1.00258 w_k1_INTC_s100
                                 + 1.13067 w_k1_KO_s100 + 0.84917 w_k1_MAT_s100
                                 + 1.1927 w_k1_MSFT_s100 + 1.1026 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.789271 w_k1_ADM_s100
                                 + 0.650825 w_k1_AXP_s100
                                 + 0.548066 w_k1_DIS_s100
                                 + 0.63585 w_k1_FDX_s100
                                 + 0.988477 w_k1_IBM_s100
                                 + 0.691313 w_k1_INTC_s100
                                 + 0.955207 w_k1_KO_s100
                                 + 0.94341 w_k1_MAT_s100
                                 + 0.782993 w_k1_MSFT_s100
                                 + 0.818227 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.893802 w_k1_ADM_s101
                                 + 0.790873 w_k1_AXP_s101
                                 + 0.712719 w_k1_DIS_s101
                                 + 0.900344 w_k1_FDX_s101
                                 + 0.685711 w_k1_IBM_s101
                                 + 0.841848 w_k1_INTC_s101
                                 + 0.746202 w_k1_KO_s101
                                 + 0.64378 w_k1_MAT_s101
                                 + 0.798303 w_k1_MSFT_s101
                                 + 0.881859 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.961406 w_k1_ADM_s101
                                 + 0.761795 w_k1_AXP_s101
                                 + 1.02214 w_k1_DIS_s101
                                 + 0.974383 w_k1_FDX_s101
                                 + 1.06805 w_k1_IBM_s101
                                 + 1.04633 w_k1_INTC_s101
                                 + 1.21415 w_k1_KO_s101
                                 + 0.835341 w_k1_MAT_s101
                                 + 0.988622 w_k1_MSFT_s101
                                 + 1.01312 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.04169 w_k1_ADM_s101
                                 + 1.50042 w_k1_AXP_s101
                                 + 0.950702 w_k1_DIS_s101
                                 + 1.15798 w_k1_FDX_s101
                                 + 1.01356 w_k1_IBM_s101
                                 + 0.983197 w_k1_INTC_s101
                                 + 1.02825 w_k1_KO_s101 + 1.00704 w_k1_MAT_s101
                                 + 1.05282 w_k1_MSFT_s101
                                 + 1.13815 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.42444 w_k1_ADM_s101
                                 + 1.37589 w_k1_AXP_s101
                                 + 1.61249 w_k1_DIS_s101
                                 + 1.27703 w_k1_FDX_s101
                                 + 1.48229 w_k1_IBM_s101
                                 + 1.24693 w_k1_INTC_s101
                                 + 1.54236 w_k1_KO_s101 + 1.19249 w_k1_MAT_s101
                                 + 1.22189 w_k1_MSFT_s101
                                 + 1.34666 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.949906 w_k1_ADM_s101
                                 + 1.06231 w_k1_AXP_s101
                                 + 0.961059 w_k1_DIS_s101
                                 + 1.00047 w_k1_FDX_s101
                                 + 1.35624 w_k1_IBM_s101
                                 + 1.10517 w_k1_INTC_s101
                                 + 1.24768 w_k1_KO_s101 + 1.22799 w_k1_MAT_s101
                                 + 1.04063 w_k1_MSFT_s101
                                 + 1.03351 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.779683 w_k1_ADM_s101
                                 + 0.706697 w_k1_AXP_s101
                                 + 0.614751 w_k1_DIS_s101
                                 + 0.743336 w_k1_FDX_s101
                                 + 0.81374 w_k1_IBM_s101
                                 + 0.822123 w_k1_INTC_s101
                                 + 0.769029 w_k1_KO_s101 + 0.7121 w_k1_MAT_s101
                                 + 0.643309 w_k1_MSFT_s101
                                 + 0.74036 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.860739 w_k1_ADM_s101
                                 + 1.04627 w_k1_AXP_s101
                                 + 0.994266 w_k1_DIS_s101
                                 + 1.24049 w_k1_FDX_s101
                                 + 0.725187 w_k1_IBM_s101
                                 + 0.839268 w_k1_INTC_s101
                                 + 0.826455 w_k1_KO_s101
                                 + 0.723223 w_k1_MAT_s101
                                 + 1.04376 w_k1_MSFT_s101
                                 + 0.864032 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.681098 w_k1_ADM_s101
                                 + 0.792977 w_k1_AXP_s101
                                 + 0.647477 w_k1_DIS_s101
                                 + 0.791335 w_k1_FDX_s101
                                 + 1.16025 w_k1_IBM_s101
                                 + 0.884276 w_k1_INTC_s101
                                 + 0.922926 w_k1_KO_s101
                                 + 0.896051 w_k1_MAT_s101
                                 + 0.849586 w_k1_MSFT_s101
                                 + 0.767346 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.35146 w_k1_ADM_s102
                                 + 1.42924 w_k1_AXP_s102 + 1.3624 w_k1_DIS_s102
                                 + 1.36017 w_k1_FDX_s102
                                 + 0.866267 w_k1_IBM_s102
                                 + 1.53261 w_k1_INTC_s102
                                 + 1.28247 w_k1_KO_s102 + 1.18256 w_k1_MAT_s102
                                 + 1.53724 w_k1_MSFT_s102
                                 + 1.22055 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.12974 w_k1_ADM_s102
                                 + 1.35833 w_k1_AXP_s102
                                 + 0.879086 w_k1_DIS_s102
                                 + 0.988292 w_k1_FDX_s102
                                 + 1.00916 w_k1_IBM_s102
                                 + 1.1016 w_k1_INTC_s102
                                 + 0.990501 w_k1_KO_s102
                                 + 1.41302 w_k1_MAT_s102
                                 + 1.03706 w_k1_MSFT_s102
                                 + 1.07122 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.811388 w_k1_ADM_s102
                                 + 0.675194 w_k1_AXP_s102
                                 + 0.589371 w_k1_DIS_s102
                                 + 0.718301 w_k1_FDX_s102
                                 + 0.789186 w_k1_IBM_s102
                                 + 0.917669 w_k1_INTC_s102
                                 + 0.851156 w_k1_KO_s102
                                 + 0.761552 w_k1_MAT_s102
                                 + 0.796686 w_k1_MSFT_s102
                                 + 0.824286 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.779513 w_k1_ADM_s102
                                 + 0.755829 w_k1_AXP_s102
                                 + 0.635081 w_k1_DIS_s102
                                 + 0.699033 w_k1_FDX_s102
                                 + 1.05766 w_k1_IBM_s102
                                 + 1.00932 w_k1_INTC_s102
                                 + 0.897484 w_k1_KO_s102
                                 + 0.839709 w_k1_MAT_s102
                                 + 0.893979 w_k1_MSFT_s102
                                 + 0.879732 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.07997 w_k1_ADM_s102
                                 + 1.39531 w_k1_AXP_s102
                                 + 1.06354 w_k1_DIS_s102
                                 + 0.949801 w_k1_FDX_s102
                                 + 1.58854 w_k1_IBM_s102
                                 + 0.965576 w_k1_INTC_s102
                                 + 1.20445 w_k1_KO_s102 + 1.36328 w_k1_MAT_s102
                                 + 0.891317 w_k1_MSFT_s102
                                 + 1.01742 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.997695 w_k1_ADM_s102
                                 + 0.9212 w_k1_AXP_s102 + 0.86365 w_k1_DIS_s102
                                 + 1.09952 w_k1_FDX_s102
                                 + 0.647924 w_k1_IBM_s102
                                 + 0.94531 w_k1_INTC_s102
                                 + 0.74879 w_k1_KO_s102
                                 + 0.617363 w_k1_MAT_s102
                                 + 1.08205 w_k1_MSFT_s102
                                 + 0.852104 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.952099 w_k1_ADM_s102
                                 + 0.926311 w_k1_AXP_s102
                                 + 1.05914 w_k1_DIS_s102
                                 + 0.979042 w_k1_FDX_s102
                                 + 0.883354 w_k1_IBM_s102
                                 + 0.970294 w_k1_INTC_s102
                                 + 0.954261 w_k1_KO_s102
                                 + 0.723513 w_k1_MAT_s102
                                 + 1.17164 w_k1_MSFT_s102 + 1.0238 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.08608 w_k1_ADM_s102
                                 + 1.16658 w_k1_AXP_s102
                                 + 1.31463 w_k1_DIS_s102
                                 + 1.03825 w_k1_FDX_s102
                                 + 1.23728 w_k1_IBM_s102
                                 + 1.06026 w_k1_INTC_s102
                                 + 1.19033 w_k1_KO_s102 + 1.22595 w_k1_MAT_s102
                                 + 1.01747 w_k1_MSFT_s102
                                 + 1.11744 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.03403 w_k1_ADM_s103
                                 + 0.850489 w_k1_AXP_s103
                                 + 0.996064 w_k1_DIS_s103
                                 + 0.921892 w_k1_FDX_s103
                                 + 0.833235 w_k1_IBM_s103
                                 + 1.34742 w_k1_INTC_s103
                                 + 1.07445 w_k1_KO_s103
                                 + 0.923864 w_k1_MAT_s103
                                 + 1.17232 w_k1_MSFT_s103
                                 + 0.915912 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.668439 w_k1_ADM_s103
                                 + 0.859423 w_k1_AXP_s103
                                 + 0.655628 w_k1_DIS_s103
                                 + 0.757015 w_k1_FDX_s103
                                 + 0.655497 w_k1_IBM_s103
                                 + 1.21408 w_k1_INTC_s103
                                 + 0.715761 w_k1_KO_s103
                                 + 0.603144 w_k1_MAT_s103
                                 + 0.901435 w_k1_MSFT_s103
                                 + 0.701155 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.798161 w_k1_ADM_s103
                                 + 0.675531 w_k1_AXP_s103
                                 + 0.646431 w_k1_DIS_s103
                                 + 0.660716 w_k1_FDX_s103
                                 + 0.986179 w_k1_IBM_s103
                                 + 0.939328 w_k1_INTC_s103
                                 + 0.746931 w_k1_KO_s103
                                 + 0.606826 w_k1_MAT_s103
                                 + 0.725537 w_k1_MSFT_s103
                                 + 0.785156 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.58961 w_k1_ADM_s103
                                 + 1.43587 w_k1_AXP_s103 + 1.3881 w_k1_DIS_s103
                                 + 1.15308 w_k1_FDX_s103
                                 + 1.47773 w_k1_IBM_s103
                                 + 1.27729 w_k1_INTC_s103
                                 + 1.45995 w_k1_KO_s103 + 1.27521 w_k1_MAT_s103
                                 + 1.22296 w_k1_MSFT_s103
                                 + 1.25598 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.939513 w_k1_ADM_s103
                                 + 1.02909 w_k1_AXP_s103
                                 + 0.864588 w_k1_DIS_s103
                                 + 0.98421 w_k1_FDX_s103
                                 + 1.18257 w_k1_IBM_s103
                                 + 1.33484 w_k1_INTC_s103
                                 + 1.02268 w_k1_KO_s103
                                 + 0.809421 w_k1_MAT_s103
                                 + 0.909272 w_k1_MSFT_s103
                                 + 0.962314 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.4861 w_k1_ADM_s103
                                 + 1.36162 w_k1_AXP_s103
                                 + 1.06326 w_k1_DIS_s103
                                 + 1.10117 w_k1_FDX_s103
                                 + 1.05719 w_k1_IBM_s103
                                 + 1.03165 w_k1_INTC_s103
                                 + 0.971432 w_k1_KO_s103
                                 + 0.787062 w_k1_MAT_s103
                                 + 1.01126 w_k1_MSFT_s103 + 1.0329 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.59286 w_k1_ADM_s103
                                 + 1.8372 w_k1_AXP_s103 + 1.94336 w_k1_DIS_s103
                                 + 1.6754 w_k1_FDX_s103 + 1.23995 w_k1_IBM_s103
                                 + 1.90376 w_k1_INTC_s103
                                 + 1.28878 w_k1_KO_s103 + 1.01711 w_k1_MAT_s103
                                 + 1.74381 w_k1_MSFT_s103 + 1.2131 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.20018 w_k1_ADM_s103
                                 + 1.04983 w_k1_AXP_s103
                                 + 1.46452 w_k1_DIS_s103
                                 + 1.21338 w_k1_FDX_s103
                                 + 0.973115 w_k1_IBM_s103
                                 + 1.63142 w_k1_INTC_s103
                                 + 1.25324 w_k1_KO_s103 + 0.64748 w_k1_MAT_s103
                                 + 1.53687 w_k1_MSFT_s103
                                 + 1.14588 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.939264 w_k1_ADM_s104
                                 + 0.89928 w_k1_AXP_s104
                                 + 0.928794 w_k1_DIS_s104
                                 + 0.948064 w_k1_FDX_s104
                                 + 1.02353 w_k1_IBM_s104
                                 + 0.762501 w_k1_INTC_s104
                                 + 1.05126 w_k1_KO_s104 + 1.75122 w_k1_MAT_s104
                                 + 0.95562 w_k1_MSFT_s104
                                 + 1.01511 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.52803 w_k1_ADM_s104
                                 + 1.58376 w_k1_AXP_s104
                                 + 1.85851 w_k1_DIS_s104
                                 + 1.50987 w_k1_FDX_s104
                                 + 1.33314 w_k1_IBM_s104
                                 + 1.44447 w_k1_INTC_s104
                                 + 1.52695 w_k1_KO_s104 + 1.89549 w_k1_MAT_s104
                                 + 1.33034 w_k1_MSFT_s104
                                 + 1.20777 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.872112 w_k1_ADM_s104
                                 + 0.771176 w_k1_AXP_s104
                                 + 0.952547 w_k1_DIS_s104
                                 + 0.879112 w_k1_FDX_s104
                                 + 1.2816 w_k1_IBM_s104
                                 + 1.03903 w_k1_INTC_s104
                                 + 1.29219 w_k1_KO_s104 + 1.22082 w_k1_MAT_s104
                                 + 0.968183 w_k1_MSFT_s104
                                 + 0.972895 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.15251 w_k1_ADM_s104
                                 + 1.10739 w_k1_AXP_s104
                                 + 1.09619 w_k1_DIS_s104
                                 + 1.04034 w_k1_FDX_s104
                                 + 1.15742 w_k1_IBM_s104
                                 + 0.839656 w_k1_INTC_s104
                                 + 1.21822 w_k1_KO_s104 + 1.44368 w_k1_MAT_s104
                                 + 1.02061 w_k1_MSFT_s104
                                 + 1.08693 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.855359 w_k1_ADM_s104
                                 + 0.721165 w_k1_AXP_s104
                                 + 1.03442 w_k1_DIS_s104 + 0.893 w_k1_FDX_s104
                                 + 0.888882 w_k1_IBM_s104
                                 + 0.83569 w_k1_INTC_s104
                                 + 1.14934 w_k1_KO_s104 + 1.07624 w_k1_MAT_s104
                                 + 1.10022 w_k1_MSFT_s104
                                 + 0.868801 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.941289 w_k1_ADM_s104
                                 + 0.752381 w_k1_AXP_s104
                                 + 0.680846 w_k1_DIS_s104
                                 + 0.743798 w_k1_FDX_s104
                                 + 0.988144 w_k1_IBM_s104
                                 + 0.662821 w_k1_INTC_s104
                                 + 0.829307 w_k1_KO_s104 + 1.0758 w_k1_MAT_s104
                                 + 0.718559 w_k1_MSFT_s104
                                 + 0.789084 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.841504 w_k1_ADM_s104
                                 + 0.623946 w_k1_AXP_s104
                                 + 0.694728 w_k1_DIS_s104
                                 + 0.797029 w_k1_FDX_s104
                                 + 0.65824 w_k1_IBM_s104
                                 + 0.54673 w_k1_INTC_s104
                                 + 0.786673 w_k1_KO_s104
                                 + 0.861048 w_k1_MAT_s104
                                 + 0.804072 w_k1_MSFT_s104
                                 + 0.84703 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.27316 w_k1_ADM_s104
                                 + 1.03561 w_k1_AXP_s104
                                 + 1.24841 w_k1_DIS_s104
                                 + 1.22888 w_k1_FDX_s104
                                 + 1.01721 w_k1_IBM_s104
                                 + 1.19843 w_k1_INTC_s104
                                 + 1.13188 w_k1_KO_s104 + 0.89755 w_k1_MAT_s104
                                 + 1.20084 w_k1_MSFT_s104
                                 + 1.06342 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.746883 w_k1_ADM_s105
                                 + 0.832458 w_k1_AXP_s105
                                 + 1.05845 w_k1_DIS_s105
                                 + 0.829994 w_k1_FDX_s105
                                 + 1.18027 w_k1_IBM_s105
                                 + 0.833893 w_k1_INTC_s105
                                 + 1.04921 w_k1_KO_s105 + 1.26927 w_k1_MAT_s105
                                 + 0.890684 w_k1_MSFT_s105
                                 + 0.895071 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.524917 w_k1_ADM_s105
                                 + 0.53362 w_k1_AXP_s105
                                 + 0.595495 w_k1_DIS_s105
                                 + 0.687614 w_k1_FDX_s105
                                 + 0.729791 w_k1_IBM_s105
                                 + 0.713325 w_k1_INTC_s105
                                 + 0.667823 w_k1_KO_s105
                                 + 0.574259 w_k1_MAT_s105
                                 + 0.832301 w_k1_MSFT_s105
                                 + 0.799542 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.21109 w_k1_ADM_s105
                                 + 0.999446 w_k1_AXP_s105
                                 + 1.04461 w_k1_DIS_s105
                                 + 1.14723 w_k1_FDX_s105
                                 + 0.826198 w_k1_IBM_s105
                                 + 1.00546 w_k1_INTC_s105
                                 + 0.943926 w_k1_KO_s105
                                 + 0.761875 w_k1_MAT_s105
                                 + 1.07131 w_k1_MSFT_s105
                                 + 1.00267 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.677361 w_k1_ADM_s105
                                 + 0.624343 w_k1_AXP_s105
                                 + 0.748559 w_k1_DIS_s105
                                 + 0.786114 w_k1_FDX_s105
                                 + 0.92366 w_k1_IBM_s105
                                 + 0.817029 w_k1_INTC_s105
                                 + 0.78566 w_k1_KO_s105
                                 + 0.852378 w_k1_MAT_s105
                                 + 0.885562 w_k1_MSFT_s105
                                 + 0.805937 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.15191 w_k1_ADM_s105
                                 + 1.69953 w_k1_AXP_s105
                                 + 1.69413 w_k1_DIS_s105
                                 + 1.40764 w_k1_FDX_s105
                                 + 1.16882 w_k1_IBM_s105
                                 + 1.11482 w_k1_INTC_s105
                                 + 1.18914 w_k1_KO_s105 + 1.56692 w_k1_MAT_s105
                                 + 1.41905 w_k1_MSFT_s105
                                 + 1.19472 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.873485 w_k1_ADM_s105
                                 + 0.915497 w_k1_AXP_s105
                                 + 0.767816 w_k1_DIS_s105
                                 + 0.948311 w_k1_FDX_s105
                                 + 0.977217 w_k1_IBM_s105
                                 + 0.839458 w_k1_INTC_s105
                                 + 0.855397 w_k1_KO_s105
                                 + 0.986661 w_k1_MAT_s105
                                 + 0.956698 w_k1_MSFT_s105
                                 + 1.03438 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.21734 w_k1_ADM_s105
                                 + 1.12432 w_k1_AXP_s105
                                 + 1.12183 w_k1_DIS_s105
                                 + 0.909278 w_k1_FDX_s105
                                 + 1.71278 w_k1_IBM_s105
                                 + 0.799026 w_k1_INTC_s105
                                 + 1.15858 w_k1_KO_s105 + 1.61912 w_k1_MAT_s105
                                 + 0.959856 w_k1_MSFT_s105
                                 + 1.06932 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.512872 w_k1_ADM_s105
                                 + 0.548082 w_k1_AXP_s105
                                 + 0.479084 w_k1_DIS_s105
                                 + 0.57687 w_k1_FDX_s105
                                 + 0.968908 w_k1_IBM_s105
                                 + 0.718326 w_k1_INTC_s105
                                 + 0.728689 w_k1_KO_s105
                                 + 1.01016 w_k1_MAT_s105
                                 + 0.776323 w_k1_MSFT_s105
                                 + 0.719802 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.25219 w_k1_ADM_s106
                                 + 1.03804 w_k1_AXP_s106
                                 + 1.00947 w_k1_DIS_s106
                                 + 1.04686 w_k1_FDX_s106
                                 + 0.866881 w_k1_IBM_s106
                                 + 0.978129 w_k1_INTC_s106
                                 + 0.975729 w_k1_KO_s106
                                 + 0.802224 w_k1_MAT_s106
                                 + 1.08585 w_k1_MSFT_s106
                                 + 1.00226 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.03165 w_k1_ADM_s106
                                 + 0.767417 w_k1_AXP_s106
                                 + 0.692243 w_k1_DIS_s106
                                 + 0.773489 w_k1_FDX_s106
                                 + 0.960528 w_k1_IBM_s106
                                 + 0.904227 w_k1_INTC_s106
                                 + 0.902193 w_k1_KO_s106
                                 + 0.982762 w_k1_MAT_s106
                                 + 0.87224 w_k1_MSFT_s106
                                 + 0.903307 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.924157 w_k1_ADM_s106
                                 + 0.769578 w_k1_AXP_s106
                                 + 1.07037 w_k1_DIS_s106
                                 + 0.948213 w_k1_FDX_s106
                                 + 1.10472 w_k1_IBM_s106
                                 + 0.936527 w_k1_INTC_s106
                                 + 1.2416 w_k1_KO_s106 + 0.98047 w_k1_MAT_s106
                                 + 0.936936 w_k1_MSFT_s106
                                 + 1.12095 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.26298 w_k1_ADM_s106
                                 + 1.54991 w_k1_AXP_s106
                                 + 1.31392 w_k1_DIS_s106 + 1.2632 w_k1_FDX_s106
                                 + 1.18931 w_k1_IBM_s106
                                 + 0.957215 w_k1_INTC_s106
                                 + 1.03764 w_k1_KO_s106 + 1.25999 w_k1_MAT_s106
                                 + 0.993327 w_k1_MSFT_s106
                                 + 1.0783 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.04292 w_k1_ADM_s106
                                 + 0.830103 w_k1_AXP_s106
                                 + 0.767431 w_k1_DIS_s106
                                 + 0.829411 w_k1_FDX_s106
                                 + 1.34737 w_k1_IBM_s106
                                 + 0.873572 w_k1_INTC_s106
                                 + 1.13796 w_k1_KO_s106 + 1.44158 w_k1_MAT_s106
                                 + 0.925273 w_k1_MSFT_s106
                                 + 0.958514 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.49811 w_k1_ADM_s106
                                 + 1.01482 w_k1_AXP_s106 + 1.3416 w_k1_DIS_s106
                                 + 1.25085 w_k1_FDX_s106
                                 + 0.79212 w_k1_IBM_s106
                                 + 0.952472 w_k1_INTC_s106
                                 + 1.15045 w_k1_KO_s106
                                 + 0.866139 w_k1_MAT_s106
                                 + 1.16675 w_k1_MSFT_s106
                                 + 1.16779 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.28091 w_k1_ADM_s106
                                 + 1.19623 w_k1_AXP_s106 + 1.4908 w_k1_DIS_s106
                                 + 1.19532 w_k1_FDX_s106
                                 + 1.54232 w_k1_IBM_s106
                                 + 1.27995 w_k1_INTC_s106
                                 + 1.20294 w_k1_KO_s106 + 1.44557 w_k1_MAT_s106
                                 + 1.11942 w_k1_MSFT_s106
                                 + 1.20068 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.794343 w_k1_ADM_s106
                                 + 0.882833 w_k1_AXP_s106
                                 + 0.805869 w_k1_DIS_s106
                                 + 0.942438 w_k1_FDX_s106
                                 + 0.699602 w_k1_IBM_s106
                                 + 0.931942 w_k1_INTC_s106 + 0.896 w_k1_KO_s106
                                 + 0.781654 w_k1_MAT_s106
                                 + 0.945974 w_k1_MSFT_s106
                                 + 1.00804 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.24449 w_k1_ADM_s107
                                 + 1.03583 w_k1_AXP_s107
                                 + 1.38124 w_k1_DIS_s107
                                 + 1.10086 w_k1_FDX_s107
                                 + 1.31653 w_k1_IBM_s107
                                 + 1.10902 w_k1_INTC_s107
                                 + 1.36665 w_k1_KO_s107 + 1.11172 w_k1_MAT_s107
                                 + 1.1578 w_k1_MSFT_s107 + 1.11177 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.925327 w_k1_ADM_s107
                                 + 0.693885 w_k1_AXP_s107
                                 + 0.652548 w_k1_DIS_s107
                                 + 0.776991 w_k1_FDX_s107
                                 + 0.755469 w_k1_IBM_s107
                                 + 0.596944 w_k1_INTC_s107
                                 + 0.750641 w_k1_KO_s107
                                 + 0.825043 w_k1_MAT_s107
                                 + 0.713829 w_k1_MSFT_s107
                                 + 0.832145 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.26546 w_k1_ADM_s107
                                 + 1.03221 w_k1_AXP_s107
                                 + 0.864751 w_k1_DIS_s107
                                 + 0.999157 w_k1_FDX_s107
                                 + 0.917341 w_k1_IBM_s107
                                 + 0.706635 w_k1_INTC_s107
                                 + 0.879583 w_k1_KO_s107
                                 + 0.887364 w_k1_MAT_s107
                                 + 0.829045 w_k1_MSFT_s107
                                 + 0.940657 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.623186 w_k1_ADM_s107
                                 + 0.597481 w_k1_AXP_s107
                                 + 0.554998 w_k1_DIS_s107
                                 + 0.612316 w_k1_FDX_s107
                                 + 0.908915 w_k1_IBM_s107
                                 + 0.849198 w_k1_INTC_s107
                                 + 0.724334 w_k1_KO_s107
                                 + 0.763506 w_k1_MAT_s107
                                 + 0.685229 w_k1_MSFT_s107
                                 + 0.733697 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.32044 w_k1_ADM_s107
                                 + 0.735991 w_k1_AXP_s107
                                 + 0.894539 w_k1_DIS_s107
                                 + 0.980594 w_k1_FDX_s107
                                 + 0.731065 w_k1_IBM_s107
                                 + 0.97297 w_k1_INTC_s107
                                 + 0.900666 w_k1_KO_s107
                                 + 0.59379 w_k1_MAT_s107
                                 + 0.993035 w_k1_MSFT_s107
                                 + 0.930787 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.899505 w_k1_ADM_s107
                                 + 0.76686 w_k1_AXP_s107
                                 + 0.805628 w_k1_DIS_s107
                                 + 0.823781 w_k1_FDX_s107
                                 + 1.1977 w_k1_IBM_s107
                                 + 0.741222 w_k1_INTC_s107
                                 + 0.970977 w_k1_KO_s107
                                 + 1.35484 w_k1_MAT_s107
                                 + 0.907633 w_k1_MSFT_s107
                                 + 0.860036 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.44245 w_k1_ADM_s107
                                 + 1.21535 w_k1_AXP_s107
                                 + 1.51404 w_k1_DIS_s107
                                 + 1.33374 w_k1_FDX_s107
                                 + 0.929158 w_k1_IBM_s107
                                 + 1.32225 w_k1_INTC_s107
                                 + 1.21562 w_k1_KO_s107 + 1.38682 w_k1_MAT_s107
                                 + 1.3229 w_k1_MSFT_s107 + 1.12618 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.40639 w_k1_ADM_s107
                                 + 1.03287 w_k1_AXP_s107
                                 + 1.03009 w_k1_DIS_s107
                                 + 1.06835 w_k1_FDX_s107 + 1.2477 w_k1_IBM_s107
                                 + 0.936827 w_k1_INTC_s107
                                 + 1.18122 w_k1_KO_s107 + 1.62051 w_k1_MAT_s107
                                 + 1.01955 w_k1_MSFT_s107 + 1.0531 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.23484 w_k1_ADM_s110
                                 + 0.905599 w_k1_AXP_s110
                                 + 1.2179 w_k1_DIS_s110 + 1.19132 w_k1_FDX_s110
                                 + 0.782996 w_k1_IBM_s110
                                 + 1.12619 w_k1_INTC_s110
                                 + 0.832253 w_k1_KO_s110
                                 + 0.633821 w_k1_MAT_s110
                                 + 1.12559 w_k1_MSFT_s110
                                 + 0.976154 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.26526 w_k1_ADM_s110
                                 + 0.805098 w_k1_AXP_s110
                                 + 0.836563 w_k1_DIS_s110
                                 + 0.879961 w_k1_FDX_s110
                                 + 1.10555 w_k1_IBM_s110
                                 + 0.980972 w_k1_INTC_s110
                                 + 0.921834 w_k1_KO_s110
                                 + 1.03563 w_k1_MAT_s110
                                 + 0.904328 w_k1_MSFT_s110
                                 + 1.03935 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.803545 w_k1_ADM_s110
                                 + 0.727403 w_k1_AXP_s110
                                 + 0.656616 w_k1_DIS_s110
                                 + 0.783497 w_k1_FDX_s110
                                 + 0.82728 w_k1_IBM_s110
                                 + 0.866348 w_k1_INTC_s110
                                 + 0.701665 w_k1_KO_s110
                                 + 0.56697 w_k1_MAT_s110
                                 + 0.72259 w_k1_MSFT_s110
                                 + 0.802952 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.909794 w_k1_ADM_s110
                                 + 0.750617 w_k1_AXP_s110
                                 + 0.590876 w_k1_DIS_s110
                                 + 0.767272 w_k1_FDX_s110
                                 + 0.651915 w_k1_IBM_s110
                                 + 0.907152 w_k1_INTC_s110
                                 + 0.662685 w_k1_KO_s110
                                 + 0.504114 w_k1_MAT_s110
                                 + 0.823876 w_k1_MSFT_s110
                                 + 0.857613 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.2039 w_k1_ADM_s110
                                 + 0.860693 w_k1_AXP_s110
                                 + 0.883624 w_k1_DIS_s110
                                 + 0.992511 w_k1_FDX_s110
                                 + 0.727726 w_k1_IBM_s110
                                 + 0.960933 w_k1_INTC_s110
                                 + 0.753203 w_k1_KO_s110
                                 + 0.718183 w_k1_MAT_s110
                                 + 0.982704 w_k1_MSFT_s110
                                 + 0.860262 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.29537 w_k1_ADM_s110
                                 + 1.15943 w_k1_AXP_s110
                                 + 1.38808 w_k1_DIS_s110
                                 + 1.08747 w_k1_FDX_s110
                                 + 1.50967 w_k1_IBM_s110
                                 + 1.1499 w_k1_INTC_s110 + 1.29003 w_k1_KO_s110
                                 + 1.38114 w_k1_MAT_s110
                                 + 1.22677 w_k1_MSFT_s110 + 1.1115 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.38966 w_k1_ADM_s110
                                 + 1.18195 w_k1_AXP_s110
                                 + 1.15498 w_k1_DIS_s110
                                 + 1.16187 w_k1_FDX_s110
                                 + 1.08757 w_k1_IBM_s110
                                 + 1.22263 w_k1_INTC_s110
                                 + 1.08348 w_k1_KO_s110
                                 + 0.849014 w_k1_MAT_s110
                                 + 0.95388 w_k1_MSFT_s110
                                 + 1.04184 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.60809 w_k1_ADM_s110
                                 + 1.38021 w_k1_AXP_s110
                                 + 1.36523 w_k1_DIS_s110
                                 + 1.26195 w_k1_FDX_s110
                                 + 1.08172 w_k1_IBM_s110
                                 + 1.20923 w_k1_INTC_s110
                                 + 1.22297 w_k1_KO_s110 + 1.48574 w_k1_MAT_s110
                                 + 1.23563 w_k1_MSFT_s110
                                 + 1.20588 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.679545 w_k1_ADM_s111
                                 + 0.717504 w_k1_AXP_s111
                                 + 0.852446 w_k1_DIS_s111
                                 + 0.83945 w_k1_FDX_s111
                                 + 0.886988 w_k1_IBM_s111
                                 + 0.882362 w_k1_INTC_s111
                                 + 0.834728 w_k1_KO_s111
                                 + 0.842412 w_k1_MAT_s111
                                 + 0.844388 w_k1_MSFT_s111
                                 + 0.91273 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.15097 w_k1_ADM_s111
                                 + 1.39923 w_k1_AXP_s111
                                 + 1.65853 w_k1_DIS_s111
                                 + 1.47473 w_k1_FDX_s111
                                 + 0.918609 w_k1_IBM_s111
                                 + 1.34196 w_k1_INTC_s111
                                 + 1.25168 w_k1_KO_s111
                                 + 0.957065 w_k1_MAT_s111
                                 + 1.18372 w_k1_MSFT_s111
                                 + 1.17764 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.818503 w_k1_ADM_s111
                                 + 0.865506 w_k1_AXP_s111
                                 + 1.11227 w_k1_DIS_s111
                                 + 1.10645 w_k1_FDX_s111
                                 + 0.674567 w_k1_IBM_s111
                                 + 1.096 w_k1_INTC_s111 + 1.02113 w_k1_KO_s111
                                 + 0.615211 w_k1_MAT_s111
                                 + 1.04537 w_k1_MSFT_s111
                                 + 0.979836 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.01445 w_k1_ADM_s111
                                 + 1.21274 w_k1_AXP_s111
                                 + 1.60067 w_k1_DIS_s111
                                 + 1.20212 w_k1_FDX_s111
                                 + 1.18082 w_k1_IBM_s111
                                 + 1.26941 w_k1_INTC_s111
                                 + 1.25718 w_k1_KO_s111 + 1.21768 w_k1_MAT_s111
                                 + 1.20517 w_k1_MSFT_s111
                                 + 1.20401 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.976966 w_k1_ADM_s111
                                 + 1.01833 w_k1_AXP_s111
                                 + 1.01362 w_k1_DIS_s111
                                 + 1.02972 w_k1_FDX_s111
                                 + 0.913064 w_k1_IBM_s111
                                 + 0.871257 w_k1_INTC_s111
                                 + 0.945813 w_k1_KO_s111
                                 + 1.08297 w_k1_MAT_s111
                                 + 0.826111 w_k1_MSFT_s111
                                 + 1.0503 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.934777 w_k1_ADM_s111
                                 + 1.28423 w_k1_AXP_s111
                                 + 1.27961 w_k1_DIS_s111
                                 + 1.23387 w_k1_FDX_s111
                                 + 0.938847 w_k1_IBM_s111
                                 + 1.11404 w_k1_INTC_s111
                                 + 1.06328 w_k1_KO_s111 + 1.03325 w_k1_MAT_s111
                                 + 1.09784 w_k1_MSFT_s111
                                 + 1.10349 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.817858 w_k1_ADM_s111
                                 + 0.63895 w_k1_AXP_s111
                                 + 0.616329 w_k1_DIS_s111
                                 + 0.691472 w_k1_FDX_s111
                                 + 0.724327 w_k1_IBM_s111
                                 + 0.862539 w_k1_INTC_s111
                                 + 0.72093 w_k1_KO_s111
                                 + 0.516236 w_k1_MAT_s111
                                 + 0.743469 w_k1_MSFT_s111
                                 + 0.844526 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.960409 w_k1_ADM_s111
                                 + 0.923457 w_k1_AXP_s111
                                 + 0.977632 w_k1_DIS_s111
                                 + 0.879636 w_k1_FDX_s111
                                 + 1.34155 w_k1_IBM_s111
                                 + 0.925939 w_k1_INTC_s111
                                 + 1.14927 w_k1_KO_s111 + 1.33819 w_k1_MAT_s111
                                 + 0.895371 w_k1_MSFT_s111
                                 + 1.00215 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.33819 w_k1_ADM_s112
                                 + 1.33966 w_k1_AXP_s112
                                 + 1.56256 w_k1_DIS_s112
                                 + 1.51133 w_k1_FDX_s112
                                 + 0.987318 w_k1_IBM_s112
                                 + 1.32924 w_k1_INTC_s112
                                 + 1.05394 w_k1_KO_s112 + 1.57416 w_k1_MAT_s112
                                 + 1.31636 w_k1_MSFT_s112
                                 + 1.29043 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.3686 w_k1_ADM_s112
                                 + 1.23501 w_k1_AXP_s112
                                 + 0.988807 w_k1_DIS_s112
                                 + 1.10225 w_k1_FDX_s112
                                 + 0.946029 w_k1_IBM_s112
                                 + 1.02043 w_k1_INTC_s112
                                 + 0.969248 w_k1_KO_s112
                                 + 1.16483 w_k1_MAT_s112
                                 + 0.953126 w_k1_MSFT_s112
                                 + 1.05347 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.968024 w_k1_ADM_s112
                                 + 0.95288 w_k1_AXP_s112 + 0.9192 w_k1_DIS_s112
                                 + 0.877523 w_k1_FDX_s112
                                 + 1.28305 w_k1_IBM_s112
                                 + 1.04434 w_k1_INTC_s112
                                 + 1.20562 w_k1_KO_s112 + 1.17198 w_k1_MAT_s112
                                 + 0.994258 w_k1_MSFT_s112
                                 + 0.932361 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.4807 w_k1_ADM_s112
                                 + 1.85088 w_k1_AXP_s112
                                 + 1.45152 w_k1_DIS_s112
                                 + 1.28135 w_k1_FDX_s112
                                 + 1.42372 w_k1_IBM_s112
                                 + 1.29498 w_k1_INTC_s112
                                 + 1.26523 w_k1_KO_s112 + 1.57954 w_k1_MAT_s112
                                 + 1.12673 w_k1_MSFT_s112
                                 + 1.12929 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.30392 w_k1_ADM_s112
                                 + 1.15491 w_k1_AXP_s112
                                 + 1.53283 w_k1_DIS_s112
                                 + 1.20114 w_k1_FDX_s112
                                 + 1.08079 w_k1_IBM_s112
                                 + 1.26541 w_k1_INTC_s112
                                 + 1.24315 w_k1_KO_s112 + 1.43316 w_k1_MAT_s112
                                 + 1.28199 w_k1_MSFT_s112
                                 + 1.07466 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.964374 w_k1_ADM_s112
                                 + 0.745123 w_k1_AXP_s112
                                 + 0.799343 w_k1_DIS_s112
                                 + 0.886988 w_k1_FDX_s112
                                 + 0.684901 w_k1_IBM_s112
                                 + 0.924716 w_k1_INTC_s112
                                 + 0.889258 w_k1_KO_s112
                                 + 0.97295 w_k1_MAT_s112
                                 + 1.04409 w_k1_MSFT_s112
                                 + 0.887548 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.914697 w_k1_ADM_s112
                                 + 0.808748 w_k1_AXP_s112
                                 + 1.06134 w_k1_DIS_s112
                                 + 0.842787 w_k1_FDX_s112
                                 + 0.956049 w_k1_IBM_s112
                                 + 1.09855 w_k1_INTC_s112
                                 + 1.01628 w_k1_KO_s112
                                 + 0.828315 w_k1_MAT_s112
                                 + 0.973346 w_k1_MSFT_s112
                                 + 1.03186 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.939164 w_k1_ADM_s112
                                 + 1.28627 w_k1_AXP_s112
                                 + 1.64849 w_k1_DIS_s112
                                 + 1.08812 w_k1_FDX_s112
                                 + 1.77346 w_k1_IBM_s112
                                 + 0.96834 w_k1_INTC_s112
                                 + 1.37991 w_k1_KO_s112 + 2.02885 w_k1_MAT_s112
                                 + 1.19271 w_k1_MSFT_s112
                                 + 1.28166 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.969156 w_k1_ADM_s113
                                 + 0.771392 w_k1_AXP_s113
                                 + 0.910179 w_k1_DIS_s113
                                 + 0.830755 w_k1_FDX_s113
                                 + 0.955828 w_k1_IBM_s113
                                 + 0.831338 w_k1_INTC_s113
                                 + 0.961747 w_k1_KO_s113
                                 + 0.889049 w_k1_MAT_s113
                                 + 0.961607 w_k1_MSFT_s113
                                 + 0.891866 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.4212 w_k1_ADM_s113
                                 + 1.37498 w_k1_AXP_s113
                                 + 1.63371 w_k1_DIS_s113
                                 + 1.28351 w_k1_FDX_s113
                                 + 1.06708 w_k1_IBM_s113
                                 + 1.23808 w_k1_INTC_s113
                                 + 1.52947 w_k1_KO_s113 + 1.49084 w_k1_MAT_s113
                                 + 1.45409 w_k1_MSFT_s113
                                 + 1.27216 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.16747 w_k1_ADM_s113
                                 + 0.93007 w_k1_AXP_s113
                                 + 0.910836 w_k1_DIS_s113
                                 + 0.963974 w_k1_FDX_s113
                                 + 0.646422 w_k1_IBM_s113
                                 + 0.889596 w_k1_INTC_s113
                                 + 0.888289 w_k1_KO_s113
                                 + 0.727416 w_k1_MAT_s113
                                 + 0.982156 w_k1_MSFT_s113
                                 + 0.869695 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.983841 w_k1_ADM_s113
                                 + 1.14773 w_k1_AXP_s113
                                 + 0.94289 w_k1_DIS_s113
                                 + 0.981122 w_k1_FDX_s113
                                 + 0.977751 w_k1_IBM_s113
                                 + 1.03387 w_k1_INTC_s113
                                 + 1.05485 w_k1_KO_s113 + 1.33947 w_k1_MAT_s113
                                 + 1.08536 w_k1_MSFT_s113
                                 + 1.00603 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.876558 w_k1_ADM_s113
                                 + 0.671993 w_k1_AXP_s113
                                 + 0.535226 w_k1_DIS_s113
                                 + 0.715002 w_k1_FDX_s113
                                 + 0.665276 w_k1_IBM_s113
                                 + 0.70059 w_k1_INTC_s113
                                 + 0.720459 w_k1_KO_s113
                                 + 0.695722 w_k1_MAT_s113
                                 + 0.768196 w_k1_MSFT_s113
                                 + 0.740117 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.820112 w_k1_ADM_s113
                                 + 1.00736 w_k1_AXP_s113
                                 + 0.952464 w_k1_DIS_s113
                                 + 0.888312 w_k1_FDX_s113
                                 + 1.37589 w_k1_IBM_s113
                                 + 0.857962 w_k1_INTC_s113
                                 + 1.10756 w_k1_KO_s113 + 1.38822 w_k1_MAT_s113
                                 + 0.987959 w_k1_MSFT_s113
                                 + 0.934698 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.960759 w_k1_ADM_s113
                                 + 1.51152 w_k1_AXP_s113
                                 + 1.50467 w_k1_DIS_s113
                                 + 1.19434 w_k1_FDX_s113
                                 + 1.53715 w_k1_IBM_s113
                                 + 1.34215 w_k1_INTC_s113
                                 + 1.42948 w_k1_KO_s113 + 1.65806 w_k1_MAT_s113
                                 + 1.23772 w_k1_MSFT_s113
                                 + 1.13708 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.25261 w_k1_ADM_s113
                                 + 1.21289 w_k1_AXP_s113
                                 + 1.33267 w_k1_DIS_s113
                                 + 1.19755 w_k1_FDX_s113
                                 + 0.884348 w_k1_IBM_s113
                                 + 1.06479 w_k1_INTC_s113
                                 + 1.18229 w_k1_KO_s113 + 1.0597 w_k1_MAT_s113
                                 + 1.18574 w_k1_MSFT_s113
                                 + 1.22084 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.785977 w_k1_ADM_s114
                                 + 0.931674 w_k1_AXP_s114
                                 + 0.800189 w_k1_DIS_s114
                                 + 0.813805 w_k1_FDX_s114
                                 + 1.10074 w_k1_IBM_s114
                                 + 0.926927 w_k1_INTC_s114
                                 + 0.897375 w_k1_KO_s114
                                 + 1.13692 w_k1_MAT_s114
                                 + 0.851981 w_k1_MSFT_s114
                                 + 0.877909 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.602585 w_k1_ADM_s114
                                 + 0.858167 w_k1_AXP_s114
                                 + 0.636523 w_k1_DIS_s114
                                 + 0.695978 w_k1_FDX_s114
                                 + 1.25212 w_k1_IBM_s114
                                 + 0.84288 w_k1_INTC_s114
                                 + 1.00446 w_k1_KO_s114 + 1.42826 w_k1_MAT_s114
                                 + 0.783765 w_k1_MSFT_s114
                                 + 0.89837 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.03533 w_k1_ADM_s114
                                 + 1.2901 w_k1_AXP_s114 + 1.26187 w_k1_DIS_s114
                                 + 1.11983 w_k1_FDX_s114
                                 + 0.922917 w_k1_IBM_s114
                                 + 1.06681 w_k1_INTC_s114
                                 + 1.13171 w_k1_KO_s114 + 1.23724 w_k1_MAT_s114
                                 + 1.13384 w_k1_MSFT_s114 + 1.0853 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.91858 w_k1_ADM_s114
                                 + 1.05923 w_k1_AXP_s114
                                 + 1.05041 w_k1_DIS_s114
                                 + 1.06934 w_k1_FDX_s114 + 1.1868 w_k1_IBM_s114
                                 + 0.989609 w_k1_INTC_s114
                                 + 1.1494 w_k1_KO_s114 + 1.25425 w_k1_MAT_s114
                                 + 1.00267 w_k1_MSFT_s114
                                 + 1.01895 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.923953 w_k1_ADM_s114
                                 + 0.950332 w_k1_AXP_s114
                                 + 0.830871 w_k1_DIS_s114
                                 + 0.894313 w_k1_FDX_s114
                                 + 0.713934 w_k1_IBM_s114
                                 + 0.985364 w_k1_INTC_s114
                                 + 0.894009 w_k1_KO_s114
                                 + 0.932129 w_k1_MAT_s114
                                 + 0.904988 w_k1_MSFT_s114
                                 + 0.943282 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.643832 w_k1_ADM_s114
                                 + 0.710384 w_k1_AXP_s114
                                 + 0.529442 w_k1_DIS_s114
                                 + 0.646664 w_k1_FDX_s114
                                 + 0.761355 w_k1_IBM_s114
                                 + 0.736759 w_k1_INTC_s114
                                 + 0.736972 w_k1_KO_s114 + 1.0358 w_k1_MAT_s114
                                 + 0.683328 w_k1_MSFT_s114
                                 + 0.793182 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.966783 w_k1_ADM_s114
                                 + 0.79354 w_k1_AXP_s114
                                 + 0.906545 w_k1_DIS_s114
                                 + 0.951288 w_k1_FDX_s114
                                 + 0.895085 w_k1_IBM_s114
                                 + 0.879483 w_k1_INTC_s114
                                 + 0.900378 w_k1_KO_s114
                                 + 0.858231 w_k1_MAT_s114
                                 + 1.01808 w_k1_MSFT_s114
                                 + 0.896135 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.1034 w_k1_ADM_s114
                                 + 1.34159 w_k1_AXP_s114
                                 + 1.58577 w_k1_DIS_s114
                                 + 1.33005 w_k1_FDX_s114 + 1.418 w_k1_IBM_s114
                                 + 1.29999 w_k1_INTC_s114
                                 + 1.56176 w_k1_KO_s114 + 1.68558 w_k1_MAT_s114
                                 + 1.28889 w_k1_MSFT_s114
                                 + 1.18468 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.957889 w_k1_ADM_s115
                                 + 0.786109 w_k1_AXP_s115
                                 + 0.70404 w_k1_DIS_s115
                                 + 0.810165 w_k1_FDX_s115
                                 + 0.814986 w_k1_IBM_s115
                                 + 0.718506 w_k1_INTC_s115
                                 + 0.887637 w_k1_KO_s115
                                 + 0.694808 w_k1_MAT_s115
                                 + 0.782488 w_k1_MSFT_s115
                                 + 0.883193 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.896892 w_k1_ADM_s115
                                 + 0.77954 w_k1_AXP_s115
                                 + 0.795092 w_k1_DIS_s115
                                 + 0.846255 w_k1_FDX_s115
                                 + 1.12118 w_k1_IBM_s115
                                 + 1.21452 w_k1_INTC_s115 + 1.1918 w_k1_KO_s115
                                 + 0.926262 w_k1_MAT_s115
                                 + 1.05967 w_k1_MSFT_s115
                                 + 0.998902 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.01064 w_k1_ADM_s115
                                 + 1.42798 w_k1_AXP_s115
                                 + 1.32648 w_k1_DIS_s115 + 1.2091 w_k1_FDX_s115
                                 + 1.42129 w_k1_IBM_s115
                                 + 1.30557 w_k1_INTC_s115
                                 + 1.30774 w_k1_KO_s115 + 1.56444 w_k1_MAT_s115
                                 + 1.30498 w_k1_MSFT_s115
                                 + 1.10096 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.91453 w_k1_ADM_s115
                                 + 0.996792 w_k1_AXP_s115
                                 + 1.12483 w_k1_DIS_s115
                                 + 0.963977 w_k1_FDX_s115
                                 + 1.03913 w_k1_IBM_s115
                                 + 0.894743 w_k1_INTC_s115
                                 + 1.09981 w_k1_KO_s115
                                 + 0.918506 w_k1_MAT_s115
                                 + 0.985483 w_k1_MSFT_s115
                                 + 0.917828 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.915834 w_k1_ADM_s115
                                 + 1.05321 w_k1_AXP_s115 + 1.016 w_k1_DIS_s115
                                 + 0.859721 w_k1_FDX_s115
                                 + 1.35543 w_k1_IBM_s115
                                 + 0.978642 w_k1_INTC_s115
                                 + 1.22931 w_k1_KO_s115 + 1.21497 w_k1_MAT_s115
                                 + 0.973419 w_k1_MSFT_s115
                                 + 0.967973 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.31143 w_k1_ADM_s115
                                 + 1.24505 w_k1_AXP_s115
                                 + 1.09242 w_k1_DIS_s115
                                 + 1.12144 w_k1_FDX_s115
                                 + 1.06594 w_k1_IBM_s115
                                 + 1.03166 w_k1_INTC_s115
                                 + 1.10582 w_k1_KO_s115 + 0.78797 w_k1_MAT_s115
                                 + 1.06601 w_k1_MSFT_s115
                                 + 1.00752 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.21642 w_k1_ADM_s115
                                 + 1.57289 w_k1_AXP_s115
                                 + 1.42968 w_k1_DIS_s115
                                 + 1.49489 w_k1_FDX_s115
                                 + 0.928726 w_k1_IBM_s115
                                 + 1.22446 w_k1_INTC_s115
                                 + 1.05966 w_k1_KO_s115 + 1.23943 w_k1_MAT_s115
                                 + 1.30703 w_k1_MSFT_s115 + 1.1758 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.960921 w_k1_ADM_s115
                                 + 1.00751 w_k1_AXP_s115
                                 + 1.01147 w_k1_DIS_s115
                                 + 1.09388 w_k1_FDX_s115
                                 + 0.749118 w_k1_IBM_s115
                                 + 0.954459 w_k1_INTC_s115
                                 + 0.90901 w_k1_KO_s115
                                 + 0.645417 w_k1_MAT_s115
                                 + 1.08887 w_k1_MSFT_s115 + 1.0692 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.10743 w_k1_ADM_s116
                                 + 1.56052 w_k1_AXP_s116
                                 + 1.12631 w_k1_DIS_s116
                                 + 1.21036 w_k1_FDX_s116
                                 + 1.00858 w_k1_IBM_s116
                                 + 1.06316 w_k1_INTC_s116 + 1.1144 w_k1_KO_s116
                                 + 1.55807 w_k1_MAT_s116
                                 + 1.18907 w_k1_MSFT_s116
                                 + 1.02453 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.72794 w_k1_ADM_s116
                                 + 0.842694 w_k1_AXP_s116
                                 + 0.687396 w_k1_DIS_s116
                                 + 0.736052 w_k1_FDX_s116
                                 + 0.946936 w_k1_IBM_s116
                                 + 0.832253 w_k1_INTC_s116
                                 + 0.758788 w_k1_KO_s116
                                 + 1.03595 w_k1_MAT_s116
                                 + 0.812596 w_k1_MSFT_s116
                                 + 0.780315 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.26787 w_k1_ADM_s116
                                 + 1.51777 w_k1_AXP_s116
                                 + 1.64689 w_k1_DIS_s116
                                 + 1.42995 w_k1_FDX_s116
                                 + 0.991112 w_k1_IBM_s116
                                 + 1.11291 w_k1_INTC_s116 + 1.178 w_k1_KO_s116
                                 + 1.49116 w_k1_MAT_s116
                                 + 1.42828 w_k1_MSFT_s116
                                 + 1.22648 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.851019 w_k1_ADM_s116
                                 + 0.985802 w_k1_AXP_s116
                                 + 1.17185 w_k1_DIS_s116
                                 + 1.13837 w_k1_FDX_s116
                                 + 0.787688 w_k1_IBM_s116
                                 + 0.812475 w_k1_INTC_s116
                                 + 1.00959 w_k1_KO_s116
                                 + 0.979044 w_k1_MAT_s116
                                 + 1.08972 w_k1_MSFT_s116
                                 + 1.00015 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.575605 w_k1_ADM_s116
                                 + 0.610808 w_k1_AXP_s116
                                 + 0.532491 w_k1_DIS_s116
                                 + 0.665671 w_k1_FDX_s116
                                 + 0.626067 w_k1_IBM_s116
                                 + 0.764662 w_k1_INTC_s116
                                 + 0.620075 w_k1_KO_s116
                                 + 0.726587 w_k1_MAT_s116
                                 + 0.721167 w_k1_MSFT_s116
                                 + 0.778978 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.821421 w_k1_ADM_s116
                                 + 0.888211 w_k1_AXP_s116
                                 + 0.940234 w_k1_DIS_s116
                                 + 0.885442 w_k1_FDX_s116
                                 + 1.04172 w_k1_IBM_s116
                                 + 0.940488 w_k1_INTC_s116
                                 + 0.944826 w_k1_KO_s116
                                 + 1.35377 w_k1_MAT_s116
                                 + 0.950567 w_k1_MSFT_s116
                                 + 0.952145 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.27545 w_k1_ADM_s116
                                 + 1.57386 w_k1_AXP_s116
                                 + 1.57808 w_k1_DIS_s116
                                 + 1.32948 w_k1_FDX_s116
                                 + 1.49968 w_k1_IBM_s116
                                 + 1.32389 w_k1_INTC_s116
                                 + 1.50082 w_k1_KO_s116 + 1.71354 w_k1_MAT_s116
                                 + 1.40246 w_k1_MSFT_s116 + 1.1109 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.80972 w_k1_ADM_s116
                                 + 1.19203 w_k1_AXP_s116
                                 + 1.03464 w_k1_DIS_s116
                                 + 1.03182 w_k1_FDX_s116
                                 + 1.21794 w_k1_IBM_s116
                                 + 1.00701 w_k1_INTC_s116
                                 + 1.02475 w_k1_KO_s116 + 1.5246 w_k1_MAT_s116
                                 + 1.02091 w_k1_MSFT_s116
                                 + 0.935345 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.10224 w_k1_ADM_s117
                                 + 1.44602 w_k1_AXP_s117
                                 + 1.32582 w_k1_DIS_s117 + 1.3775 w_k1_FDX_s117
                                 + 1.13155 w_k1_IBM_s117
                                 + 1.35998 w_k1_INTC_s117
                                 + 1.09514 w_k1_KO_s117 + 1.01287 w_k1_MAT_s117
                                 + 1.20584 w_k1_MSFT_s117 + 1.0198 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.71845 w_k1_ADM_s117
                                 + 0.686495 w_k1_AXP_s117
                                 + 0.803437 w_k1_DIS_s117
                                 + 0.915689 w_k1_FDX_s117
                                 + 0.642004 w_k1_IBM_s117
                                 + 0.949602 w_k1_INTC_s117
                                 + 0.659969 w_k1_KO_s117
                                 + 0.647584 w_k1_MAT_s117
                                 + 1.01157 w_k1_MSFT_s117
                                 + 0.762064 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.805491 w_k1_ADM_s117
                                 + 0.96863 w_k1_AXP_s117
                                 + 0.853437 w_k1_DIS_s117
                                 + 1.05006 w_k1_FDX_s117
                                 + 0.941554 w_k1_IBM_s117
                                 + 1.33802 w_k1_INTC_s117
                                 + 1.00253 w_k1_KO_s117 + 1.1707 w_k1_MAT_s117
                                 + 1.08406 w_k1_MSFT_s117
                                 + 0.938835 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.0305 w_k1_ADM_s117
                                 + 1.08251 w_k1_AXP_s117
                                 + 1.25737 w_k1_DIS_s117
                                 + 1.08645 w_k1_FDX_s117 + 1.3724 w_k1_IBM_s117
                                 + 1.08608 w_k1_INTC_s117
                                 + 1.24573 w_k1_KO_s117 + 1.62925 w_k1_MAT_s117
                                 + 1.10572 w_k1_MSFT_s117
                                 + 1.09559 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.18932 w_k1_ADM_s117
                                 + 1.82221 w_k1_AXP_s117
                                 + 2.01522 w_k1_DIS_s117
                                 + 1.59131 w_k1_FDX_s117
                                 + 1.30137 w_k1_IBM_s117
                                 + 1.58679 w_k1_INTC_s117
                                 + 1.42354 w_k1_KO_s117 + 1.28533 w_k1_MAT_s117
                                 + 1.41785 w_k1_MSFT_s117
                                 + 1.36122 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.13097 w_k1_ADM_s117
                                 + 1.05871 w_k1_AXP_s117
                                 + 1.27555 w_k1_DIS_s117 + 1.3921 w_k1_FDX_s117
                                 + 0.874197 w_k1_IBM_s117
                                 + 1.19311 w_k1_INTC_s117
                                 + 1.08183 w_k1_KO_s117 + 1.08683 w_k1_MAT_s117
                                 + 1.30335 w_k1_MSFT_s117
                                 + 1.04988 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.806417 w_k1_ADM_s117
                                 + 0.838773 w_k1_AXP_s117
                                 + 0.804302 w_k1_DIS_s117
                                 + 0.817645 w_k1_FDX_s117
                                 + 1.21816 w_k1_IBM_s117
                                 + 0.880463 w_k1_INTC_s117
                                 + 1.10756 w_k1_KO_s117 + 1.32122 w_k1_MAT_s117
                                 + 0.824144 w_k1_MSFT_s117
                                 + 0.806467 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.744554 w_k1_ADM_s117
                                 + 0.898242 w_k1_AXP_s117
                                 + 1.27436 w_k1_DIS_s117
                                 + 1.08242 w_k1_FDX_s117
                                 + 0.909066 w_k1_IBM_s117
                                 + 1.15729 w_k1_INTC_s117 + 1.1835 w_k1_KO_s117
                                 + 1.07181 w_k1_MAT_s117
                                 + 1.28655 w_k1_MSFT_s117
                                 + 1.07346 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0515267855500176 w_k1_ADM_s000
                                 + 0.0247104381584613 w_k1_ADM_s001
                                 + 0.0782349264285415 w_k1_ADM_s002
                                 + 0.0384772246989664 w_k1_ADM_s003
                                 + 0.0299486107020496 w_k1_ADM_s004
                                 + 0.0248868243933675 w_k1_ADM_s005
                                 + 0.0189980810646262 w_k1_ADM_s006
                                 + 0.0171502925482041 w_k1_ADM_s007
                                 + 0.0493176177275494 w_k1_ADM_s010
                                 + 0.0398944847343598 w_k1_ADM_s011
                                 + 0.0144533459673974 w_k1_ADM_s012
                                 + 0.0627249402480897 w_k1_ADM_s013
                                 + 0.0686975263748197 w_k1_ADM_s014
                                 + 0.046716881139371 w_k1_ADM_s015
                                 + 0.0424959987936822 w_k1_ADM_s016
                                 + 0.0440714108172462 w_k1_ADM_s017
                                 + 0.015558136502502 w_k1_ADM_s100
                                 + 0.0156677119492366 w_k1_ADM_s101
                                 + 0.0110470145418629 w_k1_ADM_s102
                                 + 0.00788953900998325 w_k1_ADM_s103
                                 + 0.0165438879672102 w_k1_ADM_s104
                                 + 0.0122219633008516 w_k1_ADM_s105
                                 + 0.0121430162563459 w_k1_ADM_s106
                                 + 0.0369580395051338 w_k1_ADM_s107
                                 + 0.0142932077312284 w_k1_ADM_s110
                                 + 0.0245982126613348 w_k1_ADM_s111
                                 + 0.0486424500733856 w_k1_ADM_s112
                                 + 0.0378956861016902 w_k1_ADM_s113
                                 + 0.0383018238123871 w_k1_ADM_s114
                                 + 0.025044179759656 w_k1_ADM_s115
                                 + 0.0392403324047606 w_k1_ADM_s116
                                 + 0.00894050657496417 w_k1_ADM_s117
                                 + 0.0568156235636293 w_k1_AXP_s000
                                 + 0.0284549562812792 w_k1_AXP_s001
                                 + 0.0775005726602838 w_k1_AXP_s002
                                 + 0.0379422519347819 w_k1_AXP_s003
                                 + 0.033156301177647 w_k1_AXP_s004
                                 + 0.0238278599530818 w_k1_AXP_s005
                                 + 0.019550906303399 w_k1_AXP_s006
                                 + 0.0182301597573 w_k1_AXP_s007
                                 + 0.0467764371211712 w_k1_AXP_s010
                                 + 0.0338333252224301 w_k1_AXP_s011
                                 + 0.0146496394372415 w_k1_AXP_s012
                                 + 0.0646398967816826 w_k1_AXP_s013
                                 + 0.0711144195403075 w_k1_AXP_s014
                                 + 0.0424642213857273 w_k1_AXP_s015
                                 + 0.0443133219184634 w_k1_AXP_s016
                                 + 0.0465379406183462 w_k1_AXP_s017
                                 + 0.0159937628422565 w_k1_AXP_s100
                                 + 0.0161685294076001 w_k1_AXP_s101
                                 + 0.0115387404106246 w_k1_AXP_s102
                                 + 0.0076206791788176 w_k1_AXP_s103
                                 + 0.0145976988764304 w_k1_AXP_s104
                                 + 0.0133667831956792 w_k1_AXP_s105
                                 + 0.011130571003727 w_k1_AXP_s106
                                 + 0.0289834650094516 w_k1_AXP_s107
                                 + 0.0111089173383485 w_k1_AXP_s110
                                 + 0.0266145250844964 w_k1_AXP_s111
                                 + 0.0462886181786615 w_k1_AXP_s112
                                 + 0.0381198025538116 w_k1_AXP_s113
                                 + 0.0442361699476671 w_k1_AXP_s114
                                 + 0.0263127177541742 w_k1_AXP_s115
                                 + 0.0483616038356927 w_k1_AXP_s116
                                 + 0.00996589629313286 w_k1_AXP_s117
                                 + 0.057129410279695 w_k1_DIS_s000
                                 + 0.0268720189203924 w_k1_DIS_s001
                                 + 0.0773118307329224 w_k1_DIS_s002
                                 + 0.0385359704585261 w_k1_DIS_s003
                                 + 0.0305130856968396 w_k1_DIS_s004
                                 + 0.0232622789078358 w_k1_DIS_s005
                                 + 0.0190089731273782 w_k1_DIS_s006
                                 + 0.0174897654674898 w_k1_DIS_s007
                                 + 0.0472291795016663 w_k1_DIS_s010
                                 + 0.0393732401923663 w_k1_DIS_s011
                                 + 0.0171923947399592 w_k1_DIS_s012
                                 + 0.0651165666863828 w_k1_DIS_s013
                                 + 0.0715579232078013 w_k1_DIS_s014
                                 + 0.0456350088838256 w_k1_DIS_s015
                                 + 0.044340900667907 w_k1_DIS_s016
                                 + 0.045950671694513 w_k1_DIS_s017
                                 + 0.0178373828990108 w_k1_DIS_s100
                                 + 0.0157782937169901 w_k1_DIS_s101
                                 + 0.0110337498274526 w_k1_DIS_s102
                                 + 0.00748992523601202 w_k1_DIS_s103
                                 + 0.0166319321072639 w_k1_DIS_s104
                                 + 0.0141728662090035 w_k1_DIS_s105
                                 + 0.0113911240403016 w_k1_DIS_s106
                                 + 0.0321882981899789 w_k1_DIS_s107
                                 + 0.0118613127174426 w_k1_DIS_s110
                                 + 0.0303332392473631 w_k1_DIS_s111
                                 + 0.0487241172114736 w_k1_DIS_s112
                                 + 0.0392081234533978 w_k1_DIS_s113
                                 + 0.0426480248461093 w_k1_DIS_s114
                                 + 0.0248970427836564 w_k1_DIS_s115
                                 + 0.0453311317113941 w_k1_DIS_s116
                                 + 0.0110373073937913 w_k1_DIS_s117
                                 + 0.0528226975495021 w_k1_FDX_s000
                                 + 0.0263382524677564 w_k1_FDX_s001
                                 + 0.073885072759998 w_k1_FDX_s002
                                 + 0.0373394172482151 w_k1_FDX_s003
                                 + 0.0301333252194513 w_k1_FDX_s004
                                 + 0.0228192498221164 w_k1_FDX_s005
                                 + 0.0184660603495569 w_k1_FDX_s006
                                 + 0.0179585063680879 w_k1_FDX_s007
                                 + 0.048010432841886 w_k1_FDX_s010
                                 + 0.0352014244737437 w_k1_FDX_s011
                                 + 0.0159711576119559 w_k1_FDX_s012
                                 + 0.065651819009355 w_k1_FDX_s013
                                 + 0.0668520587563631 w_k1_FDX_s014
                                 + 0.0427899708860593 w_k1_FDX_s015
                                 + 0.0426528378880858 w_k1_FDX_s016
                                 + 0.0456923112495151 w_k1_FDX_s017
                                 + 0.0164716731908001 w_k1_FDX_s100
                                 + 0.0162889199796815 w_k1_FDX_s101
                                 + 0.0106753583230962 w_k1_FDX_s102
                                 + 0.00711368619263594 w_k1_FDX_s103
                                 + 0.0158677706620922 w_k1_FDX_s104
                                 + 0.0131919802556965 w_k1_FDX_s105
                                 + 0.0111910815685772 w_k1_FDX_s106
                                 + 0.0313490539044637 w_k1_FDX_s107
                                 + 0.0116807760211857 w_k1_FDX_s110
                                 + 0.0275208171913514 w_k1_FDX_s111
                                 + 0.0451747022068809 w_k1_FDX_s112
                                 + 0.0359441373792471 w_k1_FDX_s113
                                 + 0.0416666804939335 w_k1_FDX_s114
                                 + 0.0251692219294033 w_k1_FDX_s115
                                 + 0.0444318473543187 w_k1_FDX_s116
                                 + 0.0108863217976775 w_k1_FDX_s117
                                 + 0.0548219439807971 w_k1_IBM_s000
                                 + 0.0250036430905132 w_k1_IBM_s001
                                 + 0.0705142477206271 w_k1_IBM_s002
                                 + 0.0398731471237111 w_k1_IBM_s003
                                 + 0.0295042002507982 w_k1_IBM_s004
                                 + 0.0233959568723212 w_k1_IBM_s005
                                 + 0.0196347948906881 w_k1_IBM_s006
                                 + 0.0190991933327266 w_k1_IBM_s007
                                 + 0.0474325442412401 w_k1_IBM_s010
                                 + 0.0388731815960004 w_k1_IBM_s011
                                 + 0.0164251249269304 w_k1_IBM_s012
                                 + 0.0664729768146534 w_k1_IBM_s013
                                 + 0.0666817284572962 w_k1_IBM_s014
                                 + 0.0435178872734475 w_k1_IBM_s015
                                 + 0.0439564171217421 w_k1_IBM_s016
                                 + 0.0465687119820498 w_k1_IBM_s017
                                 + 0.0169157275916027 w_k1_IBM_s100
                                 + 0.016312409605572 w_k1_IBM_s101
                                 + 0.0108737258896635 w_k1_IBM_s102
                                 + 0.00725894173655734 w_k1_IBM_s103
                                 + 0.0162710675128143 w_k1_IBM_s104
                                 + 0.0140878074672236 w_k1_IBM_s105
                                 + 0.0112375235538146 w_k1_IBM_s106
                                 + 0.0310748998067428 w_k1_IBM_s107
                                 + 0.0121277953588045 w_k1_IBM_s110
                                 + 0.0272186568048783 w_k1_IBM_s111
                                 + 0.0475958159778063 w_k1_IBM_s112
                                 + 0.0367589181559973 w_k1_IBM_s113
                                 + 0.043726736556503 w_k1_IBM_s114
                                 + 0.0248291940152761 w_k1_IBM_s115
                                 + 0.0440770751386611 w_k1_IBM_s116
                                 + 0.0105436976726374 w_k1_IBM_s117
                                 + 0.0521972367380536 w_k1_INTC_s000
                                 + 0.0258555007151258 w_k1_INTC_s001
                                 + 0.070202544925829 w_k1_INTC_s002
                                 + 0.0387546846738058 w_k1_INTC_s003
                                 + 0.0326373937430033 w_k1_INTC_s004
                                 + 0.0232268727663457 w_k1_INTC_s005
                                 + 0.0203567729794059 w_k1_INTC_s006
                                 + 0.0169263327075405 w_k1_INTC_s007
                                 + 0.0461953861774095 w_k1_INTC_s010
                                 + 0.0376728866903955 w_k1_INTC_s011
                                 + 0.0180195533481627 w_k1_INTC_s012
                                 + 0.0697670527688025 w_k1_INTC_s013
                                 + 0.0628793420785324 w_k1_INTC_s014
                                 + 0.044821471423678 w_k1_INTC_s015
                                 + 0.0413939557640622 w_k1_INTC_s016
                                 + 0.0432428368170119 w_k1_INTC_s017
                                 + 0.016441629804851 w_k1_INTC_s100
                                 + 0.0155051256994015 w_k1_INTC_s101
                                 + 0.0114047168993159 w_k1_INTC_s102
                                 + 0.00922213922303742 w_k1_INTC_s103
                                 + 0.0143233594093095 w_k1_INTC_s104
                                 + 0.0120195492958325 w_k1_INTC_s105
                                 + 0.0105426767337154 w_k1_INTC_s106
                                 + 0.0292073811072852 w_k1_INTC_s107
                                 + 0.0122350999196594 w_k1_INTC_s110
                                 + 0.0277292462068252 w_k1_INTC_s111
                                 + 0.0476185458909178 w_k1_INTC_s112
                                 + 0.0356031356087692 w_k1_INTC_s113
                                 + 0.0418752502318048 w_k1_INTC_s114
                                 + 0.024368316379659 w_k1_INTC_s115
                                 + 0.042179178978511 w_k1_INTC_s116
                                 + 0.0115176715768361 w_k1_INTC_s117
                                 + 0.0557572471958572 w_k1_KO_s000
                                 + 0.0266174161111994 w_k1_KO_s001
                                 + 0.0725436382141987 w_k1_KO_s002
                                 + 0.0344538847035942 w_k1_KO_s003
                                 + 0.0315836052074092 w_k1_KO_s004
                                 + 0.0228993998486907 w_k1_KO_s005
                                 + 0.0191419049609981 w_k1_KO_s006
                                 + 0.0182327597919312 w_k1_KO_s007
                                 + 0.04546654839552 w_k1_KO_s010
                                 + 0.0439670993296434 w_k1_KO_s011
                                 + 0.0155441291199195 w_k1_KO_s012
                                 + 0.0627731883797007 w_k1_KO_s013
                                 + 0.0705606961604802 w_k1_KO_s014
                                 + 0.048834481436317 w_k1_KO_s015
                                 + 0.0398880412913506 w_k1_KO_s016
                                 + 0.0431122110112329 w_k1_KO_s017
                                 + 0.0177367020281731 w_k1_KO_s100
                                 + 0.0168391114709711 w_k1_KO_s101
                                 + 0.0110643602621439 w_k1_KO_s102
                                 + 0.00737082145552214 w_k1_KO_s103
                                 + 0.0177157331303949 w_k1_KO_s104
                                 + 0.0129957411587246 w_k1_KO_s105
                                 + 0.0113518483950652 w_k1_KO_s106
                                 + 0.0320500373699458 w_k1_KO_s107
                                 + 0.0111640552509222 w_k1_KO_s110
                                 + 0.028097479968889 w_k1_KO_s111
                                 + 0.0485616806191547 w_k1_KO_s112
                                 + 0.0397879983922853 w_k1_KO_s113
                                 + 0.0451234325440664 w_k1_KO_s114
                                 + 0.0259077805255817 w_k1_KO_s115
                                 + 0.0433564626353552 w_k1_KO_s116
                                 + 0.0108948990100877 w_k1_KO_s117
                                 + 0.0521026732452862 w_k1_MAT_s000
                                 + 0.0284256148956723 w_k1_MAT_s001
                                 + 0.0724453341247094 w_k1_MAT_s002
                                 + 0.0368043657593372 w_k1_MAT_s003
                                 + 0.0323524086638937 w_k1_MAT_s004
                                 + 0.0193583717038545 w_k1_MAT_s005
                                 + 0.0188559638635053 w_k1_MAT_s006
                                 + 0.0152943687753251 w_k1_MAT_s007
                                 + 0.0545971698149291 w_k1_MAT_s010
                                 + 0.034004481441214 w_k1_MAT_s011
                                 + 0.0174918011642146 w_k1_MAT_s012
                                 + 0.0649435503533793 w_k1_MAT_s013
                                 + 0.0728559456538425 w_k1_MAT_s014
                                 + 0.0473782299448954 w_k1_MAT_s015
                                 + 0.0431134769318383 w_k1_MAT_s016
                                 + 0.043051724374193 w_k1_MAT_s017
                                 + 0.0178885700667084 w_k1_MAT_s100
                                 + 0.0141460565254818 w_k1_MAT_s101
                                 + 0.01085196848301 w_k1_MAT_s102
                                 + 0.00584619399054959 w_k1_MAT_s103
                                 + 0.0196963720697199 w_k1_MAT_s104
                                 + 0.0149647331715147 w_k1_MAT_s105
                                 + 0.0113449802817551 w_k1_MAT_s106
                                 + 0.0334817168927373 w_k1_MAT_s107
                                 + 0.0109917196922631 w_k1_MAT_s110
                                 + 0.0272873384380989 w_k1_MAT_s111
                                 + 0.0542832978853062 w_k1_MAT_s112
                                 + 0.041194959572125 w_k1_MAT_s113
                                 + 0.0515085066000668 w_k1_MAT_s114
                                 + 0.0228034215145102 w_k1_MAT_s115
                                 + 0.0557772305489256 w_k1_MAT_s116
                                 + 0.0118646995081673 w_k1_MAT_s117
                                 + 0.0564786356534707 w_k1_MSFT_s000
                                 + 0.0244743537107974 w_k1_MSFT_s001
                                 + 0.0747671450753469 w_k1_MSFT_s002
                                 + 0.0370124844969972 w_k1_MSFT_s003
                                 + 0.0318194560131591 w_k1_MSFT_s004
                                 + 0.0220765338954903 w_k1_MSFT_s005
                                 + 0.0198314879690708 w_k1_MSFT_s006
                                 + 0.0165011569897993 w_k1_MSFT_s007
                                 + 0.0499013216121562 w_k1_MSFT_s010
                                 + 0.0362427467228159 w_k1_MSFT_s011
                                 + 0.0161025773090003 w_k1_MSFT_s012
                                 + 0.067185341291156 w_k1_MSFT_s013
                                 + 0.0649156592670575 w_k1_MSFT_s014
                                 + 0.051135413826663 w_k1_MSFT_s015
                                 + 0.0385148805012867 w_k1_MSFT_s016
                                 + 0.0408307481902946 w_k1_MSFT_s017
                                 + 0.0182503791265999 w_k1_MSFT_s100
                                 + 0.0152445089406748 w_k1_MSFT_s101
                                 + 0.0114801713069385 w_k1_MSFT_s102
                                 + 0.00786820523661473 w_k1_MSFT_s103
                                 + 0.0160745604753382 w_k1_MSFT_s104
                                 + 0.0136991578973437 w_k1_MSFT_s105
                                 + 0.0107977846682748 w_k1_MSFT_s106
                                 + 0.0309545208186559 w_k1_MSFT_s107
                                 + 0.0115848391233547 w_k1_MSFT_s110
                                 + 0.026166024808581 w_k1_MSFT_s111
                                 + 0.0475792785120757 w_k1_MSFT_s112
                                 + 0.0389597971805228 w_k1_MSFT_s113
                                 + 0.0419021240198683 w_k1_MSFT_s114
                                 + 0.0254649869694236 w_k1_MSFT_s115
                                 + 0.0455866770622789 w_k1_MSFT_s116
                                 + 0.0113054696344961 w_k1_MSFT_s117
                                 + 0.05684967075522 w_k1_PG_s000
                                 + 0.0273066672923023 w_k1_PG_s001
                                 + 0.0754290475846234 w_k1_PG_s002
                                 + 0.0334647478302414 w_k1_PG_s003
                                 + 0.0290200418944522 w_k1_PG_s004
                                 + 0.0221843061008308 w_k1_PG_s005
                                 + 0.0189970806662659 w_k1_PG_s006
                                 + 0.0179866529286987 w_k1_PG_s007
                                 + 0.0466974074534163 w_k1_PG_s010
                                 + 0.0365440187953175 w_k1_PG_s011
                                 + 0.0165607683833697 w_k1_PG_s012
                                 + 0.0726159528727238 w_k1_PG_s013
                                 + 0.0767417668060048 w_k1_PG_s014
                                 + 0.0428973322839077 w_k1_PG_s015
                                 + 0.0399197245397804 w_k1_PG_s016
                                 + 0.0463047695900272 w_k1_PG_s017
                                 + 0.0165261416190967 w_k1_PG_s100
                                 + 0.0158270075402903 w_k1_PG_s101
                                 + 0.0109473900394741 w_k1_PG_s102
                                 + 0.00691042650359066 w_k1_PG_s103
                                 + 0.0155863781648954 w_k1_PG_s104
                                 + 0.01297161022577 w_k1_PG_s105
                                 + 0.0113791585140688 w_k1_PG_s106
                                 + 0.0301397209278927 w_k1_PG_s107
                                 + 0.0116511607756793 w_k1_PG_s110
                                 + 0.0278083835222276 w_k1_PG_s111
                                 + 0.045436534357037 w_k1_PG_s112
                                 + 0.0363699196055524 w_k1_PG_s113
                                 + 0.0419678396672801 w_k1_PG_s114
                                 + 0.0245726716443915 w_k1_PG_s115
                                 + 0.0417614273913278 w_k1_PG_s116
                                 + 0.00989527795639705 w_k1_PG_s117 >= 1.11
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
